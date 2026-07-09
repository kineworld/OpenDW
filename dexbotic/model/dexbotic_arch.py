"""Dexbotic top-level model architecture abstractions.

This file is intended to be the architecture index for Dexbotic model families.
The VLA implementation in the full Dexbotic repository lives here already; this
WAM-only slice keeps the new generative/world abstractions small so they can be
ported back beside the existing VLA classes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Mapping

import torch
from torch import nn
from PIL import Image


class DexboticGenerativeModel(nn.Module):
    """
    Base class for Dexbotic models driven by a generative backbone.

    This class is the common contract for WM, WAM, and hybrid VLA+WM models.
    Compared with the VLA-only ``DexboticVLMModel`` in full Dexbotic, these
    models do not necessarily have an LLM, tokenizer, image-token convention, or
    HuggingFace causal-LM output. The shared assumption is instead:

    - a primary generative ``backbone`` exists and can be optimized or inspected;
    - inputs are multi-modal and method-specific;
    - outputs may include video, action, state, latent tensors, and metrics;
    - training exposes ``training_loss(sample) -> (loss, metrics)``;
    - inference exposes ``infer(...) -> dict``.

    Concrete methods such as DW05, DreamZero, or Cosmos3 should inherit one of
    the subclasses below and implement the modality-specific details locally.
    """

    architecture_family = "generative"
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()

    @property
    def backbone(self) -> nn.Module:
        """
        Primary generative backbone for optimization or inspection.

        Examples include a Wan/Cosmos DiT, a MoT wrapper, or a causal world
        model. This intentionally does not prescribe a field name like ``dit``
        or ``vae`` so that unrelated WM/WAM methods can share the interface.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backbone is not defined.")

    @property
    def device(self) -> torch.device:
        if hasattr(self, "_dexbotic_device"):
            return self._dexbotic_device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @device.setter
    def device(self, value) -> None:
        self._dexbotic_device = torch.device(value)

    @property
    def torch_dtype(self) -> torch.dtype:
        if hasattr(self, "_dexbotic_torch_dtype"):
            return self._dexbotic_torch_dtype
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch_dtype.setter
    def torch_dtype(self, value) -> None:
        self._dexbotic_torch_dtype = value

    def build_inputs(self, sample: Mapping[str, Any], **kwargs) -> dict[str, Any]:
        """
        Normalize a raw sample into model-ready inputs.

        Dataset formats vary substantially across WM/WAM families. A concrete
        model may use this hook to move tensors to device, encode prompts,
        compute VAE latents, build causal masks, or split history/current
        windows before loss or inference.
        """
        raise NotImplementedError

    def training_loss(self, sample: Mapping[str, Any], **kwargs):
        """
        Return ``(loss, metrics)`` for one training batch.

        ``loss`` should be a scalar tensor used for backpropagation. ``metrics``
        should be a plain dictionary with scalar values that trainers can log
        without knowing method internals.
        """
        raise NotImplementedError

    def infer(self, *args, **kwargs) -> dict[str, Any]:
        """
        Generate model outputs from method-specific inputs.

        The returned dictionary should use stable semantic keys such as
        ``video``, ``action``, ``state``, ``latent``, or ``metrics`` when those
        outputs exist. Methods can include additional keys as needed.
        """
        raise NotImplementedError

    def get_trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Return parameters that should be optimized by a generic trainer."""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def save_checkpoint(self, *args, **kwargs):
        """Save method-specific model weights or adapter state."""
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        """Load method-specific model weights or adapter state."""
        raise NotImplementedError


class DexboticWorldModel(DexboticGenerativeModel):
    """
    Base class for world models that generate future world state or video.

    WM models may be text/video conditioned, image-to-video, video-to-video, or
    latent dynamics models. They are not required to predict robot actions.
    """

    architecture_family = "wm"
    input_modalities = ("image", "video", "text", "state", "history")
    output_modalities = ("video", "state", "latent", "metrics")

    @property
    def vae_module(self) -> nn.Module:
        """Video VAE used to move between pixel/video space and latent space."""
        vae = getattr(self, "vae", None)
        if vae is None:
            raise AttributeError(f"{self.__class__.__name__} does not define `vae`.")
        return vae

    @staticmethod
    def _check_resize_height_width(height: int, width: int, num_frames: int) -> tuple[int, int, int]:
        """Round video dimensions to the latent-grid constraints used by common video VAEs."""
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        tiled: bool = False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ) -> torch.Tensor:
        """Encode normalized video tensors into latent tensors through ``self.vae``."""
        return self.vae_module.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled: bool = False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ) -> torch.Tensor:
        """Encode a single normalized condition image as one video latent frame."""
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        latents = self.vae_module.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(latents, list):
            latents = latents[0].unsqueeze(0)
        return latents

    def _decode_latents(
        self,
        latents: torch.Tensor,
        tiled: bool = False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ) -> list[Image.Image]:
        """Decode latent video into RGB PIL frames for evaluation or inference output."""
        video_tensor = self.vae_module.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _decode_latents_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent video into a tensor when downstream metrics need tensor space."""
        vae = self.vae_module
        if not hasattr(vae, "single_decode"):
            return vae.decode(latents, device=self.device, tiled=False)
        return vae.single_decode(latents, self.device)

    def infer_video(self, *args, **kwargs):
        """Convenience alias for world models whose primary output is video."""
        return self.infer(*args, **kwargs)


class DexboticWorldActionModel(DexboticWorldModel):
    """
    Base class for world-action models.

    WAMs jointly reason about future observations and robot controls. A method
    may train video and action objectives together, predict actions from a world
    rollout backbone, or use action tokens/registers inside the generative
    model.
    """

    architecture_family = "wam"
    input_modalities = (
        "image",
        "video",
        "text",
        "action",
        "state",
        "proprio",
        "history",
    )
    output_modalities = ("video", "action", "state", "latent", "metrics")

    @torch.no_grad()
    def encode_prompt(self, prompt: str | Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts through a method-provided tokenizer and text encoder."""
        text_encoder = getattr(self, "text_encoder", None)
        tokenizer = getattr(self, "tokenizer", None)
        if text_encoder is None or tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Provide cached `context/context_mask` or enable text encoder loading."
            )
        ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = text_encoder(ids, mask)

        # Wan-style text encoders return padded embeddings; zero padded tokens so
        # downstream cross-attention sees a clean cached-context equivalent.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, valid_len in enumerate(seq_lens):
            prompt_emb[i, valid_len:] = 0
        return prompt_emb.to(device=self.device), torch.ones_like(mask)

    @staticmethod
    def _is_training_loss_step() -> bool:
        """Return whether code currently runs under a differentiable training step."""
        return torch.is_grad_enabled()

    def _random_condition_drop(self, drop_prob: float) -> bool:
        """Shared CFG-style condition dropout synchronized across distributed ranks."""
        drop_prob = float(drop_prob)
        if drop_prob <= 0.0 or not self._is_training_loss_step():
            return False
        if drop_prob >= 1.0:
            return True
        draw = torch.rand((), device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(draw, src=0)
        return bool(draw.item() < drop_prob)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project proprio/state into a condition token and append it to text context."""
        proprio_encoder = getattr(self, "proprio_encoder", None)
        proprio_dim = getattr(self, "proprio_dim", None)
        if proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if proprio_dim is None or proprio.shape[1] != proprio_dim:
            raise ValueError(f"`proprio` last dim must be {proprio_dim}, got {proprio.shape[1]}")

        proprio_token = proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    def _add_condition_drop_metrics(self, loss_dict: dict[str, float], inputs: dict[str, Any]) -> dict[str, float]:
        """Add generic condition-drop metrics to a method's loss dictionary."""
        if getattr(self, "proprio_encoder", None) is not None and "proprio_dropped" in inputs:
            loss_dict["proprio_dropped"] = float(bool(inputs["proprio_dropped"]))
        return loss_dict

    def _select_proprio_index(
        self,
        *,
        num_video_frames: int,
        action_horizon: int,
        latent_frames: int,
    ) -> int:
        """Select the timestep used when injecting proprio as a condition token."""
        return 0

    def infer_action(self, *args, **kwargs):
        """Generate an action sequence or action distribution."""
        raise NotImplementedError


class DexboticWorldVLAModel(DexboticGenerativeModel):
    """
    Base class for hybrid models that combine VLA and world-model components.

    This family covers models such as DW05-TV where a VLM/action expert is
    trained together with a world model, connected through projectors or shared
    latent states. It is deliberately separate from pure WAM so that pure world
    models are not forced to carry VLM assumptions.
    """

    architecture_family = "vla_wm"
    input_modalities = (
        "image",
        "video",
        "text",
        "token",
        "action",
        "state",
        "history",
    )
    output_modalities = ("token", "video", "action", "state", "latent", "metrics")

    @property
    def vlm_backbone(self) -> nn.Module:
        """Language/vision-language backbone used by the hybrid model."""
        raise NotImplementedError

    @property
    def world_backbone(self) -> nn.Module:
        """Generative world backbone used by the hybrid model."""
        raise NotImplementedError

    @property
    def backbone(self) -> nn.Module:
        return self.world_backbone


__all__ = [
    "DexboticGenerativeModel",
    "DexboticWorldActionModel",
    "DexboticWorldModel",
    "DexboticWorldVLAModel",
]
