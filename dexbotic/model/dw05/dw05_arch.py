"""DW05 world-action model architecture.

This module is the Dexbotic-style method entrypoint for DW05.  It keeps the
method-specific config and concrete WAM implementation under the method package,
while the generic WAM contract lives in :mod:`dexbotic.model.dexbotic_arch`.
Runtime concerns such as local model-cache roots are intentionally configured by
the experiment layer instead of being baked into this architecture module.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dexbotic.model.dexbotic_arch import DexboticWorldActionModel
from dexbotic.model.dw05.modules.action_expert import ActionDiT
from dexbotic.model.dw05.modules.mot_backbone import MoT
from dexbotic.model.modules.wan22.helpers.loader import load_wan22_ti2v_5b_components
from dexbotic.model.modules.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


logger = logging.getLogger(__name__)


WAN22_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"


def _to_plain_dict(value: Any, *, name: str, default: dict | None = None) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        value = {} if default is None else dict(default)
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def _parse_dw05_kwargs(video_dit_config, action_dit_config, video_scheduler, action_scheduler, loss):
    video_dit_config = _to_plain_dict(video_dit_config, name="video_dit_config")
    action_dit_config = _to_plain_dict(action_dit_config, name="action_dit_config")
    video_scheduler = _to_plain_dict(video_scheduler, name="video_scheduler")
    action_scheduler = _to_plain_dict(action_scheduler, name="action_scheduler")
    loss = _to_plain_dict(loss, name="loss")

    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required - set(action_scheduler.keys())
    if missing:
        raise ValueError(f"`action_scheduler` missing required keys: {sorted(missing)}.")
    return video_dit_config, action_dit_config, video_scheduler, action_scheduler, loss


def create_dw05(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    action_drop_prob: float = 0.5,
    num_hist_frames: int = 1,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = False,
    proprio_dim: int | None = 32,
    proprio_drop_prob: float = 0.0,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build the DW05 action-MoT WAM method."""
    if int(num_hist_frames) != 1:
        raise ValueError("DW05 action-MoT uses exactly one history frame.")

    video_dit_config, action_dit_config, video_scheduler, action_scheduler, loss = (
        _parse_dw05_kwargs(
            video_dit_config,
            action_dit_config,
            video_scheduler,
            action_scheduler,
            loss,
        )
    )
    from dexbotic.model.dw05 import DW05WorldActionModel

    return DW05WorldActionModel.from_wan22_pretrained(
        num_hist_frames=1,
        action_drop_prob=float(action_drop_prob),
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        proprio_drop_prob=float(proprio_drop_prob),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


class DW05Wan22WorldActionModel(DexboticWorldActionModel):
    """DW05 Wan2.2 world-action architecture.

    This class owns the DW05-specific module graph and shared Wan2.2 data-flow
    helpers: Wan video expert, ActionDiT expert, MoT attention backbone, VAE,
    text context, proprio context injection, and video/action flow schedulers.
    Concrete DW05 variants inherit it and override only the parts that change,
    such as history window construction or MoT attention layout.
    """

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        proprio_drop_prob: float = 0.0,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.proprio_drop_prob = float(proprio_drop_prob)
        if self.proprio_drop_prob < 0.0 or self.proprio_drop_prob > 1.0:
            raise ValueError(f"`proprio_drop_prob` must be in [0, 1], got {self.proprio_drop_prob}.")

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @property
    def backbone(self) -> nn.Module:
        """Return the generative mixture-of-transformers backbone used by DW05."""
        return self.mot

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = WAN22_MODEL_ID,
        tokenizer_model_id: str = WAN22_TOKENIZER_MODEL_ID,
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        proprio_drop_prob: float = 0.0,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        """Build DW05's Wan2.2 video expert, ActionDiT expert, VAE, and MoT backbone."""
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for DW05Wan22WorldActionModel.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for DW05.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            proprio_drop_prob=proprio_drop_prob,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def configure_trainable_modules(self) -> None:
        """Freeze DW05 support modules and leave only trainable adapters/backbone active."""
        self.eval()
        self.requires_grad_(False)
        self.dit.train()
        self.dit.requires_grad_(True)
        extra_modules_fn = getattr(self, "extra_trainable_modules", None)
        if callable(extra_modules_fn):
            for module in extra_modules_fn():
                module.train()
                module.requires_grad_(True)
        extra_params_fn = getattr(self, "extra_trainable_parameters", None)
        if callable(extra_params_fn):
            for param in extra_params_fn():
                param.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)

    def get_trainable_parameters(self):
        """Return DW05 parameters optimized by the generative trainer."""
        trainable_params = list(self.dit.parameters())
        extra_modules_fn = getattr(self, "extra_trainable_modules", None)
        if callable(extra_modules_fn):
            for module in extra_modules_fn():
                trainable_params.extend(list(module.parameters()))
        extra_params_fn = getattr(self, "extra_trainable_parameters", None)
        if callable(extra_params_fn):
            trainable_params.extend(list(extra_params_fn()))
        if self.proprio_encoder is not None:
            trainable_params.extend(list(self.proprio_encoder.parameters()))
        return trainable_params

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build DW05's base MoT attention mask for one clean frame plus noisy action."""
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _prepare_training_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        num_video_frames: int,
        action_horizon: int,
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Move text context to device and optionally append one proprio token.

        DW05 variants choose the proprio timestep by overriding
        :meth:`_select_proprio_index`.  Keeping this here lets core, history, and
        action-MoT share the same context preparation while preserving their own
        temporal alignment rules.
        """
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio_dropped = False
        if self.proprio_encoder is None:
            return context, context_mask, proprio_dropped

        if proprio is None:
            raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
        if proprio.ndim != 3:
            raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
        if proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")

        proprio_index = int(
            self._select_proprio_index(
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                latent_frames=latent_frames,
            )
        )
        if proprio_index < 0 or proprio_index >= proprio.shape[1]:
            raise ValueError(
                f"Selected proprio index {proprio_index} is out of range for proprio sequence length {proprio.shape[1]}."
            )

        proprio_dropped = self._random_condition_drop(self.proprio_drop_prob)
        if not proprio_dropped:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio[:, proprio_index, :].to(device=self.device, dtype=self.torch_dtype),
            )
        return context, context_mask, proprio_dropped

    def _normalize_single_infer_image(self, image: torch.Tensor, *, name: str = "input_image") -> torch.Tensor:
        """Normalize a single inference condition image to DW05 device/dtype."""
        if image is None:
            raise ValueError(f"`{name}` must be provided.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"`{name}` must have shape [1,3,H,W] or [3,H,W], got {tuple(image.shape)}.")
        return image.to(device=self.device, dtype=self.torch_dtype)

    def _normalize_infer_proprio(
        self,
        proprio: Optional[torch.Tensor],
        *,
        strict: bool = False,
    ) -> Optional[torch.Tensor]:
        """Validate optional inference proprio and move it to DW05 device/dtype."""
        if proprio is None:
            return None
        if self.proprio_dim is None:
            message = "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
            if strict:
                raise ValueError(message)
            logger.warning("`proprio` was provided but `proprio_dim=None`; dropping proprio for inference.")
            return None
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 2 and proprio.shape[0] == 1:
            pass
        else:
            raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        return proprio.to(device=self.device, dtype=self.torch_dtype)

    def _normalize_infer_action_condition(
        self,
        action: Optional[torch.Tensor],
        *,
        batch_size: Optional[int] = 1,
        action_horizon: Optional[int] = None,
        name: str = "action",
    ) -> Optional[torch.Tensor]:
        """Validate an optional GT action condition used by DW05 inference."""
        if action is None:
            return None
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"`{name}` must have shape [B,T,D] or [T,D], got {tuple(action.shape)}")
        if batch_size is not None and action.shape[0] != int(batch_size):
            raise ValueError(
                f"`{name}` batch must be {int(batch_size)}, got {action.shape[0]} for shape {tuple(action.shape)}"
            )
        if action_horizon is not None and action.shape[1] != int(action_horizon):
            raise ValueError(
                f"`{name}` horizon must be {int(action_horizon)}, got {action.shape[1]} for shape {tuple(action.shape)}"
            )
        if action.shape[2] != self.action_expert.action_dim:
            raise ValueError(f"`{name}` last dim must be {self.action_expert.action_dim}, got {action.shape[2]}")
        return action.to(device=self.device, dtype=self.torch_dtype)

    def _prepare_infer_context(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        strict_proprio: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare prompt/cached text context and optional proprio for DW05 inference."""
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio = self._normalize_infer_proprio(proprio, strict=strict_proprio)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    def _sample_training_flow(
        self,
        clean: torch.Tensor,
        scheduler: WanContinuousFlowMatchScheduler,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample DW05 flow-matching noise, timestep, noisy input, and target."""
        noise = torch.randn_like(clean)
        timestep = scheduler.sample_training_t(
            batch_size=clean.shape[0],
            device=self.device,
            dtype=clean.dtype,
        )
        noisy = scheduler.add_noise(clean, noise, timestep)
        target = scheduler.training_target(clean, noise, timestep)
        return noise, timestep, noisy, target

    @staticmethod
    def _restore_clean_latents(
        noisy_latents: torch.Tensor,
        clean_latents: Optional[torch.Tensor],
        clean_latent_count: Optional[int] = None,
    ) -> torch.Tensor:
        """Overwrite the clean condition prefix after adding diffusion noise."""
        if clean_latents is None:
            return noisy_latents
        if clean_latent_count is None:
            clean_latent_count = int(clean_latents.shape[2])
        noisy_latents[:, :, :clean_latent_count] = clean_latents[:, :, :clean_latent_count]
        return noisy_latents

    def _run_mot(
        self,
        *,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_kwargs: Optional[dict[str, Any]] = None,
        action_tokens: Optional[torch.Tensor] = None,
        action_freqs: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
        action_t_mod: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run DW05's Wan/ActionDiT experts through the MoT backbone."""
        if action_tokens is None:
            action_tokens = action_pre["tokens"]
        if action_freqs is None:
            action_freqs = action_pre["freqs"]
        if action_context is None:
            action_context = action_pre["context"]
        if action_context_mask is None:
            action_context_mask = action_pre["context_mask"]
        if action_t_mod is None:
            action_t_mod = action_pre["t_mod"]

        if attention_mask is None:
            kwargs = dict(attention_mask_kwargs or {})
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
                **kwargs,
            )

        return self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_freqs,
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_context,
                    "mask": action_context_mask,
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_t_mod,
            },
        )

    def _prefill_video_cache(self, video_pre: dict[str, Any], attention_mask: torch.Tensor):
        """Prefill MoT video KV cache for action-only denoising."""
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return video_kv_cache, video_seq_len

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        clean_latent_count: Optional[int] = None,
        action_condition_start_latent: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict joint video/action velocities with the standard DW05 MoT path."""
        video_pre_kwargs: dict[str, Any] = {}
        if clean_latent_count is not None:
            video_pre_kwargs["clean_latent_count"] = int(clean_latent_count)
        if action_condition_start_latent is not None:
            video_pre_kwargs["action_condition_start_latent"] = int(action_condition_start_latent)

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            **video_pre_kwargs,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask_kwargs = {}
        if clean_latent_count is not None:
            attention_mask_kwargs["clean_latent_count"] = int(clean_latent_count)
        tokens_out = self._run_mot(
            video_pre=video_pre,
            action_pre=action_pre,
            attention_mask_kwargs=attention_mask_kwargs,
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        """Compute per-sample latent-video MSE with optional frame-padding mask."""
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @staticmethod
    def _compute_action_loss_per_sample(
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        action_dim_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample action MSE with optional timestep and dimension masks."""
        action_elem_mse = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_dim_mask is not None:
            dim_mask = action_dim_mask.to(device=action_elem_mse.device, dtype=action_elem_mse.dtype)
            if dim_mask.ndim != 3:
                raise ValueError(f"`action_dim_mask` must be [B,T,D], got shape {tuple(action_dim_mask.shape)}")
            dim_sum = dim_mask.sum(dim=2).clamp(min=1.0)
            action_loss_token = (action_elem_mse * dim_mask).sum(dim=2) / dim_sum
        else:
            action_loss_token = action_elem_mse.mean(dim=2)

        if action_is_pad is None:
            return action_loss_token.mean(dim=1)

        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (action_loss_token * valid).sum(dim=1) / valid_sum

    @staticmethod
    def _reduce_weighted_loss(
        loss_per_sample: torch.Tensor,
        sample_weight: torch.Tensor,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reduce a per-sample weighted objective into one scalar."""
        weighted = loss_per_sample * sample_weight.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        if sample_mask is None:
            return weighted.mean()
        mask = sample_mask.to(device=weighted.device, dtype=weighted.dtype).reshape(-1)
        return (weighted * mask).sum() / mask.sum().clamp(min=1.0)

    def _reduce_action_loss(
        self,
        action_loss_per_sample: torch.Tensor,
        action_weight: torch.Tensor,
        has_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate per-sample action loss, masking out wm-only samples when present."""
        return self._reduce_weighted_loss(action_loss_per_sample, action_weight, has_action)

    def _weighted_video_loss(self, loss_per_sample: torch.Tensor, timestep_video: torch.Tensor) -> torch.Tensor:
        """Apply DW05 video scheduler weights and reduce to one scalar."""
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )
        return self._reduce_weighted_loss(loss_per_sample, video_weight)

    def _weighted_action_loss(
        self,
        loss_per_sample: torch.Tensor,
        timestep_action: torch.Tensor,
        has_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply DW05 action scheduler weights and reduce to one scalar."""
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )
        return self._reduce_action_loss(loss_per_sample, action_weight, has_action)

@dataclass
class DW05ModelConfig:
    """Model configuration for the DW05 WAM method.

    The fields describe DW05 model shape, Wan2.2/ActionDiT expert settings,
    scheduler parameters, and loss weights.  Environment-specific paths, such as
    the local Wan2.2 cache root, are supplied by the experiment layer before
    calling :meth:`build_model`.

    DW05 is the only concrete method in this WAM-only package, so new configs
    should use ``architecture='dw05'``.
    """

    architecture: str = "dw05"
    model_id: str = WAN22_MODEL_ID
    tokenizer_model_id: str = WAN22_TOKENIZER_MODEL_ID
    action_dit_pretrained_path: Optional[str] = None
    tokenizer_max_len: int = 128
    load_text_encoder: bool = False
    redirect_common_files: bool = True
    skip_dit_load_from_pretrain: bool = False
    mot_checkpoint_mixed_attn: bool = False
    action_drop_prob: float = 0.5
    proprio_dim: Optional[int] = 32
    proprio_drop_prob: float = 0.0
    action_dim: int = 32
    video_hidden_dim: int = 3072
    video_ffn_dim: int = 14336
    action_hidden_dim: int = 1024
    action_ffn_dim: int = 4096
    num_layers: int = 30
    num_heads: int = 24
    attn_head_dim: int = 128
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1.0e-6
    video_train_shift: float = 5.0
    video_infer_shift: float = 5.0
    action_train_shift: float = 5.0
    action_infer_shift: float = 5.0
    num_train_timesteps: int = 1000
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    def build_video_dit_config(self) -> dict[str, Any]:
        """Build Wan2.2 video expert constructor kwargs for DW05."""
        return {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "hidden_dim": int(self.video_hidden_dim),
            "ffn_dim": int(self.video_ffn_dim),
            "freq_dim": int(self.freq_dim),
            "text_dim": int(self.text_dim),
            "out_dim": 48,
            "num_heads": int(self.num_heads),
            "attn_head_dim": int(self.attn_head_dim),
            "num_layers": int(self.num_layers),
            "eps": float(self.eps),
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": bool(self.mot_checkpoint_mixed_attn),
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": int(self.action_dim),
            "action_group_causal_mask_mode": "group_diagonal",
        }

    def build_action_dit_config(self) -> dict[str, Any]:
        """Build ActionDiT expert constructor kwargs for DW05."""
        return {
            "action_dim": int(self.action_dim),
            "hidden_dim": int(self.action_hidden_dim),
            "ffn_dim": int(self.action_ffn_dim),
            "num_heads": int(self.num_heads),
            "attn_head_dim": int(self.attn_head_dim),
            "num_layers": int(self.num_layers),
            "text_dim": int(self.text_dim),
            "freq_dim": int(self.freq_dim),
            "eps": float(self.eps),
            "use_gradient_checkpointing": bool(self.mot_checkpoint_mixed_attn),
        }

    def build_scheduler_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build video/action flow-matching scheduler configs."""
        video_scheduler = {
            "train_shift": float(self.video_train_shift),
            "infer_shift": float(self.video_infer_shift),
            "num_train_timesteps": int(self.num_train_timesteps),
        }
        action_scheduler = {
            "train_shift": float(self.action_train_shift),
            "infer_shift": float(self.action_infer_shift),
            "num_train_timesteps": int(self.num_train_timesteps),
        }
        return video_scheduler, action_scheduler

    def build_loss_config(self) -> dict[str, Any]:
        """Build loss-weight config consumed by the DW05 factory."""
        return {
            "lambda_video": float(self.loss_lambda_video),
            "lambda_action": float(self.loss_lambda_action),
        }

    def build_model(
        self,
        *,
        model_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> nn.Module:
        """Instantiate the concrete DW05 WAM model."""
        if self.architecture != "dw05":
            raise ValueError(
                f"Unsupported WAM architecture: {self.architecture}. "
                "Only 'dw05' is available in this WAM-only repository."
            )

        video_scheduler, action_scheduler = self.build_scheduler_configs()
        return create_dw05(
            model_id=self.model_id,
            tokenizer_model_id=self.tokenizer_model_id,
            video_dit_config=self.build_video_dit_config(),
            action_drop_prob=float(self.action_drop_prob),
            num_hist_frames=1,
            tokenizer_max_len=int(self.tokenizer_max_len),
            load_text_encoder=bool(self.load_text_encoder),
            proprio_dim=self.proprio_dim,
            proprio_drop_prob=float(self.proprio_drop_prob),
            action_dit_config=self.build_action_dit_config(),
            action_dit_pretrained_path=self.action_dit_pretrained_path,
            skip_dit_load_from_pretrain=bool(self.skip_dit_load_from_pretrain),
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            loss=self.build_loss_config(),
            mot_checkpoint_mixed_attn=bool(self.mot_checkpoint_mixed_attn),
            redirect_common_files=bool(self.redirect_common_files),
            model_dtype=model_dtype,
            device=device,
        )

__all__ = [
    "DW05ModelConfig",
    "DW05Wan22WorldActionModel",
    "WAN22_MODEL_ID",
    "WAN22_TOKENIZER_MODEL_ID",
    "create_dw05",
]
