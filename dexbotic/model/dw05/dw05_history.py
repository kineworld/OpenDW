import logging
from typing import Any, List, Optional

import torch

from .dw05_core import DW05Core

logger = logging.getLogger(__name__)


class DW05History(DW05Core):
    """DW05Core with historical video context plus a clean current frame.

    num_hist_frames=1 degrades to original DW05Core behavior. For H>1,
    the first H-1 clean latents encode historical sampled-frame blocks and
    the H-th clean latent encodes the current frame. The current/future VAE
    window starts at that current frame, so the two noisy video latents align
    to future frames after the current frame.
    """

    def __init__(self, *args, num_hist_frames: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        if num_hist_frames < 1:
            raise ValueError(f"`num_hist_frames` must be >= 1, got {num_hist_frames}")
        self.num_hist_frames = num_hist_frames

    @classmethod
    def from_wan22_pretrained(cls, num_hist_frames: int = 2, **kwargs):
        instance = super().from_wan22_pretrained(**kwargs)
        instance.num_hist_frames = int(num_hist_frames)
        return instance

    def _select_proprio_index(
        self,
        *,
        num_video_frames: int,
        action_horizon: int,
        latent_frames: int,
    ) -> int:
        H = int(getattr(self, "num_hist_frames", 1))
        if H <= 1:
            return super()._select_proprio_index(
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                latent_frames=latent_frames,
            )
        action_offset, _ = self._hist_action_offset(
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
        )
        return action_offset

    def _history_sampled_len(self) -> int:
        H = int(getattr(self, "num_hist_frames", 1))
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        return max(0, H - 1) * temporal_factor

    def _hist_action_offset(self, *, num_video_frames: int, action_horizon: int) -> tuple[int, int]:
        transitions = int(num_video_frames) - 1
        if transitions <= 0:
            raise ValueError(f"Video must contain at least 2 sampled frames, got {num_video_frames}.")
        if int(action_horizon) % transitions != 0:
            raise ValueError(
                "`sample['action']` temporal dimension must be divisible by sampled video transitions: "
                f"action_horizon={action_horizon}, transitions={transitions}."
            )
        action_video_freq_ratio = int(action_horizon) // transitions
        history_sampled_len = self._history_sampled_len()
        if history_sampled_len >= int(num_video_frames):
            raise ValueError(
                "History sampled-frame prefix must leave a current frame: "
                f"history_sampled_len={history_sampled_len}, num_video_frames={num_video_frames}."
            )
        return history_sampled_len * action_video_freq_ratio, action_video_freq_ratio

    @torch.no_grad()
    def _encode_history_block_latents(self, history_video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if history_video.ndim != 5:
            raise ValueError(f"`history_video` must be [B,3,T,H,W], got shape {tuple(history_video.shape)}.")
        if history_video.shape[2] <= 0:
            raise ValueError("`history_video` must contain at least one sampled frame.")
        if history_video.shape[2] % temporal_factor != 0:
            raise ValueError(
                "Historical sampled frames must form complete VAE temporal blocks: "
                f"history_T={history_video.shape[2]}, temporal_factor={temporal_factor}."
            )

        # Wan VAE emits a single-frame first latent, then one latent per 4-frame chunk.
        # Duplicate the first history frame as the causal cache anchor and discard its latent,
        # so returned latents correspond to history chunks [0..3], [4..7], ... .
        anchor = history_video[:, :, 0:1]
        history_for_vae = torch.cat([anchor, history_video], dim=2)
        latents = self._encode_video_latents(history_for_vae, tiled=tiled)
        history_latents = latents[:, :, 1:]
        expected_latents = history_video.shape[2] // temporal_factor
        if history_latents.shape[2] != expected_latents:
            raise ValueError(
                "Historical VAE latent count mismatch: "
                f"got {history_latents.shape[2]}, expected {expected_latents}."
            )
        return history_latents

    @torch.no_grad()
    def _encode_hist_aligned_video_latents(self, input_video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        H = int(self.num_hist_frames)
        history_sampled_len = self._history_sampled_len()
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if input_video.shape[2] <= history_sampled_len:
            raise ValueError(
                "Video does not contain a current frame after the history prefix: "
                f"video_T={input_video.shape[2]}, history_sampled_len={history_sampled_len}."
            )

        history_video = input_video[:, :, :history_sampled_len]
        current_future_video = input_video[:, :, history_sampled_len:]
        if current_future_video.shape[2] % temporal_factor != 1:
            raise ValueError(
                "Current/future VAE window must satisfy T % temporal_factor == 1: "
                f"current_future_T={current_future_video.shape[2]}, temporal_factor={temporal_factor}."
            )
        if current_future_video.shape[2] <= 1:
            raise ValueError(f"Current/future VAE window must include future frames, got T={current_future_video.shape[2]}.")

        history_latents = self._encode_history_block_latents(history_video, tiled=tiled)
        current_future_latents = self._encode_video_latents(current_future_video, tiled=tiled)
        if history_latents.shape[2] != H - 1:
            raise ValueError(f"Expected {H - 1} history latents, got {history_latents.shape[2]}.")
        if current_future_latents.shape[2] < 2:
            raise ValueError(
                "Current/future VAE window must produce at least current + one future latent, "
                f"got {current_future_latents.shape[2]}."
            )
        return torch.cat([history_latents, current_future_latents], dim=2)

    @torch.no_grad()
    def _encode_hist_condition_latents(
        self,
        *,
        input_images: Optional[List[torch.Tensor]],
        input_image: Optional[torch.Tensor],
        tiled: bool = False,
    ) -> torch.Tensor:
        history_sampled_len = self._history_sampled_len()
        expected_frames = history_sampled_len + 1

        if input_images is not None:
            images = list(input_images)
            if len(images) == history_sampled_len:
                if input_image is None:
                    raise ValueError("`input_images` contains history frames only; `input_image` is required as current frame.")
                images.append(input_image)
            elif len(images) != expected_frames:
                raise ValueError(
                    "Hist inference expects sampled condition frames [history..., current]: "
                    f"got {len(images)} frames, expected {expected_frames}."
                )
        elif input_image is not None:
            images = [input_image] * expected_frames
        else:
            raise ValueError("Either `input_images` or `input_image` must be provided.")

        frames = [self._normalize_single_infer_image(img).unsqueeze(2) for img in images]
        condition_video = torch.cat(frames, dim=2)
        history_video = condition_video[:, :, :history_sampled_len]
        current_image = condition_video[:, :, history_sampled_len]

        history_latents = self._encode_history_block_latents(history_video, tiled=tiled)
        current_latents = self._encode_input_image_latents_tensor(current_image, tiled=tiled)
        if history_latents.shape[0] != current_latents.shape[0]:
            if current_latents.shape[0] == 1:
                current_latents = current_latents.expand(history_latents.shape[0], -1, -1, -1, -1)
            else:
                raise ValueError(
                    "History/current latent batch mismatch: "
                    f"history={history_latents.shape[0]}, current={current_latents.shape[0]}."
                )
        return torch.cat([history_latents, current_latents], dim=2)

    def _prepare_hist_video_condition_action(
        self,
        action: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if action is None:
            return None
        return self._normalize_infer_action_condition(action, batch_size=1, name="action")

    # ------------------------------------------------------------------
    # build_inputs: split sampled video into history + current/future window
    # ------------------------------------------------------------------
    def build_inputs(self, sample, tiled: bool = False):
        H = self.num_hist_frames
        if H <= 1:
            return super().build_inputs(sample, tiled)

        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("DW05Core training requires `sample['context']` and `sample['context_mask']`.")
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")

        history_sampled_len = self._history_sampled_len()
        current_future_frames = num_frames - history_sampled_len
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if current_future_frames % temporal_factor != 1:
            raise ValueError(
                "Hist current/future video T must satisfy T % temporal_factor == 1 after history offset: "
                f"num_frames={num_frames}, history_sampled_len={history_sampled_len}, "
                f"current_future_frames={current_future_frames}, temporal_factor={temporal_factor}."
            )
        if current_future_frames <= 1:
            raise ValueError(f"Hist current/future video must include future frames, got T={current_future_frames}.")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for DW05Core training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        action_offset, action_video_freq_ratio = self._hist_action_offset(
            num_video_frames=num_frames,
            action_horizon=action_horizon,
        )
        current_action_horizon = action_horizon - action_offset
        expected_current_action_horizon = (current_future_frames - 1) * action_video_freq_ratio
        if current_action_horizon != expected_current_action_horizon:
            raise ValueError(
                "Hist action/video alignment mismatch after history offset: "
                f"current_action_horizon={current_action_horizon}, expected={expected_current_action_horizon}."
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}")
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}")
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_hist_aligned_video_latents(input_video, tiled=tiled)
        first_frame_latents = input_latents[:, :, :H]
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        context, context_mask, proprio_dropped = self._prepare_training_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            latent_frames=int(input_latents.shape[2]),
        )

        full_action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": full_action[:, action_offset:],
            "full_action": full_action,
            "hist_action": full_action[:, :action_offset],
            "action_is_pad": action_is_pad[:, action_offset:] if action_is_pad is not None else None,
            "image_is_pad": image_is_pad,
            "proprio_dropped": proprio_dropped,
            "history_sampled_len": history_sampled_len,
            "action_offset": action_offset,
            "action_video_freq_ratio": action_video_freq_ratio,
        }

    # ------------------------------------------------------------------
    # training_loss
    # ------------------------------------------------------------------
    def training_loss(self, sample, tiled: bool = False):
        H = self.num_hist_frames
        if H <= 1:
            return super().training_loss(sample, tiled)

        inputs = self.build_inputs(sample, tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]

        _, timestep_video, latents, target_video = self._sample_training_flow(
            input_latents,
            self.train_video_scheduler,
        )
        latents = self._restore_clean_latents(latents, inputs["first_frame_latents"], H)

        _, timestep_action, noisy_action, target_action = self._sample_training_flow(
            action,
            self.train_action_scheduler,
        )

        gt_action_for_video = None
        action_condition_dropped = False
        if getattr(self.video_expert, "action_conditioned", False):
            action_condition_dropped = self._random_condition_drop(0.5)
            if not action_condition_dropped:
                gt_action_for_video = inputs.get("action")
        video_pre = self.video_expert.pre_dit(
            x=latents, timestep=timestep_video,
            context=context, context_mask=context_mask,
            action=gt_action_for_video,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            clean_latent_count=H,
            action_condition_start_latent=H,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=context, context_mask=context_mask,
        )

        tokens_out = self._run_mot(
            video_pre=video_pre,
            action_pre=action_pre,
            attention_mask_kwargs={"clean_latent_count": H},
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_video = pred_video[:, :, H:]
        target_video = target_video[:, :, H:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=None,
            include_initial_video_step=True,
        )
        loss_video = self._weighted_video_loss(loss_video_per_sample, timestep_video)

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        loss_action = self._weighted_action_loss(action_loss_per_sample, timestep_action)

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
        }
        if getattr(self.video_expert, "action_conditioned", False):
            loss_dict["action_condition_dropped"] = float(action_condition_dropped)
        self._add_condition_drop_metrics(loss_dict, inputs)
        return loss_total, loss_dict

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        clean_latent_count: Optional[int] = None,
    ) -> torch.Tensor:
        H = self.num_hist_frames if clean_latent_count is None else int(clean_latent_count)
        if H <= 1:
            return super()._build_mot_attention_mask(video_seq_len, action_seq_len, video_tokens_per_frame, device)

        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        clean_tokens = min(H * video_tokens_per_frame, video_seq_len)

        video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        video_mask[:clean_tokens, clean_tokens:] = False
        mask[:video_seq_len, :video_seq_len] = video_mask

        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :clean_tokens] = True
        return mask

    @torch.no_grad()
    def _predict_joint_noise_hist(
        self,
        *,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        clean_latent_count: int,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_action_for_video = self._prepare_hist_video_condition_action(gt_action)
        return super()._predict_joint_noise(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            gt_action=gt_action_for_video,
            clean_latent_count=clean_latent_count,
            action_condition_start_latent=clean_latent_count,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        action_horizon: int,
        input_images: Optional[List[torch.Tensor]] = None,
        input_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        H = self.num_hist_frames
        if H <= 1:
            img = input_image if input_image is not None else (input_images[0] if input_images else None)
            return super().infer_action(
                prompt=prompt, input_image=img, action_horizon=action_horizon,
                proprio=proprio, context=context, context_mask=context_mask,
                num_inference_steps=num_inference_steps, sigma_shift=sigma_shift,
                seed=seed, rand_device=rand_device, tiled=tiled,
            )

        self.eval()
        condition_latents = self._encode_hist_condition_latents(
            input_images=input_images,
            input_image=input_image,
            tiled=tiled,
        )
        _, _, _, latent_h, latent_w = condition_latents.shape
        _ = (latent_h, latent_w)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._prepare_infer_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            strict_proprio=True,
        )

        timestep_video_zero = torch.zeros((1,), dtype=self.torch_dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=condition_latents,
            timestep=timestep_video_zero,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            clean_latent_count=H,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            action_seq_len=action_horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            clean_latent_count=H,
        )
        video_kv_cache, video_seq_len = self._prefill_video_cache(video_pre, attention_mask)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        input_images: Optional[List[torch.Tensor]] = None,
    ) -> dict[str, Any]:
        H = self.num_hist_frames
        if H <= 1:
            return super().infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                test_action_with_infer_action=test_action_with_infer_action,
            )

        self.eval()
        condition_latents = self._encode_hist_condition_latents(
            input_images=input_images,
            input_image=input_image,
            tiled=tiled,
        )
        if condition_latents.shape[2] != H:
            raise ValueError(f"Expected {H} clean condition latents, got {condition_latents.shape[2]}.")

        history_sampled_len = self._history_sampled_len()
        if input_images is not None and len(input_images) == history_sampled_len + 1:
            current_image = input_images[-1]
        else:
            current_image = input_image
        current_image = self._normalize_single_infer_image(current_image)
        _, _, height, width = current_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        main_latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        history_latent_t = H - 1
        total_latent_t = history_latent_t + main_latent_t
        clean_latent_count = H

        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone() if input_image is not None else None,
                input_images=[img.clone() for img in input_images] if input_images is not None else None,
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, total_latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_video[:, :, :clean_latent_count] = condition_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        context, context_mask = self._prepare_infer_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            strict_proprio=True,
        )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_video, pred_action = self._predict_joint_noise_hist(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                clean_latent_count=clean_latent_count,
                gt_action=action,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, :clean_latent_count] = condition_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        main_latents = latents_video[:, :, history_latent_t:]
        return {
            "video": self._decode_latents(main_latents, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        input_images: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            input_images=input_images,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
