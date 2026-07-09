from typing import Optional

import torch

from .dw05_history import DW05History


class DW05WorldActionModel(DW05History):
    """DW05History with GT action (current window) as clean condition in MoT self-attn.

    Sequence: [v_clean(H*tpf) | v_noisy(F*tpf) | a_clean(noisy_len) | a_noisy(noisy_len)]
    v_noisy_i attends a_clean group i (group-diagonal).
    a_noisy attends v_clean only (not a_clean).
    H=1: single clean image + full GT action condition.
    H=2: two clean images + full GT action condition.
    """

    def __init__(self, *args, action_drop_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_drop_prob = float(action_drop_prob)

    @classmethod
    def from_wan22_pretrained(cls, action_drop_prob: float = 0.5, **kwargs):
        instance = super().from_wan22_pretrained(**kwargs)
        instance.action_drop_prob = float(action_drop_prob)
        return instance

    def training_loss(self, sample, tiled: bool = False, *, drop_action_condition: bool = True):
        H = self.num_hist_frames
        # Drop action condition with action_drop_prob (CFG-style) and train without the clean-action branch.
        if drop_action_condition and self._random_condition_drop(self.action_drop_prob):
            loss, loss_dict = super().training_loss(sample, tiled)
            loss_dict["action_condition_dropped"] = 1.0
            return loss, loss_dict
        inputs = self.build_inputs(sample, tiled)
        input_latents = inputs["input_latents"]   # [B, C, H+2, lh, lw]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]                 # [B, noisy_action_len, action_dim] — GT clean
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

        video_pre = self.video_expert.pre_dit(
            x=latents, timestep=timestep_video,
            context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            clean_latent_count=H,
        )

        # clean action: GT action with timestep=0
        timestep_zero = torch.zeros((batch_size,), dtype=action.dtype, device=self.device)
        clean_action_pre = self.action_expert.pre_dit(
            action_tokens=action, timestep=timestep_zero,
            context=context, context_mask=context_mask,
        )
        noisy_action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=context, context_mask=context_mask,
        )

        clean_len = clean_action_pre["tokens"].shape[1]
        noisy_len = noisy_action_pre["tokens"].shape[1]

        action_tokens_cat = torch.cat([clean_action_pre["tokens"], noisy_action_pre["tokens"]], dim=1)
        action_freqs_cat = torch.cat([clean_action_pre["freqs"], noisy_action_pre["freqs"]], dim=0)
        t_mod_clean = clean_action_pre["t_mod"].unsqueeze(1).expand(-1, clean_len, -1, -1)
        t_mod_noisy = noisy_action_pre["t_mod"].unsqueeze(1).expand(-1, noisy_len, -1, -1)
        action_t_mod_cat = torch.cat([t_mod_clean, t_mod_noisy], dim=1)
        action_ctx_mask_cat = torch.cat(
            [clean_action_pre["context_mask"], noisy_action_pre["context_mask"]], dim=1)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_clean_seq_len=clean_len,
            action_noisy_seq_len=noisy_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            clean_latent_count=H,
        )
        tokens_out = self._run_mot(
            video_pre=video_pre,
            action_pre=noisy_action_pre,
            attention_mask=attention_mask,
            action_tokens=action_tokens_cat,
            action_freqs=action_freqs_cat,
            action_context=clean_action_pre["context"],
            action_context_mask=action_ctx_mask_cat,
            action_t_mod=action_t_mod_cat,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"][:, clean_len:], noisy_action_pre)

        pred_video = pred_video[:, :, H:]
        target_video = target_video[:, :, H:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video, target_video=target_video,
            image_is_pad=None, include_initial_video_step=True)
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
            "action_condition_dropped": 0.0,
        }
        self._add_condition_drop_metrics(loss_dict, inputs)
        return loss_total, loss_dict

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
        if gt_action is None:
            return super()._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                gt_action=None,
                clean_latent_count=clean_latent_count,
                action_condition_start_latent=action_condition_start_latent,
            )
        gt_action = self._normalize_infer_action_condition(
            gt_action,
            batch_size=latents_action.shape[0],
            action_horizon=latents_action.shape[1],
            name="gt_action",
        )
        clean_count = int(clean_latent_count if clean_latent_count is not None else getattr(self, "num_hist_frames", 1))

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            clean_latent_count=clean_count,
        )

        timestep_zero = torch.zeros((gt_action.shape[0],), dtype=gt_action.dtype, device=self.device)
        clean_action_pre = self.action_expert.pre_dit(
            action_tokens=gt_action,
            timestep=timestep_zero,
            context=context,
            context_mask=context_mask,
        )
        noisy_action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        clean_len = clean_action_pre["tokens"].shape[1]
        noisy_len = noisy_action_pre["tokens"].shape[1]
        action_tokens_cat = torch.cat([clean_action_pre["tokens"], noisy_action_pre["tokens"]], dim=1)
        action_freqs_cat = torch.cat([clean_action_pre["freqs"], noisy_action_pre["freqs"]], dim=0)
        t_mod_clean = clean_action_pre["t_mod"].unsqueeze(1).expand(-1, clean_len, -1, -1)
        t_mod_noisy = noisy_action_pre["t_mod"].unsqueeze(1).expand(-1, noisy_len, -1, -1)
        action_t_mod_cat = torch.cat([t_mod_clean, t_mod_noisy], dim=1)
        action_ctx_mask_cat = torch.cat(
            [clean_action_pre["context_mask"], noisy_action_pre["context_mask"]], dim=1
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_clean_seq_len=clean_len,
            action_noisy_seq_len=noisy_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            clean_latent_count=clean_count,
        )
        tokens_out = self._run_mot(
            video_pre=video_pre,
            action_pre=noisy_action_pre,
            attention_mask=attention_mask,
            action_tokens=action_tokens_cat,
            action_freqs=action_freqs_cat,
            action_context=clean_action_pre["context"],
            action_context_mask=action_ctx_mask_cat,
            action_t_mod=action_t_mod_cat,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"][:, clean_len:], noisy_action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_seq_len: int = 0,
        action_clean_seq_len: int = 0,
        action_noisy_seq_len: int = 0,
        clean_latent_count: Optional[int] = None,
    ) -> torch.Tensor:
        H = self.num_hist_frames if clean_latent_count is None else int(clean_latent_count)
        clean_tokens = H * video_tokens_per_frame

        # infer-time fallback
        if action_seq_len > 0 and action_clean_seq_len == 0 and action_noisy_seq_len == 0:
            total = video_seq_len + action_seq_len
            mask = torch.zeros((total, total), dtype=torch.bool, device=device)
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            video_mask[:clean_tokens, clean_tokens:] = False
            mask[:video_seq_len, :video_seq_len] = video_mask
            mask[video_seq_len:, video_seq_len:] = True
            mask[video_seq_len:, :clean_tokens] = True
            return mask

        AC = action_clean_seq_len
        AN = action_noisy_seq_len
        V = video_seq_len
        F_noisy = (V - clean_tokens) // video_tokens_per_frame

        total = V + AC + AN
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        video_mask = torch.ones((V, V), dtype=torch.bool, device=device)
        video_mask[:clean_tokens, clean_tokens:] = False
        mask[:V, :V] = video_mask

        if AC > 0 and F_noisy > 0:
            actions_per_frame = AC // F_noisy
            for i in range(F_noisy):
                v_s = clean_tokens + i * video_tokens_per_frame
                v_e = v_s + video_tokens_per_frame
                a_s = V + i * actions_per_frame
                a_e = V + (i + 1) * actions_per_frame
                mask[v_s:v_e, a_s:a_e] = True

        mask[V:V+AC, :clean_tokens] = True
        mask[V:V+AC, V:V+AC] = True
        mask[V+AC:, V+AC:] = True
        mask[V+AC:, :clean_tokens] = True

        return mask
