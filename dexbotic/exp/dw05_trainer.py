"""DW05 trainer built on the generic Dexbotic generative trainer.

The shape mirrors ``dexbotic-open`` method trainers such as ``mem_trainer.py``:
the shared training mechanics live in a base trainer, while this file keeps only
DW05-specific freeze and evaluation behavior.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from PIL import Image

from dexbotic.exp.generative_trainer import DexboticGenerativeTrainer
from dexbotic.exp.utils import pil_frames_to_video_tensor, save_mp4, video_psnr, video_ssim


logger = logging.getLogger(__name__)


class DW05Trainer(DexboticGenerativeTrainer):
    """Trainer specialization for the DW05 Wan2.2 world-action method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_num_inference_steps = int(self.cfg.eval_num_inference_steps)
        self.eval_fps = int(self.cfg.get("eval_fps", 8))

    def configure_trainable_modules(self, model) -> None:
        """Delegate DW05 freeze policy to the model architecture."""
        model.configure_trainable_modules()

    def set_train_mode(self) -> None:
        model = self.accelerator.unwrap_model(self.model)
        model.configure_trainable_modules()

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"`sample['action']` must be a torch.Tensor, got {type(action)}")
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(
                    f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, "
                    f"got {action.shape[1]}"
                )
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        model.eval()

        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        num_hist = getattr(model, "num_hist_frames", 1)
        num_frames_offset = 0
        action_offset = 0
        action_video_freq_ratio = 1
        if action is not None:
            if action.shape[0] % (num_frames - 1) != 0:
                raise ValueError(
                    f"Eval action horizon must be divisible by video transitions: "
                    f"action={action.shape[0]}, video_transitions={num_frames - 1}."
                )
            action_video_freq_ratio = action.shape[0] // (num_frames - 1)
        if num_hist > 1:
            # Each extra clean history latent corresponds to four sampled video frames.
            num_frames_offset = (num_hist - 1) * 4
            if num_frames_offset >= num_frames:
                raise ValueError(
                    f"Hist eval offset {num_frames_offset} is out of range for sampled video length {num_frames}."
                )
            input_image = video0[:, num_frames_offset].unsqueeze(0)
            input_images = [video0[:, i].unsqueeze(0) for i in range(num_frames_offset + 1)]
            action_offset = num_frames_offset * action_video_freq_ratio
            num_frames = num_frames - num_frames_offset
            if action is not None:
                action = action[action_offset:]
        else:
            input_images = None

        proprio = None
        if "proprio" in sample and sample["proprio"] is not None:
            proprio_seq = sample["proprio"][0]
            if action_offset < 0 or action_offset >= proprio_seq.shape[0]:
                raise ValueError(
                    f"Hist proprio offset {action_offset} is out of range for eval proprio length {proprio_seq.shape[0]}."
                )
            proprio = proprio_seq[action_offset]

        eval_action_horizon = int(sample["action_horizon"] - action_offset)
        if eval_action_horizon <= 0:
            raise ValueError(
                f"Eval action horizon must be positive after hist offset: "
                f"raw={sample['action_horizon']} offset={action_offset}."
            )

        infer_kwargs = {
            "input_image": input_image,
            "input_images": input_images,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": eval_action_horizon,
            "proprio": proprio,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(**infer_kwargs)
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_slice = video0[:, num_frames_offset:] if num_frames_offset > 0 else video0
        gt_video_tensor = ((gt_video_slice.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        if pred_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError(
                "Eval infer prediction/GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        processor = None
        if action is not None and pred_action is not None:
            processor = getattr(self.val_dataset, "processor", None)

            proprio_for_metrics = sample.get("proprio", None)
            if processor is not None and proprio_for_metrics is not None:
                proprio = proprio_for_metrics.detach().to(device="cpu", dtype=torch.float32)
                denorm_actions = {}
                action_meta = processor.shape_meta["action"]
                state_meta = processor.shape_meta["state"]
                for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                    if not isinstance(raw_action, torch.Tensor):
                        raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                    if raw_action.ndim == 2:
                        action_btd = raw_action.unsqueeze(0)
                    elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                        action_btd = raw_action
                    else:
                        raise ValueError(
                            f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                        )
                    action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                    batch = {"action": action_btd, "state": proprio}
                    batch = processor.action_state_merger.backward(batch)
                    batch = processor.normalizer.backward(batch)
                    if processor.action_state_transforms is not None:
                        for transform in reversed(processor.action_state_transforms):
                            batch = transform.backward(batch)
                    merged_batch = {
                        "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                        "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                    }
                    merged_batch = processor.action_state_merger.forward(merged_batch)
                    denorm_action = merged_batch["action"].unsqueeze(0)
                    if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                        raise ValueError(
                            f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                        )
                    denorm_actions[action_name] = denorm_action

                pred_action_denorm = denorm_actions["pred"]
                gt_action_denorm = denorm_actions["gt"]
            else:
                pred_action_denorm = pred_action.detach().to(device="cpu", dtype=torch.float32)
                if pred_action_denorm.ndim == 2:
                    pred_action_denorm = pred_action_denorm.unsqueeze(0)
                gt_action_denorm = action.detach().to(device="cpu", dtype=torch.float32)
                if gt_action_denorm.ndim == 2:
                    gt_action_denorm = gt_action_denorm.unsqueeze(0)

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        gt_video_batch = gt_video_slice.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        if vae_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError(
                "Eval VAE reconstruction/GT shape mismatch: "
                f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)
        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat([pred_video_tensor, vae_video_tensor, gt_video_tensor], dim=2).contiguous()
        stitched_frames = []
        for timestep in range(stitched_video_tensor.shape[1]):
            frame = (
                stitched_video_tensor[:, timestep].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
            ).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=self.eval_fps)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        torch.cuda.empty_cache()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def format_eval_log(self, metrics: dict) -> str:
        description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
            self.global_step,
            metrics["val_loss"],
            metrics["psnr_rd"],
            metrics["ssim_rd"],
        )
        if "action_l2" in metrics:
            description += " action_l2=%.4f" % metrics["action_l2"]
        if "action_l1" in metrics:
            description += " action_l1=%.4f" % metrics["action_l1"]
        return description

    def build_eval_wandb_payload(self, metrics: dict) -> dict[str, float]:
        payload = {
            "eval/val_loss": float(metrics["val_loss"]),
            "eval/psnr_rg": float(metrics["psnr_rg"]),
            "eval/ssim_rg": float(metrics["ssim_rg"]),
            "eval/psnr_rd": float(metrics["psnr_rd"]),
            "eval/ssim_rd": float(metrics["ssim_rd"]),
            "eval/psnr_dg": float(metrics["psnr_dg"]),
            "eval/ssim_dg": float(metrics["ssim_dg"]),
        }
        if "action_l2" in metrics:
            payload["eval/action_l2"] = float(metrics["action_l2"])
        if "action_l1" in metrics:
            payload["eval/action_l1"] = float(metrics["action_l1"])
        return payload

__all__ = ["DW05Trainer"]
