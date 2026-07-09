"""DW05 RobotWin deployment policy.

This module is the shared runtime layer for real-robot service, RoboTwin
evaluation, WorldArena rollout generation, and the V5 online demo.  It avoids
the previous Hydra stack and loads DW05 checkpoints through the migrated
Dexbotic model package directly.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dexbotic.data.dataset.dw05.data_source import ROBOTWIN_META
from dexbotic.exp.dw05_exp import DIFFSYNTH_MODEL_BASE_PATH_ENV, default_dw05_model_base_path
from dexbotic.exp.utils import mixed_precision_to_model_dtype, normalize_mixed_precision
from dexbotic.model.dw05 import DW05ModelConfig


ROBOTWIN_PROMPT_FORMAT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROBOTWIN_STATE_ARRANGEMENT = tuple(int(x) for x in ROBOTWIN_META["state_arrangement"])
ROBOTWIN_NON_DELTA_DIMS = (6, 13)
ROBOTWIN_VALID_ARRANGED_DIMS = tuple(idx for idx, src in enumerate(ROBOTWIN_STATE_ARRANGEMENT) if src >= 0)
ROBOTWIN_RAW_DIM = max(src for src in ROBOTWIN_STATE_ARRANGEMENT if src >= 0) + 1


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse a human-friendly boolean used by CLI/env entrypoints."""
    if _is_none_like(value):
        return bool(default)
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_optional_int(value: Any) -> Optional[int]:
    return None if _is_none_like(value) else int(value)


def parse_optional_float(value: Any) -> Optional[float]:
    return None if _is_none_like(value) else float(value)


def _normalize_stats_entry(entry: dict[str, Any]) -> dict[str, np.ndarray]:
    """Normalize supported RobotWin stat schemas to flat numpy arrays."""
    if "default" in entry and isinstance(entry["default"], dict):
        entry = entry["default"]

    stats: dict[str, np.ndarray] = {}
    for stat_key, stat_value in entry.items():
        if isinstance(stat_value, dict):
            continue
        stats[stat_key] = np.asarray(stat_value, dtype=np.float32)

    aliases = {
        "global_mean": "mean",
        "global_std": "std",
        "global_min": "min",
        "global_max": "max",
        "global_q01": "q01",
        "global_q99": "q99",
    }
    for src, dst in aliases.items():
        if src in stats and dst not in stats:
            stats[dst] = stats[src]
    return stats


def _load_norm_stats(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load DW05/RobotWin norm stats as numpy arrays."""
    norm_path = Path(path).expanduser()
    if not norm_path.exists():
        raise FileNotFoundError(f"DW05 norm stats not found: {norm_path}")
    payload = json.loads(norm_path.read_text())
    raw_stats = payload["norm_stats"] if "norm_stats" in payload else payload
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in raw_stats.items():
        if key not in {"action", "state"} or not isinstance(value, dict):
            continue
        stats[key] = _normalize_stats_entry(value)
    return stats


def _checkpoint_dims(checkpoint_path: str | Path) -> tuple[int, Optional[int]]:
    """Infer action/proprio dimensions from a DW05 checkpoint without GPU load."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    state = payload.get("mot") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"DW05 checkpoint missing `mot` state dict: {checkpoint_path}")
    action_key = "mixtures.action.action_encoder.weight"
    if action_key not in state:
        raise ValueError(f"DW05 checkpoint missing `{action_key}`: {checkpoint_path}")
    action_dim = int(state[action_key].shape[1])

    proprio_dim = None
    proprio_state = payload.get("proprio_encoder") if isinstance(payload, dict) else None
    if isinstance(proprio_state, dict) and "weight" in proprio_state:
        proprio_dim = int(proprio_state["weight"].shape[1])
    return action_dim, proprio_dim


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, Image.Resampling.BILINEAR), dtype=np.uint8)


def _letterbox_resize_np(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Long-edge resize plus zero padding, matching DW05 dataset preprocessing."""
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = _resize_rgb(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def compose_robotwin_image(images: list[np.ndarray], *, layout: str = "robotwin", image_size_hw: tuple[int, int] = (384, 320)) -> np.ndarray:
    """Compose raw camera images into a single DW05 condition image."""
    if not images:
        raise ValueError("At least one image is required.")
    key = str(layout).strip().lower()
    if key == "auto":
        key = "robotwin" if len(images) == 3 else "single"
    target_h, target_w = image_size_hw

    if key == "single":
        if len(images) != 1:
            raise ValueError(f"single layout expects one image, got {len(images)}")
        return _resize_rgb(images[0], (target_w, target_h))

    if key == "robotwin":
        if len(images) != 3:
            raise ValueError(f"robotwin layout expects head/left/right images, got {len(images)}")
        top_h = (target_h * 2) // 3
        bottom_h = target_h - top_h
        left_w = target_w // 2
        right_w = target_w - left_w
        head = _letterbox_resize_np(images[0], top_h, target_w)
        left = _letterbox_resize_np(images[1], bottom_h, left_w)
        right = _letterbox_resize_np(images[2], bottom_h, right_w)
        return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)

    if key in {"robotwin_resize", "robotwin_direct"}:
        if len(images) != 3:
            raise ValueError(f"{key} layout expects head/left/right images, got {len(images)}")
        top_h = (target_h * 2) // 3
        bottom_h = target_h - top_h
        left_w = target_w // 2
        right_w = target_w - left_w
        head = _resize_rgb(images[0], (target_w, top_h))
        left = _resize_rgb(images[1], (left_w, bottom_h))
        right = _resize_rgb(images[2], (right_w, bottom_h))
        return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)

    if key == "horizontal":
        widths = [target_w // len(images)] * len(images)
        widths[-1] += target_w - sum(widths)
        return np.concatenate([_resize_rgb(img, (w, target_h)) for img, w in zip(images, widths)], axis=1)

    if key == "vertical":
        heights = [target_h // len(images)] * len(images)
        heights[-1] += target_h - sum(heights)
        return np.concatenate([_resize_rgb(img, (target_w, h)) for img, h in zip(images, heights)], axis=0)

    raise ValueError(f"Unsupported image layout: {layout!r}")


def _arrange_raw_state(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    arranged = np.zeros(len(ROBOTWIN_STATE_ARRANGEMENT), dtype=np.float32)
    for dst, src in enumerate(ROBOTWIN_STATE_ARRANGEMENT):
        if src >= 0 and src < raw.shape[0]:
            arranged[dst] = raw[src]
    return arranged


def _reverse_arranged_state(arranged: np.ndarray, *, output_dim: int = ROBOTWIN_RAW_DIM) -> np.ndarray:
    arranged = np.asarray(arranged, dtype=np.float32)
    restored = np.zeros(arranged.shape[:-1] + (output_dim,), dtype=np.float32)
    for arranged_idx, src_idx in enumerate(ROBOTWIN_STATE_ARRANGEMENT):
        if arranged_idx >= arranged.shape[-1]:
            break
        if src_idx >= 0 and src_idx < output_dim:
            restored[..., src_idx] = arranged[..., arranged_idx]
    return restored


def _quantile_normalize(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    clipped = np.clip(value, q01, q99)
    return ((clipped - q01) / (q99 - q01 + 1.0e-6) * 2.0 - 1.0).astype(np.float32)


def _quantile_denormalize(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return ((value + 1.0) * 0.5 * (q99 - q01 + 1.0e-6) + q01).astype(np.float32)


def _zscore_normalize(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return np.clip((value - mean) / (std + 1.0e-8), -5.0, 5.0).astype(np.float32)


def _zscore_denormalize(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (value * (std + 1.0e-8) + mean).astype(np.float32)


def _stats_dim(stats: dict[str, np.ndarray]) -> int:
    for key in ("mean", "q01", "min", "std"):
        if key in stats:
            return int(np.asarray(stats[key]).shape[-1])
    return 0


def _select_model_dims(arranged_with_term: np.ndarray, dim: int) -> np.ndarray:
    """Project arranged-17 DW05 values into a model's action/proprio space."""
    if dim == len(ROBOTWIN_VALID_ARRANGED_DIMS):
        return arranged_with_term[..., list(ROBOTWIN_VALID_ARRANGED_DIMS)]
    if arranged_with_term.shape[-1] < dim:
        pad = np.zeros(arranged_with_term.shape[:-1] + (dim - arranged_with_term.shape[-1],), dtype=np.float32)
        return np.concatenate([arranged_with_term, pad], axis=-1)
    return arranged_with_term[..., :dim]


def _scatter_model_dims(model_value: np.ndarray, stats_dim: int) -> np.ndarray:
    """Map a model action vector back to arranged norm-stat space."""
    value = np.asarray(model_value, dtype=np.float32)
    if value.shape[-1] == len(ROBOTWIN_VALID_ARRANGED_DIMS):
        arranged = np.zeros(value.shape[:-1] + (stats_dim,), dtype=np.float32)
        for local_idx, arranged_idx in enumerate(ROBOTWIN_VALID_ARRANGED_DIMS):
            if arranged_idx < stats_dim:
                arranged[..., arranged_idx] = value[..., local_idx]
        return arranged
    if value.shape[-1] < stats_dim:
        pad = np.zeros(value.shape[:-1] + (stats_dim - value.shape[-1],), dtype=np.float32)
        return np.concatenate([value, pad], axis=-1)
    return value[..., :stats_dim]


@dataclass
class DW05RobotWinPolicyConfig:
    """Configuration for DW05 RobotWin runtime policy."""

    checkpoint_path: str
    norm_stats_path: str
    model_base_path: Optional[str] = None
    device: str = "cuda:0"
    mixed_precision: str = "bf16"
    action_horizon: int = 32
    replan_steps: int = 8
    num_inference_steps: int = 10
    num_video_frames: int = 9
    sigma_shift: Optional[float] = None
    seed: Optional[int] = None
    rand_device: str = "cpu"
    tiled: bool = False
    delta_action: bool = False
    delta_first_frame: bool = True
    image_layout: str = "auto"
    image_size_hw: tuple[int, int] = (384, 320)
    normalization_mode: str = "auto"
    action_condition_mode: str = "auto"
    load_text_encoder: bool = True
    prompt_format: str = ROBOTWIN_PROMPT_FORMAT
    raw_state_dim: int = ROBOTWIN_RAW_DIM
    raw_action_dim: int = ROBOTWIN_RAW_DIM


class DW05RobotWinPolicy:
    """Action policy used by DW05 deployment and evaluation entrypoints."""

    def __init__(self, config: DW05RobotWinPolicyConfig):
        self.config = config
        self.norm_stats = _load_norm_stats(config.norm_stats_path)
        if "action" not in self.norm_stats:
            raise ValueError(f"DW05 norm stats must contain `action`, got {list(self.norm_stats)}")
        if "state" not in self.norm_stats:
            raise ValueError(f"DW05 norm stats must contain `state`, got {list(self.norm_stats)}")
        requested_norm_mode = str(config.normalization_mode).strip().lower()
        requested_action_mode = str(config.action_condition_mode).strip().lower()

        if config.model_base_path:
            os.environ.setdefault(DIFFSYNTH_MODEL_BASE_PATH_ENV, str(config.model_base_path))
        elif default_dw05_model_base_path():
            os.environ.setdefault(DIFFSYNTH_MODEL_BASE_PATH_ENV, str(default_dw05_model_base_path()))

        action_dim, proprio_dim = _checkpoint_dims(config.checkpoint_path)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim if proprio_dim is not None else action_dim)
        self.raw_action_dim = int(config.raw_action_dim)
        self.raw_state_dim = int(config.raw_state_dim)

        if requested_norm_mode == "auto":
            action_stats = self.norm_stats["action"]
            if "global_mean" in action_stats or _stats_dim(action_stats) == self.raw_action_dim:
                requested_norm_mode = "zscore_14d"
            else:
                requested_norm_mode = "dw05_quantile"
        if requested_norm_mode not in {"dw05_quantile", "zscore_14d"}:
            raise ValueError(
                f"Unsupported normalization_mode={config.normalization_mode!r}; "
                "expected 'auto', 'dw05_quantile', or 'zscore_14d'."
            )

        if requested_action_mode == "auto":
            requested_action_mode = "absolute" if requested_norm_mode == "zscore_14d" else "delta_first_frame"
        if requested_action_mode not in {"delta_first_frame", "absolute"}:
            raise ValueError(
                f"Unsupported action_condition_mode={config.action_condition_mode!r}; "
                "expected 'auto', 'delta_first_frame', or 'absolute'."
            )

        self.normalization_mode = requested_norm_mode
        self.action_condition_mode = requested_action_mode
        if str(config.image_layout).strip().lower() == "auto":
            config.image_layout = "robotwin_resize" if self.normalization_mode == "zscore_14d" else "robotwin"

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self.last_raw_action: Optional[np.ndarray] = None
        self.last_composed_image: Optional[np.ndarray] = None
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        dtype = mixed_precision_to_model_dtype(normalize_mixed_precision(config.mixed_precision))
        model_cfg = DW05ModelConfig(
            load_text_encoder=bool(config.load_text_encoder),
            skip_dit_load_from_pretrain=True,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            mot_checkpoint_mixed_attn=False,
        )
        self.model = model_cfg.build_model(model_dtype=dtype, device=config.device)
        self.checkpoint_payload = self.model.load_checkpoint(config.checkpoint_path)
        self.model = self.model.to(config.device).eval()

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "DW05RobotWinPolicy":
        return cls(DW05RobotWinPolicyConfig(**kwargs))

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def reset_timing_rollout(self) -> None:
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

    def get_timing_rollout(self) -> dict[str, float]:
        return dict(self._timing_rollout)

    def format_prompt(self, instruction: str) -> str:
        if os.environ.get("DEPLOY_USE_DEFAULT_PROMPT", "1").lower() in {"0", "false", "no"}:
            return instruction
        return self.config.prompt_format.format(task=instruction)

    def _image_tensor_from_arrays(self, images: list[np.ndarray]) -> torch.Tensor:
        image = compose_robotwin_image(images, layout=self.config.image_layout, image_size_hw=self.config.image_size_hw)
        self.last_composed_image = image
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.model.device, dtype=self.model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def build_image_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
        obs = observation.get("observation", observation)
        if "images" in obs:
            images = [np.asarray(image) for image in obs["images"]]
        elif all(key in obs for key in ("head_camera", "left_camera", "right_camera")):
            images = [obs["head_camera"]["rgb"], obs["left_camera"]["rgb"], obs["right_camera"]["rgb"]]
        else:
            raise KeyError("Observation must contain `images` or head/left/right camera RGB arrays.")
        return self._image_tensor_from_arrays([np.asarray(image, dtype=np.uint8) for image in images])

    def normalize_state(self, raw_state: np.ndarray) -> torch.Tensor:
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw_state.shape[0] != self.raw_state_dim:
            raise ValueError(f"Expected raw state dim {self.raw_state_dim}, got {raw_state.shape[0]}")

        if self.normalization_mode == "zscore_14d":
            normed = _zscore_normalize(raw_state, self.norm_stats["state"])
            model_value = normed[..., : self.proprio_dim]
            return torch.from_numpy(model_value).unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)

        with_term = np.concatenate([_arrange_raw_state(raw_state), np.zeros(1, dtype=np.float32)], axis=0)
        normed = _quantile_normalize(with_term, self.norm_stats["state"])
        model_value = _select_model_dims(normed, self.proprio_dim)
        return torch.from_numpy(model_value).unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)

    def normalize_action_condition(
        self,
        action_abs: np.ndarray,
        state_raw: np.ndarray,
        *,
        action_start: int = 0,
        episode_length: Optional[int] = None,
        done_tail_length: int = 15,
    ) -> torch.Tensor:
        """Normalize an absolute qpos sequence into DW05 model action space."""
        action_abs = np.asarray(action_abs, dtype=np.float32)
        state_raw = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        if action_abs.ndim != 2:
            raise ValueError(f"Expected action_abs [T,D], got {action_abs.shape}")
        if action_abs.shape[-1] != self.raw_action_dim:
            raise ValueError(f"Expected raw action dim {self.raw_action_dim}, got {action_abs.shape[-1]}")
        if state_raw.shape[0] != self.raw_state_dim:
            raise ValueError(f"Expected raw state dim {self.raw_state_dim}, got {state_raw.shape[0]}")

        if self.normalization_mode == "zscore_14d":
            if self.action_condition_mode != "absolute":
                delta = action_abs - state_raw.reshape(1, -1)
                for dim in ROBOTWIN_NON_DELTA_DIMS:
                    if dim < delta.shape[-1]:
                        delta[:, dim] = action_abs[:, dim]
                action_value = delta
            else:
                action_value = action_abs
            normed = _zscore_normalize(action_value, self.norm_stats["action"])
            model_value = normed[..., : self.action_dim]
            return torch.from_numpy(model_value).to(device=self.model.device, dtype=self.model.torch_dtype)

        delta = action_abs - state_raw.reshape(1, -1)
        for dim in ROBOTWIN_NON_DELTA_DIMS:
            if dim < delta.shape[-1]:
                delta[:, dim] = action_abs[:, dim]

        arranged = np.stack([_arrange_raw_state(row) for row in delta], axis=0)
        term = np.zeros((arranged.shape[0], 1), dtype=np.float32)
        if episode_length is not None and done_tail_length > 0:
            done_from = max(0, int(episode_length) - int(done_tail_length))
            indices = int(action_start) + np.arange(arranged.shape[0])
            term[indices >= done_from, 0] = 1.0
        with_term = np.concatenate([arranged, term], axis=-1)
        normed = _quantile_normalize(with_term, self.norm_stats["action"])
        model_value = _select_model_dims(normed, self.action_dim)
        return torch.from_numpy(model_value).to(device=self.model.device, dtype=self.model.torch_dtype)

    def denormalize_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert model action output back to raw RobotWin 14D delta/target space."""
        if isinstance(action, torch.Tensor):
            action_np = action.detach().to(dtype=torch.float32, device="cpu").numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 2:
            action_np = action_np[None]
        if action_np.ndim != 3:
            raise ValueError(f"Expected action [B,T,D] or [T,D], got {action_np.shape}")

        if self.normalization_mode == "zscore_14d":
            action_norm = action_np[..., : self.raw_action_dim]
            return _zscore_denormalize(action_norm, self.norm_stats["action"])

        stats_dim = int(self.norm_stats["action"]["q01"].shape[0])
        arranged_norm = _scatter_model_dims(action_np, stats_dim)
        arranged = _quantile_denormalize(arranged_norm, self.norm_stats["action"])
        return _reverse_arranged_state(arranged, output_dim=self.raw_action_dim)

    def infer_action_chunk(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self.build_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self.normalize_state(state_vector)
        prompt = self.format_prompt(instruction)

        infer_t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=prompt,
                input_image=image_tensor,
                action_horizon=int(self.config.action_horizon),
                proprio=proprio,
                num_inference_steps=int(self.config.num_inference_steps),
                sigma_shift=self.config.sigma_shift,
                seed=self.config.seed,
                rand_device=self.config.rand_device,
                tiled=bool(self.config.tiled),
            )
        self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]
        self.last_raw_action = action_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
        return self.denormalize_action(action_tensor)[0]

    def infer_action_chunk_with_video(self, observation: dict[str, Any], instruction: str) -> tuple[np.ndarray, list[Image.Image]]:
        image_tensor = self.build_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self.normalize_state(state_vector)
        prompt = self.format_prompt(instruction)
        with torch.no_grad():
            pred = self.model.infer_joint(
                prompt=prompt,
                input_image=image_tensor,
                num_video_frames=int(self.config.num_video_frames),
                action_horizon=int(self.config.action_horizon),
                proprio=proprio,
                num_inference_steps=int(self.config.num_inference_steps),
                sigma_shift=self.config.sigma_shift,
                seed=self.config.seed,
                rand_device=self.config.rand_device,
                tiled=bool(self.config.tiled),
                test_action_with_infer_action=False,
            )
        action_tensor = pred["action"]
        self.last_raw_action = action_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
        return self.denormalize_action(action_tensor)[0], pred["video"]

    def rollout_video_with_actions(
        self,
        *,
        prompt: str,
        init_image_tensor: torch.Tensor,
        action_abs: np.ndarray,
        state_abs: Optional[np.ndarray] = None,
        max_rollouts: Optional[int] = None,
        fps_stride: int = 4,
    ) -> list[Image.Image]:
        """Generate an open-loop video by conditioning on an absolute qpos sequence."""
        all_frames: list[Image.Image] = []
        cur_image = init_image_tensor.to(device=self.model.device, dtype=self.model.torch_dtype)
        action_abs = np.asarray(action_abs, dtype=np.float32)
        if state_abs is not None:
            state_abs = np.asarray(state_abs, dtype=np.float32)
            if state_abs.ndim == 1:
                state_abs = np.repeat(state_abs.reshape(1, -1), len(action_abs), axis=0)
            if state_abs.ndim != 2 or state_abs.shape[-1] != self.raw_state_dim:
                raise ValueError(f"Expected state_abs [T,{self.raw_state_dim}], got {state_abs.shape}")
        action_offset = 0
        rollout = 0
        chunk_size = int(self.config.action_horizon)
        while action_offset < len(action_abs):
            if max_rollouts is not None and rollout >= max_rollouts:
                break
            chunk = action_abs[action_offset : action_offset + chunk_size]
            if len(chunk) == 0:
                break
            if len(chunk) < chunk_size:
                pad = np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            state = action_abs[action_offset]
            if state_abs is not None and len(state_abs):
                state = state_abs[min(action_offset, len(state_abs) - 1)]
            action_tensor = self.normalize_action_condition(
                chunk,
                state,
                action_start=action_offset,
                episode_length=len(action_abs),
            )
            proprio = self.normalize_state(state)
            with torch.no_grad():
                pred = self.model.infer_joint(
                    prompt=prompt,
                    input_image=cur_image,
                    num_video_frames=int(self.config.num_video_frames),
                    action_horizon=chunk_size,
                    action=action_tensor,
                    proprio=proprio,
                    num_inference_steps=int(self.config.num_inference_steps),
                    sigma_shift=self.config.sigma_shift,
                    seed=self.config.seed,
                    rand_device=self.config.rand_device,
                    tiled=bool(self.config.tiled),
                    test_action_with_infer_action=False,
                )
            frames = pred["video"]
            all_frames.extend(frames if rollout == 0 else frames[1:])
            cur_image = pil_to_model_tensor(frames[-1], self.model.device, self.model.torch_dtype)
            action_offset += max(1, int(fps_stride)) * (int(self.config.num_video_frames) - 1)
            rollout += 1
        return all_frames

    def _fill_action_queue(self, observation: dict[str, Any], instruction: str) -> None:
        action_chunk = self.infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(int(self.config.replan_steps), action_chunk.shape[0])
        base_state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if self.config.delta_action:
            cumsum = np.cumsum(action_chunk[:n_exec], axis=0)
            for idx in range(n_exec):
                self.pending_actions.append((base_state + cumsum[idx]).astype(np.float32))
        elif self.config.delta_first_frame and self.action_condition_mode != "absolute":
            for idx in range(n_exec):
                abs_action = action_chunk[idx] + base_state
                for dim in ROBOTWIN_NON_DELTA_DIMS:
                    if dim < abs_action.shape[0]:
                        abs_action[dim] = action_chunk[idx, dim]
                self.pending_actions.append(abs_action.astype(np.float32))
        else:
            for idx in range(n_exec):
                self.pending_actions.append(np.asarray(action_chunk[idx], dtype=np.float32))

    def step(self, task_env: Any, observation: Optional[dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required when the action queue is empty.")
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)
        if not self.pending_actions:
            return
        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter()
        task_env.take_action(action, action_type="qpos")
        self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1


def pil_to_model_tensor(frame: Image.Image, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor * (2.0 / 255.0) - 1.0).to(device=device, dtype=dtype)


def resize_crop_tensor(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize tensor with center crop; useful for custom external clients."""
    squeezed = tensor.ndim == 3
    if squeezed:
        tensor = tensor.unsqueeze(0)
    _, _, height, width = tensor.shape
    scale = max(target_w / width, target_h / height)
    new_h = max(target_h, int(round(height * scale)))
    new_w = max(target_w, int(round(width * scale)))
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = max(0, (new_h - target_h) // 2)
    left = max(0, (new_w - target_w) // 2)
    tensor = tensor[..., top : top + target_h, left : left + target_w]
    return tensor.squeeze(0) if squeezed else tensor


__all__ = [
    "DW05RobotWinPolicy",
    "DW05RobotWinPolicyConfig",
    "ROBOTWIN_PROMPT_FORMAT",
    "compose_robotwin_image",
    "parse_bool",
    "parse_optional_float",
    "parse_optional_int",
    "pil_to_model_tensor",
]
