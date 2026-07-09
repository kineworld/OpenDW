"""RoboTwin official-eval policy adapter for DW05.

The RoboTwin evaluator imports this package from ``RoboTwin/policy`` and calls
``get_model``/``eval``/``reset_model``.  Keep this file small: model loading,
normalization, image composition, and replan semantics live in
``dexbotic.policy.dw05_policy`` so real-robot and WorldArena entrypoints share
the same behavior.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

from dexbotic.data.dataset.dw05.data_source import ROBOTWIN_META
from dexbotic.policy.dw05_policy import (
    DW05RobotWinPolicy,
    DW05RobotWinPolicyConfig,
    parse_bool,
    parse_optional_float,
    parse_optional_int,
)


logger = logging.getLogger(__name__)


class DW05RobotWinDeployPolicy(DW05RobotWinPolicy):
    """Thin subclass used by RoboTwin's policy registry."""

    pass


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _resolve_path(value: Any, *, base: Path = Path.cwd()) -> Optional[str]:
    if _is_none_like(value):
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _resolve_norm_stats_path(args: dict[str, Any], checkpoint_path: str) -> str:
    explicit = _resolve_path(args.get("norm_stats_path") or args.get("dataset_stats_path"))
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    meta_path = ROBOTWIN_META.get("norm_stats_path")
    if meta_path:
        candidates.append(Path(str(meta_path)))
    for parent in list(Path(checkpoint_path).expanduser().resolve().parents)[:4]:
        candidates.append(parent / "norm_stats.json")

    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "Failed to resolve DW05 norm stats. Pass dataset_stats_path=/path/to/norm_stats.json "
        "or norm_stats_path=/path/to/norm_stats.json."
    )


def get_model(usr_args: dict[str, Any]):
    """Build a DW05 policy for RoboTwin's official evaluation harness."""
    checkpoint_path = _resolve_path(usr_args.get("ckpt_setting") or usr_args.get("ckpt"))
    if checkpoint_path is None:
        raise ValueError("`ckpt_setting` or `ckpt` is required for DW05 RoboTwin evaluation.")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = str(usr_args.get("device") or "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"

    action_horizon = parse_optional_int(usr_args.get("action_horizon")) or 32
    replan_steps = parse_optional_int(usr_args.get("replan_steps")) or min(8, action_horizon)
    norm_stats_path = _resolve_norm_stats_path(usr_args, checkpoint_path)
    model_base_path = _resolve_path(usr_args.get("model_base_path") or os.environ.get("DW05_MODEL_BASE_PATH"))

    return DW05RobotWinDeployPolicy(
        DW05RobotWinPolicyConfig(
            checkpoint_path=checkpoint_path,
            norm_stats_path=norm_stats_path,
            model_base_path=model_base_path,
            device=device,
            mixed_precision=str(usr_args.get("mixed_precision") or "bf16"),
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            num_inference_steps=parse_optional_int(usr_args.get("num_inference_steps")) or 10,
            num_video_frames=parse_optional_int(usr_args.get("num_video_frames")) or 9,
            sigma_shift=parse_optional_float(usr_args.get("sigma_shift")),
            seed=parse_optional_int(usr_args.get("seed")),
            text_cfg_scale=float(usr_args.get("text_cfg_scale") or 1.0),
            action_cfg_scale=float(usr_args.get("action_cfg_scale") or 1.0),
            negative_prompt=str(usr_args.get("negative_prompt") or ""),
            rand_device=str(usr_args.get("rand_device") or "cpu"),
            tiled=parse_bool(usr_args.get("tiled"), False),
            delta_action=parse_bool(usr_args.get("delta_action"), False),
            delta_first_frame=parse_bool(usr_args.get("delta_first_frame"), True),
            image_layout=str(usr_args.get("image_layout") or "auto"),
            normalization_mode=str(usr_args.get("normalization_mode") or "auto"),
            action_condition_mode=str(usr_args.get("action_condition_mode") or "auto"),
            load_text_encoder=parse_bool(usr_args.get("load_text_encoder"), True),
        )
    )


def encode_obs(observation: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return observation


def eval(TASK_ENV, model, observation: Optional[dict[str, Any]]):
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()
