#!/usr/bin/env python
"""Batch WorldArena video rollout generation with DW05."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dexbotic.data.dataset.dw05.data_source import ROBOTWIN_META
from dexbotic.policy.dw05_policy import (
    DW05RobotWinPolicy,
    DW05RobotWinPolicyConfig,
    ROBOTWIN_PROMPT_FORMAT,
    compose_robotwin_image,
)


DEFAULT_DATA_ROOT = Path(os.environ.get("DW05_WORLDARENA_DATA_ROOT", "./data/robotwin_val_worldarena"))
DEFAULT_LEFT_MP4 = os.environ.get("DW05_LEFT_MP4", "")
DEFAULT_RIGHT_MP4 = os.environ.get("DW05_RIGHT_MP4", "")
DEFAULT_OUTPUT_ROOT = Path("./evaluate_results/worldarena/dw05")
DEFAULT_CKPT = os.environ.get("DW05_CKPT_PATH", "")
DEFAULT_V5_STATS = os.environ.get("DW05_NORM_STATS_PATH", "")


def _env_default(name: str, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def parse_episodes(raw: str | None) -> list[int]:
    if not raw:
        return list(range(1, 251))
    episodes: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            episodes.extend(range(start, end + 1))
        else:
            episodes.append(int(part))
    return episodes


def instruction_roots(data_root: Path, raw: str | None) -> list[Path]:
    if raw:
        roots = [Path(x.strip()).expanduser() for x in raw.split(",") if x.strip()]
        if not roots:
            raise ValueError("--instr-roots did not contain any paths.")
        return roots
    return [data_root / "instructions/fixed_scene_task"]


def preprocess_instruction(instruction: str) -> str:
    task = instruction.strip().strip("\"'“”")
    marker = "enters the frame to "
    marker_idx = task.lower().rfind(marker)
    if marker_idx >= 0:
        task = task[marker_idx + len(marker) :]
    return task.strip().strip("\"'“”")


def shard_episodes(episodes: list[int], process_index: int, world_size: int) -> list[int]:
    if world_size < 1:
        raise ValueError(f"--world-size must be >= 1, got {world_size}")
    if process_index < 0 or process_index >= world_size:
        raise ValueError(f"--process-index must be in [0, {world_size}), got {process_index}")
    return episodes[process_index::world_size]


def read_first_frame(mp4_path: str) -> np.ndarray:
    reader = imageio.get_reader(mp4_path)
    try:
        return np.asarray(reader.get_data(0))
    finally:
        reader.close()


def build_image_tensor(policy: DW05RobotWinPolicy, head: np.ndarray, left: np.ndarray, right: np.ndarray) -> torch.Tensor:
    image = compose_robotwin_image(
        [head, left, right],
        layout=policy.config.image_layout,
        image_size_hw=policy.config.image_size_hw,
    )
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return (tensor * (2.0 / 255.0) - 1.0).to(device=policy.model.device, dtype=policy.model.torch_dtype)


def split_state_action(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[-1] != 14:
        raise ValueError(f"Expected qpos [T,14], got {qpos.shape}")
    if len(qpos) <= 1:
        return qpos[:1].copy(), qpos.copy()
    return qpos[:-1].copy(), qpos[1:].copy()


def save_video(frames: list[Image.Image], output_dir: Path, name: str, fps: int = 24, head_height: int = 256) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{name}.mp4"
    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    head_frames = [
        np.asarray(Image.fromarray(arr[:head_height]).resize((640, 480), Image.Resampling.BILINEAR))
        for arr in arrays
    ]
    imageio.mimsave(str(video_path), head_frames, fps=fps)
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(_env_default("CKPT_PATH", DEFAULT_CKPT)))
    parser.add_argument("--norm-stats-path", default=str(_env_default("DW05_NORM_STATS_PATH", DEFAULT_V5_STATS)))
    parser.add_argument("--model-base-path", default=str(_env_default("DW05_MODEL_BASE_PATH", "")) or None)
    parser.add_argument("--output-root", default=str(_env_default("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--output-name", default=str(_env_default("OUTPUT_NAME", "dw05_worldarena")))
    parser.add_argument("--data-root", default=str(_env_default("WORLDARENA_DATA_ROOT", DEFAULT_DATA_ROOT)))
    parser.add_argument("--hdf5-root", default=str(_env_default("WORLDARENA_HDF5_ROOT", "")))
    parser.add_argument("--head-root", default=str(_env_default("WORLDARENA_HEAD_ROOT", "")))
    parser.add_argument("--instr-roots", default=str(_env_default("WORLDARENA_INSTR_ROOTS", "")))
    parser.add_argument("--left-mp4", default=str(_env_default("WORLDARENA_LEFT_MP4", DEFAULT_LEFT_MP4)))
    parser.add_argument("--right-mp4", default=str(_env_default("WORLDARENA_RIGHT_MP4", DEFAULT_RIGHT_MP4)))
    parser.add_argument("--episodes", default=str(_env_default("EPISODES", "")) or None)
    parser.add_argument("--process-index", type=int, default=int(_env_default("PROCESS_INDEX", os.environ.get("RANK", 0))))
    parser.add_argument("--world-size", type=int, default=int(_env_default("WORLD_SIZE", 1)))
    parser.add_argument("--device", default=str(_env_default("DEVICE", "cuda:0")))
    parser.add_argument("--mixed-precision", default=str(_env_default("MIXED_PRECISION", "bf16")), choices=["no", "fp16", "bf16"])
    parser.add_argument("--num-inference-steps", type=int, default=int(_env_default("NUM_INFERENCE_STEPS", 20)))
    parser.add_argument("--action-horizon", type=int, default=int(_env_default("ACTION_HORIZON", 32)))
    parser.add_argument("--num-video-frames", type=int, default=int(_env_default("NUM_VIDEO_FRAMES", 9)))
    parser.add_argument("--seed", type=int, default=int(_env_default("SEED", 42)))
    parser.add_argument("--max-rollouts", type=int, default=int(_env_default("MAX_ROLLOUTS", 0)))
    parser.add_argument("--normalization-mode", default=str(_env_default("DW05_NORMALIZATION_MODE", "auto")), choices=["auto", "zscore_14d", "dw05_quantile"])
    parser.add_argument("--image-layout", default=str(_env_default("DW05_IMAGE_LAYOUT", "auto")))
    parser.add_argument("--policy-action-condition-mode", default=str(_env_default("DW05_ACTION_CONDITION_MODE", "auto")), choices=["auto", "absolute", "delta_first_frame"])
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--head-height", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    hdf5_root = Path(args.hdf5_root).expanduser() if args.hdf5_root else data_root / "data/fixed_scene_task"
    head_root = Path(args.head_root).expanduser() if args.head_root else data_root / "first_frame/fixed_scene_task"
    roots = instruction_roots(data_root, args.instr_roots)
    if len(roots) != 1:
        raise ValueError("WorldArena DW05 eval expects exactly one instruction root per process.")
    episodes = shard_episodes(parse_episodes(args.episodes), args.process_index, args.world_size)
    output_root = Path(args.output_root).expanduser()
    output_dir = output_root / args.output_name

    policy = DW05RobotWinPolicy(
        DW05RobotWinPolicyConfig(
            checkpoint_path=args.ckpt,
            norm_stats_path=args.norm_stats_path,
            model_base_path=args.model_base_path,
            device=args.device,
            mixed_precision=args.mixed_precision,
            action_horizon=args.action_horizon,
            replan_steps=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            num_video_frames=args.num_video_frames,
            seed=args.seed,
            rand_device="cpu",
            delta_first_frame=False,
            image_layout=args.image_layout,
            normalization_mode=args.normalization_mode,
            action_condition_mode=args.policy_action_condition_mode,
            load_text_encoder=True,
        )
    )
    left_frame = read_first_frame(args.left_mp4)
    right_frame = read_first_frame(args.right_mp4)
    max_rollouts = args.max_rollouts if args.max_rollouts > 0 else None

    saved: list[str] = []
    for episode in tqdm(episodes, desc="WorldArena"):
        head_path = head_root / f"episode{episode}.png"
        hdf5_path = hdf5_root / f"episode{episode}.hdf5"
        instr_path = roots[0] / f"episode{episode}.json"
        head = np.asarray(Image.open(head_path).convert("RGB"))
        init_image = build_image_tensor(policy, head, left_frame, right_frame)
        with h5py.File(hdf5_path, "r") as handle:
            qpos = handle["joint_action/vector"][:].astype(np.float32)
        state_np, action_np = split_state_action(qpos)
        task = preprocess_instruction(json.loads(instr_path.read_text(encoding="utf-8"))["instruction"])
        prompt = ROBOTWIN_PROMPT_FORMAT.format(task=task)
        frames = policy.rollout_video_with_actions(
            prompt=prompt,
            init_image_tensor=init_image,
            action_abs=action_np,
            state_abs=state_np,
            max_rollouts=max_rollouts,
        )
        saved.append(str(save_video(frames, output_dir, f"episode{episode}", fps=args.fps, head_height=args.head_height)))

    summary = {
        "ckpt": args.ckpt,
        "norm_stats_path": args.norm_stats_path,
        "data_root": str(data_root),
        "hdf5_root": str(hdf5_root),
        "head_root": str(head_root),
        "instruction_root": str(roots[0]),
        "episodes": episodes,
        "process_index": args.process_index,
        "world_size": args.world_size,
        "output_dir": str(output_dir),
        "saved_videos": saved,
        "num_saved_videos": len(saved),
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "max_rollouts": max_rollouts,
        "normalization_mode": policy.normalization_mode,
        "image_layout": policy.config.image_layout,
        "policy_action_condition_mode": policy.action_condition_mode,
    }
    summary_name = f"{args.output_name}_summary.json"
    if args.world_size > 1:
        summary_name = f"{args.output_name}_summary_rank{args.process_index}.json"
    summary_path = output_root / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
