#!/usr/bin/env python
"""Run one RoboTwin task with the migrated DW05 policy.

This script is intentionally a small bridge around the official RoboTwin
``script/eval_policy.py``.  It creates/validates the ``policy/dw05_policy``
symlink and forwards DW05 runtime settings through RoboTwin's override list.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_NAME = "dw05_policy"
POLICY_SOURCE_DIR = REPO_ROOT / "playground" / "benchmarks" / "robotwin2" / POLICY_NAME


def _resolve_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _ensure_policy_symlink(robotwin_root: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")
    target = policy_root / POLICY_NAME
    source = POLICY_SOURCE_DIR.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"DW05 policy source directory not found: {source}")
    if not target.exists() and not target.is_symlink():
        target.symlink_to(source, target_is_directory=True)
        return target
    if target.is_symlink() and target.resolve() == source:
        return target
    raise RuntimeError(f"Policy path conflict: {target}. Remove it or point it to {source}.")


def _append_override(overrides: list[str], key: str, value, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    if isinstance(value, bool):
        text = "True" if value else "False"
    elif value is None:
        text = "None"
    else:
        text = repr(str(value))
    overrides.extend([f"--{key}", text])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, help="Path to external RoboTwin checkout.")
    parser.add_argument("--ckpt", required=True, help="DW05 checkpoint path.")
    parser.add_argument("--task-name", required=True, help="RoboTwin task name, e.g. press_stapler.")
    parser.add_argument("--task-config", default="demo_clean", help="RoboTwin task config: demo_clean/demo_randomized.")
    parser.add_argument("--norm-stats-path", default=None, help="DW05 norm_stats.json path.")
    parser.add_argument("--model-base-path", default=None, help="Local Wan2.2 model root.")
    parser.add_argument("--output-dir", default="./evaluate_results/robotwin/dw05", help="Evaluation output root.")
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value for the RoboTwin worker.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-num-episodes", type=int, default=50)
    parser.add_argument("--instruction-type", default="seen")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--delta-first-frame", action="store_true", default=True)
    parser.add_argument("--no-delta-first-frame", dest="delta_first_frame", action="store_false")
    parser.add_argument("--delta-action", action="store_true", default=False)
    parser.add_argument("--normalization-mode", default="auto", choices=["auto", "zscore_14d", "dw05_quantile"])
    parser.add_argument("--image-layout", default="auto")
    parser.add_argument("--policy-action-condition-mode", default="auto", choices=["auto", "absolute", "delta_first_frame"])
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--extra-override", action="append", default=[], help="Raw RoboTwin --key value pair as 'key=value'.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robotwin_root = _resolve_path(args.robotwin_root)
    ckpt = _resolve_path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    _ensure_policy_symlink(robotwin_root)

    output_dir = _resolve_path(args.output_dir)
    run_dir = output_dir / ckpt.stem / datetime.now().strftime("%Y%m%d_%H%M%S") / args.task_name
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides: list[str] = []
    _append_override(overrides, "task_name", args.task_name)
    _append_override(overrides, "task_config", args.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt))
    _append_override(overrides, "policy_name", POLICY_NAME)
    _append_override(overrides, "seed", args.seed)
    _append_override(overrides, "instruction_type", args.instruction_type)
    _append_override(overrides, "eval_num_episodes", args.eval_num_episodes)
    _append_override(overrides, "eval_output_dir", str(run_dir))
    _append_override(overrides, "mixed_precision", args.mixed_precision)
    _append_override(overrides, "device", args.device)
    _append_override(overrides, "dataset_stats_path", args.norm_stats_path)
    _append_override(overrides, "norm_stats_path", args.norm_stats_path)
    _append_override(overrides, "model_base_path", args.model_base_path)
    _append_override(overrides, "action_horizon", args.action_horizon)
    _append_override(overrides, "replan_steps", args.replan_steps)
    _append_override(overrides, "num_inference_steps", args.num_inference_steps)
    _append_override(overrides, "delta_first_frame", args.delta_first_frame)
    _append_override(overrides, "delta_action", args.delta_action)
    _append_override(overrides, "normalization_mode", args.normalization_mode)
    _append_override(overrides, "image_layout", args.image_layout)
    _append_override(overrides, "action_condition_mode", args.policy_action_condition_mode)
    _append_override(overrides, "text_cfg_scale", args.text_cfg_scale)
    _append_override(overrides, "rand_device", args.rand_device)
    _append_override(overrides, "tiled", args.tiled)
    for raw in args.extra_override:
        key, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(f"--extra-override must be key=value, got {raw!r}")
        _append_override(overrides, key, value, skip_none=False)

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    if args.model_base_path:
        env.setdefault("DW05_MODEL_BASE_PATH", str(_resolve_path(args.model_base_path)))
        env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(_resolve_path(args.model_base_path)))

    log_file = run_dir / "eval.log"
    print("Running:", " ".join(cmd))
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"RoboTwin eval failed with rc={return_code}. Log: {log_file}")
    print(f"RoboTwin eval finished. Log: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
