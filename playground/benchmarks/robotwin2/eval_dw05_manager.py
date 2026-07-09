#!/usr/bin/env python
"""Launch DW05 RoboTwin evaluations across tasks and GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
SINGLE_ENTRY = REPO_ROOT / "playground" / "benchmarks" / "robotwin2" / "eval_dw05_single.py"


@dataclass
class RunningJob:
    task_name: str
    phase: str
    gpu_id: int
    process: subprocess.Popen


def _read_tasks(path: Optional[str], raw_tasks: Optional[str]) -> list[str]:
    if raw_tasks:
        return [item.strip() for item in raw_tasks.split(",") if item.strip()]
    if not path:
        raise ValueError("Pass --tasks or --task-file.")
    task_path = Path(path).expanduser()
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    tasks = []
    for line in task_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tasks.append(line)
    if not tasks:
        raise ValueError(f"No tasks found in {task_path}")
    return tasks


def _parse_success_rate(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    value = None
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError:
            continue
    return value


def _mean(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--model-base-path", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task names.")
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--phases", default="clean,random", help="Comma-separated phases: clean,random.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1)
    parser.add_argument("--output-dir", default="./evaluate_results/robotwin/dw05")
    parser.add_argument("--eval-num-episodes", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--normalization-mode", default="auto", choices=["auto", "zscore_14d", "dw05_quantile"])
    parser.add_argument("--image-layout", default="auto")
    parser.add_argument("--policy-action-condition-mode", default="auto", choices=["auto", "absolute", "delta_first_frame"])
    parser.add_argument("--poll-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_gpus <= 0 or args.max_tasks_per_gpu <= 0:
        raise ValueError("--num-gpus and --max-tasks-per-gpu must be positive.")
    tasks = _read_tasks(args.task_file, args.tasks)
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    phase_config = {"clean": "demo_clean", "random": "demo_randomized"}
    for phase in phases:
        if phase not in phase_config:
            raise ValueError(f"Unsupported phase {phase!r}; expected one of {sorted(phase_config)}")

    output_root = Path(args.output_dir).expanduser().resolve() / Path(args.ckpt).stem / time.strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    pending = deque((task, phase) for task in tasks for phase in phases)
    running: list[RunningJob] = []
    rates: dict[str, dict[str, Optional[float]]] = {task: {phase: None for phase in phases} for task in tasks}

    def gpu_running_count(gpu_id: int) -> int:
        return sum(1 for job in running if job.gpu_id == gpu_id and job.process.poll() is None)

    def launch(task: str, phase: str, gpu_id: int) -> RunningJob:
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            "--robotwin-root",
            args.robotwin_root,
            "--ckpt",
            args.ckpt,
            "--task-name",
            task,
            "--task-config",
            phase_config[phase],
            "--output-dir",
            str(output_root),
            "--gpu-id",
            str(gpu_id),
            "--eval-num-episodes",
            str(args.eval_num_episodes),
            "--action-horizon",
            str(args.action_horizon),
            "--replan-steps",
            str(args.replan_steps),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--mixed-precision",
            args.mixed_precision,
            "--normalization-mode",
            args.normalization_mode,
            "--image-layout",
            args.image_layout,
            "--policy-action-condition-mode",
            args.policy_action_condition_mode,
        ]
        if args.norm_stats_path:
            cmd.extend(["--norm-stats-path", args.norm_stats_path])
        if args.model_base_path:
            cmd.extend(["--model-base-path", args.model_base_path])
        print(f"[launch] gpu={gpu_id} task={task} phase={phase}: {' '.join(cmd)}", flush=True)
        return RunningJob(task, phase, gpu_id, subprocess.Popen(cmd, cwd=str(REPO_ROOT)))

    def try_launch() -> None:
        for gpu_id in range(args.num_gpus):
            while pending and gpu_running_count(gpu_id) < args.max_tasks_per_gpu:
                task, phase = pending.popleft()
                running.append(launch(task, phase, gpu_id))

    try_launch()
    failures = []
    while running:
        for job in list(running):
            rc = job.process.poll()
            if rc is None:
                continue
            running.remove(job)
            if rc != 0:
                failures.append({"task": job.task_name, "phase": job.phase, "gpu_id": job.gpu_id, "return_code": rc})
            result_file = output_root / Path(args.ckpt).stem / job.task_name / f"_result_{phase_config[job.phase].replace('demo_', '')}.txt"
            rates[job.task_name][job.phase] = _parse_success_rate(result_file)
            try_launch()
        if running:
            time.sleep(args.poll_interval)

    csv_path = output_root / "summary.csv"
    json_path = output_root / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", *[f"{phase}_success_rate" for phase in phases]])
        for task in tasks:
            writer.writerow([task, *[rates[task][phase] for phase in phases]])
        writer.writerow(["__overall__", *[_mean([rates[task][phase] for task in tasks]) for phase in phases]])
    json_path.write_text(json.dumps({"rates": rates, "failures": failures}, indent=2), encoding="utf-8")
    print(f"Summary: {csv_path}\n{json_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} RoboTwin jobs failed. See {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
