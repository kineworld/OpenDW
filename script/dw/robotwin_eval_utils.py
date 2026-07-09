#!/usr/bin/env python3
"""RoboTwin evaluation helper commands for DW05.

These utilities are intentionally standalone: they allocate task/seed shards,
merge RoboTwin shard results, and aggregate timing logs without importing the
DW05 model stack.

Subcommands:
  emit-queue       allocate per-shard task/seed work items
  merge-results    merge _result_shard*.txt into _result.txt and _summary.txt
  analyze-timing   aggregate timing lines and per-step JSONL files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _read_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            seed = rec.get("seed")
            if seed is not None:
                out[int(seed)] = rec
    return out


def valid_seeds_for_task(cache_dir: Path, task: str) -> list[int]:
    recs = _read_jsonl(cache_dir / f"{task}.jsonl")
    return sorted([s for s, r in recs.items() if r.get("valid") is True])


def chunk(values: list[int], n: int) -> list[list[int]]:
    if n <= 0:
        return [values]
    q, r = divmod(len(values), n)
    out: list[list[int]] = []
    i = 0
    for k in range(n):
        size = q + (1 if k < r else 0)
        out.append(values[i:i + size])
        i += size
    return out


def emit_for_task(
    task: str,
    cache_dir: Optional[Path],
    total_episodes: int,
    total_workers: int,
    base_seed: int,
    require_cache: bool,
    seed_multiplier: int = 1,
    allow_uncached_candidates: bool = False,
) -> tuple[list[str], str]:
    mode = "legacy"
    seeds: Optional[list[int]] = None
    known_seeds: set[int] = set()
    seed_multiplier = max(seed_multiplier, 1)
    requested_candidates = max(total_episodes, total_episodes * seed_multiplier)

    if cache_dir is not None and cache_dir.is_dir():
        cache_records = _read_jsonl(cache_dir / f"{task}.jsonl")
        known_seeds = set(cache_records)
        valid = sorted([s for s, r in cache_records.items() if r.get("valid") is True])
        if len(valid) >= total_episodes:
            seeds = valid[:min(len(valid), requested_candidates)]
            mode = f"cache(valid={len(valid)},candidates={len(seeds)})"
        else:
            msg = (f"[seed_alloc] {task}: cache has {len(valid)} valid seeds "
                   f"(< requested {total_episodes}); falling back to legacy scan.")
            if require_cache and not allow_uncached_candidates:
                print(msg, file=sys.stderr)
                sys.exit(2)
            print(msg, file=sys.stderr)
            if valid:
                seeds = valid[:]

    if allow_uncached_candidates:
        seen = known_seeds | set(seeds or [])
        extra: list[int] = []
        seed = base_seed
        while len(seen) + len(extra) < requested_candidates:
            if seed not in seen:
                extra.append(seed)
            seed += 1
        seeds = (seeds or []) + extra
        mode = f"{mode}+uncached_candidates({len(seeds)})"

    lines: list[str] = []
    if seeds is not None:
        shard_base, shard_rem = divmod(total_episodes, total_workers)
        episode_offset = 0
        cursor = 0
        for shard_id in range(total_workers):
            target_count = shard_base + (1 if shard_id < shard_rem else 0)
            candidate_count = max(target_count, target_count * seed_multiplier)
            ch = seeds[cursor:cursor + candidate_count]
            cursor += candidate_count
            if len(ch) < target_count:
                print(
                    f"[seed_alloc] {task}: shard {shard_id} has only {len(ch)} "
                    f"candidate seeds for {target_count} requested episodes",
                    file=sys.stderr,
                )
                if require_cache and not allow_uncached_candidates:
                    sys.exit(2)
            if not ch:
                lines.append(f"{task}|{base_seed}|{target_count}|{shard_id}|{episode_offset}|0|")
                episode_offset += target_count
                continue
            seed_csv = ",".join(str(s) for s in ch)
            lines.append(f"{task}|{ch[0]}|{target_count}|{shard_id}|{episode_offset}|0|{seed_csv}")
            episode_offset += target_count
    else:
        shard_base, shard_rem = divmod(total_episodes, total_workers)
        episode_offset = 0
        for shard_id in range(total_workers):
            test_num = shard_base + (1 if shard_id < shard_rem else 0)
            local_st_seed = base_seed + episode_offset
            lines.append(f"{task}|{local_st_seed}|{test_num}|{shard_id}|{episode_offset}|0|")
            episode_offset += test_num
    return lines, mode


def cmd_emit_queue(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir.strip() else None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    report_lines: list[str] = []
    n_cache, n_legacy = 0, 0
    for task in tasks:
        lines, mode = emit_for_task(
            task=task,
            cache_dir=cache_dir,
            total_episodes=args.total_episodes,
            total_workers=args.total_workers,
            base_seed=args.base_seed,
            require_cache=args.require_cache,
            seed_multiplier=args.seed_multiplier,
            allow_uncached_candidates=args.allow_uncached_candidates,
        )
        for ln in lines:
            print(ln)
        if mode.startswith("cache"):
            n_cache += 1
        else:
            n_legacy += 1
        report_lines.append(f"{task:40s} {mode}")

    report = (
        f"=== seed_alloc summary: cache-mode tasks={n_cache} "
        f"legacy-mode tasks={n_legacy} (cache_dir={cache_dir})\n"
        + "\n".join(report_lines)
    )
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
    print(report, file=sys.stderr)
    return 0


@dataclass
class ShardResult:
    shard_id: str
    path: Path
    suc: int
    total: int
    timestamp: str
    instruction_type: str

    @property
    def rate(self) -> float:
        return self.suc / self.total if self.total else 0.0


def _parse_shard_file(path: Path) -> ShardResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    suc_m = re.search(r"suc=(\d+)", text)
    total_m = re.search(r"total=(\d+)", text)
    ts_m = re.search(r"Timestamp:\s*(.+)", text)
    it_m = re.search(r"Instruction Type:\s*(.+)", text)
    shard_id = path.stem.replace("_result_shard", "")
    return ShardResult(
        shard_id=shard_id,
        path=path,
        suc=int(suc_m.group(1)) if suc_m else 0,
        total=int(total_m.group(1)) if total_m else 0,
        timestamp=ts_m.group(1).strip() if ts_m else "",
        instruction_type=it_m.group(1).strip() if it_m else "",
    )


def find_result_dirs(log_root: Path, task_filter: str | None = None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(log_root.glob("**/_result_shard*.txt")):
        if task_filter and task_filter not in path.parts:
            continue
        result_dir = path.parent
        rel = path.relative_to(log_root)
        task_name = rel.parts[0] if rel.parts else "unknown"
        if task_name not in out:
            out[task_name] = result_dir
    return out


def merge_task(result_dir: Path, cleanup: bool) -> tuple[int, int, list[ShardResult]]:
    shard_files = sorted(result_dir.glob("_result_shard*.txt"))
    if not shard_files:
        return 0, 0, []
    shards = [_parse_shard_file(p) for p in shard_files]
    suc = sum(s.suc for s in shards)
    total = sum(s.total for s in shards)
    rate = suc / total if total else 0.0
    ts = next((s.timestamp for s in shards if s.timestamp), "")
    itype = next((s.instruction_type for s in shards if s.instruction_type), "")
    merged = result_dir / "_result.txt"
    merged.write_text(
        f"Timestamp: {ts}\n\nInstruction Type: {itype}\n\n"
        f"suc={suc}\n"
        f"total={total}\n"
        f"rate={rate}\n",
        encoding="utf-8",
    )
    if cleanup:
        for p in shard_files:
            p.unlink()
    return suc, total, shards


def cmd_merge_results(args: argparse.Namespace) -> int:
    log_root = args.log_root.resolve()
    if not log_root.is_dir():
        raise SystemExit(f"Not a directory: {log_root}")

    task_dirs = find_result_dirs(log_root, args.task)
    if not task_dirs:
        print(f"No _result_shard*.txt under {log_root}")
        if args.expect_tasks is not None or args.expect_total is not None:
            return 3
        return 0

    rows: list[tuple[str, int, int, float, int]] = []
    for task_name in sorted(task_dirs):
        result_dir = task_dirs[task_name]
        suc, total, shards = merge_task(result_dir, cleanup=args.cleanup)
        if not shards:
            print(f"  skip {task_name}: no shard files in {result_dir}")
            continue
        rate = suc / total if total else 0.0
        rows.append((task_name, suc, total, rate, len(shards)))
        print(f"  merged {task_name}: {suc}/{total} = {rate:.1%} ({len(shards)} shards) -> {result_dir / '_result.txt'}")

    if rows:
        summary_path = args.summary or (log_root / "_summary.txt")
        lines = [
            f"# RoboTwin merged success rates: {log_root.name}",
            f"# tasks={len(rows)}",
            "task\tsuc\ttotal\trate\tshards",
        ]
        for task_name, suc, total, rate, n_shards in rows:
            lines.append(f"{task_name}\t{suc}\t{total}\t{rate:.6f}\t{n_shards}")
        grand_suc = sum(r[1] for r in rows)
        grand_total = sum(r[2] for r in rows)
        grand_rate = grand_suc / grand_total if grand_total else 0.0
        lines.append(f"__ALL__\t{grand_suc}\t{grand_total}\t{grand_rate:.6f}\t{len(rows)}")
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSummary: {grand_suc}/{grand_total} = {grand_rate:.1%} across {len(rows)} tasks")
        print(f"Wrote {summary_path}")
        if args.expect_tasks is not None and len(rows) != args.expect_tasks:
            print(
                f"ERROR: expected {args.expect_tasks} merged tasks, got {len(rows)}",
                file=sys.stderr,
            )
            return 3
        if args.expect_total is not None and grand_total != args.expect_total:
            print(
                f"ERROR: expected {args.expect_total} total episodes, got {grand_total}",
                file=sys.stderr,
            )
            return 4
    return 0


LINE_RE = re.compile(
    r"\[timing(?::\w+)?\]\s+(?:(?:===\s+)?(?P<title>[^=]+?)\s+===\s+)?"
    r"(?P<name>[\w./_-]+):\s+total=(?P<total>[\d.]+)s\s+count=(?P<count>\d+)\s+avg=(?P<avg>[\d.]+)s\s+\((?P<pct>[\d.]+)%\)"
)
STEP_KEYS = (
    "get_obs",
    "encode_obs",
    "vla_encode_images",
    "vla_http_post",
    "vla_parse_response",
    "convert_action",
    "sim_take_action",
    "policy_step_wall",
)


def parse_log(path: Path) -> dict[str, tuple[float, int]]:
    agg: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        agg[m.group("name")].append((float(m.group("total")), int(m.group("count"))))
    return {name: (sum(t for t, _ in items), sum(c for _, c in items)) for name, items in agg.items()}


def parse_jsonl(path: Path) -> dict[str, tuple[float, int]]:
    agg: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in STEP_KEYS:
            if key in rec:
                agg[key].append((float(rec[key]), 1))
    return {key: (sum(t for t, _ in values), len(values)) for key, values in agg.items()}


def print_aggregate(global_agg: dict[str, list[tuple[float, int]]], title: str) -> None:
    rows = []
    for name, items in global_agg.items():
        total = sum(t for t, _ in items)
        count = sum(c for _, c in items)
        rows.append((total, name, count, total / count if count else 0.0))
    rows.sort(reverse=True)
    if not rows:
        print(f"No timing data in {title}")
        return
    grand = sum(r[0] for r in rows)
    print(f"=== {title} ===")
    for total, name, count, avg in rows:
        pct = total / grand * 100 if grand else 0
        print(f"  {name:40s} total={total:8.1f}s  count={count:6d}  avg={avg:.3f}s  ({pct:5.1f}%)")


def cmd_analyze_timing(args: argparse.Namespace) -> int:
    if args.jsonl:
        agg = parse_jsonl(args.jsonl)
        print_aggregate({k: [(t, c)] for k, (t, c) in agg.items()}, f"step jsonl {args.jsonl.name}")
        return 0
    if not args.path:
        raise SystemExit("path or --jsonl required")

    if args.path.is_file():
        logs = [args.path]
        jsonls = sorted(args.path.parent.glob(args.path.stem.replace(".log", "") + "_steps.jsonl"))
    else:
        logs = sorted(args.path.rglob("*_shard*.log"))
        jsonls = sorted(args.path.rglob("*_steps.jsonl"))
        if not logs:
            logs = sorted(args.path.rglob("*.log"))

    global_agg: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for log in logs:
        for name, (total, count) in parse_log(log).items():
            global_agg[name].append((total, count))
    for jsonl_path in jsonls:
        for name, (total, count) in parse_jsonl(jsonl_path).items():
            global_agg[name].append((total, count))
    print_aggregate(global_agg, f"timing aggregate ({len(logs)} logs, {len(jsonls)} jsonl)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit-queue", help="emit per-shard task queue")
    emit.add_argument("--tasks", required=True, help="comma-separated task list")
    emit.add_argument("--total-episodes", type=int, required=True)
    emit.add_argument("--total-workers", type=int, required=True)
    emit.add_argument("--base-seed", type=int, default=100000)
    emit.add_argument("--cache-dir", default="")
    emit.add_argument("--require-cache", action="store_true")
    emit.add_argument("--seed-multiplier", type=int, default=1)
    emit.add_argument("--allow-uncached-candidates", action="store_true")
    emit.add_argument("--report", default="")
    emit.set_defaults(func=cmd_emit_queue)

    merge = subparsers.add_parser("merge-results", help="merge RoboTwin shard results")
    merge.add_argument("log_root", type=Path)
    merge.add_argument("--task", type=str, default=None)
    merge.add_argument("--cleanup", action="store_true")
    merge.add_argument("--summary", type=Path, default=None)
    merge.add_argument("--expect-tasks", type=int, default=None)
    merge.add_argument("--expect-total", type=int, default=None)
    merge.set_defaults(func=cmd_merge_results)

    timing = subparsers.add_parser("analyze-timing", help="aggregate timing logs/jsonl")
    timing.add_argument("path", type=Path, nargs="?", help="log file or directory")
    timing.add_argument("--jsonl", type=Path, help="per-step JSONL from EVAL_TIMING_JSONL")
    timing.set_defaults(func=cmd_analyze_timing)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
