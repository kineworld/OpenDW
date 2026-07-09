#!/usr/bin/env python3
"""DW05 RobotWin online inference and interactive joint-condition demo.

This entrypoint keeps the original interactive web demo experience: draggable
ALOHA joint visualization, joint/EEF sliders, recording, RoboTwin sampling, and
WorldArena sampling.  The runtime model path is provided by the migrated DW05
policy stack.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import base64
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
import websockets
from PIL import Image, ImageDraw
import uvicorn

from dexbotic.data.dataset.dw05.data_source import ROBOTWIN_META
from dexbotic.policy.dw05_policy import (
    DW05RobotWinPolicy,
    DW05RobotWinPolicyConfig,
    ROBOTWIN_PROMPT_FORMAT,
    compose_robotwin_image,
    pil_to_model_tensor,
)


ONLINE_DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROBOTWIN_PROMPT_FORMAT
# Set these via environment variables or CLI flags before running.
V5_STAGED_CKPT_ROOT = Path(os.environ.get("DW05_MODEL_BASE_PATH", "./checkpoints"))
V5_DEFAULT_STEP = "020000"
V5_DEFAULT_CKPT = None   # provide via --ckpt or DW05_CKPT_PATH env var
V5_STATS_PATH = os.environ.get("DW05_NORM_STATS_PATH", "")
V5_SIM_TASK = "dw05_action_mot"
DEFAULT_DATA_ROOT = Path(os.environ.get("DW05_DEMO_DATA_ROOT", "./data/robotwin_demo"))
DEFAULT_OUTPUT_ROOT = Path("/tmp/dw05-demo-output")
DEFAULT_LEFT_MP4 = os.environ.get("DW05_LEFT_MP4", "")
DEFAULT_RIGHT_MP4 = os.environ.get("DW05_RIGHT_MP4", "")
NUM_FRAMES = 9
ACTION_STEPS_PER_ROLLOUT = (NUM_FRAMES - 1) * 4
LOCAL_MODEL_BASE_CANDIDATES = (
    Path(os.environ.get("DW05_MODEL_BASE_PATH", "./checkpoints")),
)
RELATIVE_DIM_MASK = np.array(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)
JOINT_NAMES = [
    "L J1", "L J2", "L J3", "L J4", "L J5", "L J6", "L Grip",
    "R J1", "R J2", "R J3", "R J4", "R J5", "R J6", "R Grip",
]
EEF_NAMES = [
    "L X", "L Y", "L Z", "L Roll", "L Pitch", "L Yaw", "L Grip",
    "R X", "R Y", "R Z", "R Roll", "R Pitch", "R Yaw", "R Grip",
]
ASSET_ROOT = ONLINE_DEMO_ROOT / "assets"
ALOHA_URDF_PATH = ASSET_ROOT / "aloha_tracer2_d435_dark.urdf"
ALOHA_PACKAGE_ROOTS: dict[str, Path] = {}
ARX5_ALOHA_ROOT = ASSET_ROOT
ARX5_URDF_PATH = ASSET_ROOT / "arx5_description_isaac.urdf"
_ALOHA_KINEMATICS_CACHE: dict[str, Any] | None = None
_MODEL_LOAD_CFG_LOCK = threading.Lock()


class ModelWorker:
    def __init__(self, worker_id: int, device: str, model, processor):
        self.worker_id = int(worker_id)
        self.device = str(device)
        self.model = model
        self.processor = processor
        self.busy = False
        self.jobs = 0
        self.last_acquired_at = 0.0

    @property
    def label(self) -> str:
        return f"worker{self.worker_id}:{self.device}"


class ModelWorkerPool:
    def __init__(self, workers: list[ModelWorker]):
        if not workers:
            raise ValueError("ModelWorkerPool requires at least one worker.")
        self.workers = workers
        self._queue: asyncio.Queue[ModelWorker] = asyncio.Queue()
        for worker in workers:
            self._queue.put_nowait(worker)

    @asynccontextmanager
    async def lease(self):
        worker = await self._queue.get()
        worker.busy = True
        worker.jobs += 1
        worker.last_acquired_at = time.time()
        try:
            yield worker
        finally:
            worker.busy = False
            self._queue.put_nowait(worker)

    def status(self) -> dict[str, Any]:
        return {
            "total": len(self.workers),
            "idle": sum(1 for w in self.workers if not w.busy),
            "workers": [
                {"id": w.worker_id, "device": w.device, "busy": w.busy, "jobs": w.jobs}
                for w in self.workers
            ],
        }


def _cuda_device_count() -> int:
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def _web_worker_devices(args: argparse.Namespace) -> list[str]:
    count = max(1, int(getattr(args, "web_gpu_count", 1)))
    device = str(args.device)
    if count == 1 or not device.startswith("cuda"):
        return [device]
    available = _cuda_device_count()
    if available <= 0:
        raise RuntimeError("--web_gpu_count > 1 requires visible CUDA devices.")
    start = 0
    if device.startswith("cuda:"):
        try:
            start = int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {device}") from exc
    if start + count > available:
        raise RuntimeError(f"Requested --web_gpu_count={count} from {device}, but only {available} CUDA devices are visible.")
    return [f"cuda:{i}" for i in range(start, start + count)]


def _activate_worker(worker: ModelWorker) -> None:
    if worker.device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.set_device(int(worker.device.split(":", 1)[1]))


def _load_model_and_processor_for_web_worker(
    ckpt_path: str,
    stats_path: str,
    device: str,
    cfg_name: str,
    sim_task: str | None,
    normalization_mode: str = "zscore_14d",
    image_layout: str = "robotwin_resize",
    policy_action_condition_mode: str = "absolute",
):
    ckpt = Path(ckpt_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    policy = DW05RobotWinPolicy(
        DW05RobotWinPolicyConfig(
            checkpoint_path=str(ckpt),
            norm_stats_path=str(Path(stats_path).expanduser()),
            model_base_path=os.environ.get("DW05_MODEL_BASE_PATH") or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"),
            device=device,
            mixed_precision="bf16",
            action_horizon=ACTION_STEPS_PER_ROLLOUT,
            replan_steps=ACTION_STEPS_PER_ROLLOUT,
            num_inference_steps=10,
            num_video_frames=NUM_FRAMES,
            seed=42,
            rand_device="cpu",
            delta_first_frame=False,
            image_layout=image_layout,
            normalization_mode=normalization_mode,
            action_condition_mode=policy_action_condition_mode,
            load_text_encoder=True,
        )
    )
    return policy.model, policy, policy


def _load_one_web_worker(args: argparse.Namespace, ckpt: str, worker_id: int, device: str) -> ModelWorker:
    print(f"loading_worker={worker_id} device={device}", flush=True)
    model, processor, _ = _load_model_and_processor_for_web_worker(
        ckpt,
        args.stats,
        device,
        args.cfg_name,
        sim_task=args.sim_task,
        normalization_mode=args.normalization_mode,
        image_layout=args.image_layout,
        policy_action_condition_mode=args.policy_action_condition_mode,
    )
    print(f"loaded_worker={worker_id} device={device}", flush=True)
    return ModelWorker(worker_id, device, model, processor)


def _load_web_worker_pool(args: argparse.Namespace, ckpt: str) -> ModelWorkerPool:
    devices = _web_worker_devices(args)
    if len(devices) == 1:
        return ModelWorkerPool([_load_one_web_worker(args, ckpt, 0, devices[0])])
    workers: list[ModelWorker] = []
    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="v5webload") as executor:
        futures = [executor.submit(_load_one_web_worker, args, ckpt, worker_id, device) for worker_id, device in enumerate(devices)]
        for future in as_completed(futures):
            workers.append(future.result())
    workers.sort(key=lambda w: w.worker_id)
    return ModelWorkerPool(workers)


def _session_id() -> str:
    return uuid.uuid4().hex


def _cache_session_output(session_id: str, kind: str, **values: Any) -> None:
    if not session_id:
        return
    bucket = _SESSION_OUTPUTS.setdefault(session_id, {})
    current = dict(bucket.get(kind, {}))
    current.update(values)
    current["updated_at"] = time.time()
    bucket[kind] = current
    if len(_SESSION_OUTPUTS) > 96:
        ordered = sorted(
            _SESSION_OUTPUTS.items(),
            key=lambda item: max((v.get("updated_at", 0.0) for v in item[1].values() if isinstance(v, dict)), default=0.0),
        )
        for old_sid, _ in ordered[: max(1, len(_SESSION_OUTPUTS) - 96)]:
            _SESSION_OUTPUTS.pop(old_sid, None)


def _session_output(session_id: str, kind: str) -> dict[str, Any]:
    return dict(_SESSION_OUTPUTS.get(session_id, {}).get(kind, {})) if session_id else {}


def _xml_vec(value: str | None, default: tuple[float, float, float]) -> list[float]:
    if not value:
        return [float(v) for v in default]
    vals = [float(x) for x in value.split()]
    if len(vals) != 3:
        return [float(v) for v in default]
    return vals


def _xml_origin(elem) -> dict[str, list[float]]:
    origin = elem.find("origin")
    if origin is None:
        return {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
    return {
        "xyz": _xml_vec(origin.get("xyz"), (0.0, 0.0, 0.0)),
        "rpy": _xml_vec(origin.get("rpy"), (0.0, 0.0, 0.0)),
    }


def _aloha_kinematics_payload() -> dict[str, Any]:
    import struct
    import xml.etree.ElementTree as ET

    global _ALOHA_KINEMATICS_CACHE
    if _ALOHA_KINEMATICS_CACHE is not None:
        return _ALOHA_KINEMATICS_CACHE

    root = ET.parse(ALOHA_URDF_PATH).getroot()
    joints_by_name = {j.get("name"): j for j in root.findall("joint") if j.get("name")}
    links_by_name = {l.get("name"): l for l in root.findall("link") if l.get("name")}

    arm_root = ET.parse(ARX5_URDF_PATH).getroot()
    arm_joints_by_name = {j.get("name"): j for j in arm_root.findall("joint") if j.get("name")}
    arm_links_by_name = {l.get("name"): l for l in arm_root.findall("link") if l.get("name")}

    def children_by_parent_for(joints: dict[str, Any]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for jname, joint in joints.items():
            parent = joint.find("parent")
            if parent is not None and parent.get("link"):
                out.setdefault(parent.get("link"), []).append(jname)
        return out

    children_by_parent = children_by_parent_for(joints_by_name)
    arm_children_by_parent = children_by_parent_for(arm_joints_by_name)
    mesh_cache: dict[str, dict[str, Any] | None] = {}

    def child_name(elem) -> str:
        child = elem.find("child")
        return child.get("link", "") if child is not None else ""

    def parent_name(elem) -> str:
        parent = elem.find("parent")
        return parent.get("link", "") if parent is not None else ""

    def resolve_mesh_path(mesh_name: str) -> Path | None:
        raw = str(mesh_name or "")
        if raw.startswith("package://"):
            rest = raw.removeprefix("package://")
            package, _, rel = rest.partition("/")
            root_path = ALOHA_PACKAGE_ROOTS.get(package)
            return (root_path / rel) if root_path is not None else None
        if raw.startswith("file://"):
            return Path(raw.removeprefix("file://"))
        path = Path(raw)
        if path.is_absolute():
            return path
        candidates = [
            ALOHA_URDF_PATH.parent / path,
            ARX5_URDF_PATH.parent / path,
            ARX5_ALOHA_ROOT / path,
            ASSET_ROOT / path,
            ASSET_ROOT / "mesh" / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def mesh_payload(mesh_name: str, target_triangles: int = 260, large_faces_only: bool = False) -> dict[str, Any] | None:
        mesh_path = resolve_mesh_path(mesh_name)
        key_base = str(mesh_path) if mesh_path is not None else str(mesh_name)
        target = max(1, int(target_triangles))
        key = f"{key_base}|n={target}|large={int(large_faces_only)}"
        if key in mesh_cache:
            return mesh_cache[key]
        if mesh_path is None or not mesh_path.is_file():
            mesh_cache[key] = None
            return None

        def remove_static_display_artifact(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
            """Hide the overhead display baked into the ALOHA body mesh for the online viewer."""

            if mesh_path.name != "mobile_aloha__aloha_new_description__meshes__body_v2.dae":
                return faces
            tri_vertices = vertices[faces]
            tri_min = tri_vertices.min(axis=1)
            tri_max = tri_vertices.max(axis=1)
            tri_center = tri_vertices.mean(axis=1)
            display_region = (
                (tri_center[:, 2] > 1.16)
                & (tri_min[:, 0] > -0.20)
                & (tri_max[:, 0] < -0.06)
                & (tri_min[:, 1] > -0.16)
                & (tri_max[:, 1] < 0.16)
            )
            kept = faces[~display_region]
            return kept if len(kept) else faces

        try:
            import trimesh
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if hasattr(mesh, "dump") and not hasattr(mesh, "vertices"):
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
                mesh_cache[key] = None
                return None
            faces = remove_static_display_artifact(vertices, faces)
            tri_count = int(len(faces))
            sample_count = min(tri_count, target)
            if large_faces_only:
                tri_vertices = vertices[faces]
                areas = 0.5 * np.linalg.norm(
                    np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
                    axis=1,
                )
                valid = np.flatnonzero(np.isfinite(areas) & (areas > 0))
                if len(valid):
                    order = valid[np.argsort(-areas[valid], kind="mergesort")]
                    sample_count = min(len(order), target)
                    sample_idx = np.sort(order[:sample_count])
                else:
                    sample_idx = np.linspace(0, tri_count - 1, sample_count).astype(int)
            else:
                sample_idx = np.linspace(0, tri_count - 1, sample_count).astype(int)
            coords = vertices[faces[sample_idx]].reshape(sample_count, 9)
            used_vertices = vertices[np.unique(faces.reshape(-1))]
            mins = used_vertices.min(axis=0)
            maxs = used_vertices.max(axis=0)
            payload = {
                "min": [round(float(v), 5) for v in mins],
                "max": [round(float(v), 5) for v in maxs],
                "mesh": str(mesh_path.relative_to(PROJECT_ROOT)) if str(mesh_path).startswith(str(PROJECT_ROOT)) else str(mesh_path),
                "triangles": [[round(float(v), 5) for v in row] for row in coords],
                "tri_count": tri_count,
                "sample_count": sample_count,
                "large_faces_only": large_faces_only,
            }
            mesh_cache[key] = payload
            return payload
        except Exception:
            mesh_cache[key] = None
            return None

    def link_payload(name: str, source_links: dict[str, Any] | None = None, target_triangles: int = 260, large_faces_only: bool = False) -> dict[str, Any]:
        elem = (source_links or links_by_name).get(name)
        if elem is None:
            return {"name": name}
        visual = elem.find("visual")
        collision = elem.find("collision")
        source = visual if visual is not None else collision
        geometry = source.find("geometry") if source is not None else None
        mesh = geometry.find("mesh") if geometry is not None else None
        box = geometry.find("box") if geometry is not None else None
        out: dict[str, Any] = {"name": name, "origin": _xml_origin(source) if source is not None else _xml_origin(elem)}
        if mesh is not None and mesh.get("filename"):
            out["mesh"] = mesh.get("filename")
            mesh_data = mesh_payload(mesh.get("filename", ""), target_triangles=target_triangles, large_faces_only=large_faces_only)
            if mesh_data:
                out["bbox"] = {"min": mesh_data["min"], "max": mesh_data["max"], "mesh": mesh_data["mesh"]}
                out["triangles"] = mesh_data["triangles"]
                out["tri_count"] = mesh_data["tri_count"]
                out["sample_count"] = mesh_data["sample_count"]
                out["large_faces_only"] = mesh_data.get("large_faces_only", False)
        elif box is not None and box.get("size"):
            size = [float(v) for v in box.get("size", "0 0 0").split()]
            if len(size) == 3:
                out["bbox"] = {
                    "min": [-size[0] / 2, -size[1] / 2, -size[2] / 2],
                    "max": [size[0] / 2, size[1] / 2, size[2] / 2],
                    "mesh": "urdf_box",
                }
        return out

    def joint_payload(name: str, source_joints: dict[str, Any] | None = None) -> dict[str, Any]:
        elem = (source_joints or joints_by_name)[name]
        axis = elem.find("axis")
        limit = elem.find("limit")
        return {
            "name": name,
            "type": elem.get("type", "fixed"),
            "parent": parent_name(elem),
            "child": child_name(elem),
            "origin": _xml_origin(elem),
            "axis": _xml_vec(axis.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
            "limit": {
                "lower": float(limit.get("lower", "0")) if limit is not None else 0.0,
                "upper": float(limit.get("upper", "0")) if limit is not None else 0.0,
            },
        }

    def joint_chain_to_link(link_name: str) -> list[str]:
        child_to_joint: dict[str, str] = {}
        for jname, joint in joints_by_name.items():
            child = joint.find("child")
            if child is not None and child.get("link"):
                child_to_joint[child.get("link")] = jname
        chain: list[str] = []
        cur = link_name
        seen: set[str] = set()
        while cur in child_to_joint and cur not in seen:
            seen.add(cur)
            jname = child_to_joint[cur]
            chain.append(jname)
            parent = joints_by_name[jname].find("parent")
            cur = parent.get("link", "") if parent is not None else ""
        return list(reversed(chain))

    def static_color(link_name: str) -> str:
        name = link_name.lower()
        if "camera_link" in name:
            return "#d8dce2"
        if "camera_stand" in name:
            return "#b8bec6"
        if "body" in name:
            return "#c9c9c6"
        if name in {"right_link", "left_link"} or "front_link" in name or "rear_link" in name:
            return "#777f87"
        if "base" in name:
            return "#9aa2aa"
        return "#a8adb2"

    def static_triangle_budget(link_name: str) -> int:
        name = link_name.lower()
        if name in {"base_link", "body_link", "camera_link"}:
            return 1800
        if name in {"right_link", "left_link"} or "front_link" in name or "rear_link" in name:
            return 900
        if "camera_stand" in name:
            return 1200
        return 700

    def static_parts() -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        arm_prefixes = ("fl_", "fr_", "bl_", "br_")
        for link_name in links_by_name:
            if link_name == "footprint" or link_name.startswith(arm_prefixes):
                continue
            payload = link_payload(link_name, target_triangles=static_triangle_budget(link_name), large_faces_only=True)
            if not (payload.get("triangles") or payload.get("bbox")):
                continue
            payload["static_mesh"] = True
            chain_names = joint_chain_to_link(link_name)
            parts.append({
                "name": link_name,
                "color": static_color(link_name),
                "joints": [joint_payload(name) for name in chain_names],
                "link": payload,
            })
        return parts

    def arm_descendants(
        root_link: str,
        source_children: dict[str, list[str]],
        source_links: dict[str, Any],
        source_joints: dict[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
        children: dict[str, list[dict[str, Any]]] = {}
        links: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()

        def visit(link_name: str) -> None:
            if link_name in seen:
                return
            seen.add(link_name)
            links[link_name] = link_payload(link_name, source_links)
            for jname in source_children.get(link_name, []):
                joint = joint_payload(jname, source_joints)
                children.setdefault(link_name, []).append(joint)
                child = joint.get("child")
                if child:
                    visit(child)

        visit(root_link)
        return children, links

    def arm(prefix: str, label: str, color: str) -> dict[str, Any]:
        base_joint = joint_payload(f"{prefix}_base_joint", arm_joints_by_name)
        joints = [joint_payload(f"{prefix}_joint{i}", arm_joints_by_name) for i in range(1, 9)]
        for idx, joint in enumerate(joints[:6]):
            joint["value_index"] = idx
        for joint in joints[6:8]:
            joint["value_index"] = 6
        controlled_by_name = {joint["name"]: joint for joint in joints}
        tree_children, tree_links = arm_descendants(
            base_joint["child"],
            arm_children_by_parent,
            arm_links_by_name,
            arm_joints_by_name,
        )
        for branch in tree_children.values():
            for idx, joint in enumerate(branch):
                controlled = controlled_by_name.get(joint["name"])
                if controlled:
                    branch[idx] = {**joint, **{k: controlled[k] for k in ("value_index", "gripper_sign") if k in controlled}}
        return {
            "label": label,
            "prefix": prefix,
            "color": color,
            "base_joint": base_joint,
            "joints": joints,
            "children": tree_children,
            "links": tree_links,
        }

    left_arm = arm("fl", "left", "#ff3333")
    right_arm = arm("fr", "right", "#33dd55")

    def joint_limits_for_ui() -> list[list[float]]:
        limits: list[list[float]] = []
        for arm_payload in (left_arm, right_arm):
            for joint in arm_payload["joints"][:6]:
                limit = joint.get("limit") or {}
                lo, hi = float(limit.get("lower", -3.14)), float(limit.get("upper", 3.14))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    lo, hi = -3.14, 3.14
                limits.append([round(max(lo, -3.14), 5), round(min(hi, 3.14), 5)])
            limits.append([0.0, 1.0])
        return limits

    _ALOHA_KINEMATICS_CACHE = {
        "urdf": str(ALOHA_URDF_PATH),
        "arm_urdf": str(ARX5_URDF_PATH),
        "static_parts": static_parts(),
        "arms": {
            "left": left_arm,
            "right": right_arm,
        },
        "model_action_order": JOINT_NAMES,
        "joint_limits": joint_limits_for_ui(),
        "gripper_upper": 0.04765,
    }
    return _ALOHA_KINEMATICS_CACHE


def _default_ckpt_for_step(step: str | int | None) -> str:
    step_s = str(step or V5_DEFAULT_STEP).strip()
    if step_s.startswith("step_"):
        step_s = step_s.removeprefix("step_").removesuffix(".pt")
    if V5_DEFAULT_CKPT.is_file():
        return str(V5_DEFAULT_CKPT)
    return str(V5_STAGED_CKPT_ROOT / f"dw05-v5-action-mot-step{int(step_s):06d}.pt")


def parse_episodes(raw: str | None) -> list[int]:
    if not raw:
        return list(range(1, 251))
    episodes: list[int] = []
    for part in str(raw).split(","):
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


def read_first_frame(mp4_path: str) -> np.ndarray:
    import imageio

    reader = imageio.get_reader(str(mp4_path))
    try:
        return np.asarray(reader.get_data(0))
    finally:
        reader.close()


def pil_to_tensor(img: Image.Image, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return pil_to_model_tensor(img, device, dtype)


def build_image_tensor(head: np.ndarray, left: np.ndarray, right: np.ndarray, device, dtype) -> torch.Tensor:
    image = compose_robotwin_image([head, left, right], layout="robotwin_resize", image_size_hw=(384, 320))
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return (tensor * (2.0 / 255.0) - 1.0).to(device=device, dtype=dtype)


def _split_worldarena_state_action(qpos_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = _as_2d("qpos", qpos_np)
    if len(qpos) <= 1:
        return qpos[:1].copy(), qpos.copy()
    return qpos[:-1].copy(), qpos[1:].copy()


def run_episode(
    model,
    processor,
    action_np: np.ndarray,
    prompt: str,
    init_image_tensor: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    max_rollouts: int | None,
    state_np: np.ndarray | None = None,
) -> tuple[list[Image.Image], list[Image.Image]]:
    policy = processor
    if not isinstance(policy, DW05RobotWinPolicy):
        raise TypeError("DW05 V5 demo worker was not initialized with a DW05 policy.")
    old_steps, old_seed = policy.config.num_inference_steps, policy.config.seed
    policy.config.num_inference_steps = int(num_inference_steps)
    policy.config.seed = int(seed)
    try:
        frames = policy.rollout_video_with_actions(
            prompt=prompt,
            init_image_tensor=init_image_tensor,
            action_abs=_as_2d("action_abs", action_np),
            state_abs=_align_state_to_action(state_np, action_np) if state_np is not None else None,
            max_rollouts=max_rollouts,
        )
    finally:
        policy.config.num_inference_steps = old_steps
        policy.config.seed = old_seed
    cond_frames = _joint_condition_frames(action_np, state_np)
    return frames, cond_frames


def read_gt_frames(gt_video_dir: str, ep: int, n: int | None = None) -> list[Image.Image]:
    if not gt_video_dir:
        return []
    path = Path(gt_video_dir).expanduser() / f"episode{int(ep)}.mp4"
    if not path.exists():
        return []
    return _read_worldarena_gt_video_path(path, n=n)


def save_video(
    frames: list[Image.Image],
    output_dir: Path,
    name: str,
    gt_frames: list[Image.Image] | None = None,
    cond_frames: list[Image.Image] | None = None,
    save_gt: bool = False,
    save_generated: bool = True,
    save_condition: bool = False,
    crop_main_view: bool = False,
) -> Path:
    import imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{name}.mp4"
    series: list[list[Image.Image]] = []
    if save_gt and gt_frames:
        series.append(gt_frames)
    if save_generated:
        series.append(frames)
    if save_condition and cond_frames:
        series.append(cond_frames)
    if not series:
        series = [frames]
    n = max(len(s) for s in series)
    merged = [_hconcat_pils([s[min(i, len(s) - 1)] for s in series]) for i in range(n)]
    arrays = [np.asarray(frame.convert("RGB")) for frame in merged]
    imageio.mimsave(str(video_path), arrays, fps=8)
    return video_path


def _dw05_bundle_ready(path: Path) -> bool:
    return (
        (
            (path / "vae" / "model.pth").is_file()
            or (path / "vae" / "model.safetensors").is_file()
            or (path / "vae" / "Wan2.2_VAE.pth").is_file()
            or (path / "vae" / "Wan2.2_VAE.safetensors").is_file()
        )
        and (
            (path / "text_encoder" / "model.pth").is_file()
            or (path / "text_encoder" / "model.safetensors").is_file()
            or (path / "text_encoder" / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
            or (path / "text_encoder" / "models_t5_umt5-xxl-enc-bf16.safetensors").is_file()
        )
        and (path / "tokenizer").is_dir()
        and any(
            (path / "tokenizer" / name).is_file()
            for name in ("tokenizer_config.json", "tokenizer.json", "spiece.model")
        )
    )


def _legacy_model_cache_ready(path: Path) -> bool:
    return (
        (path / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors").is_file()
        and (path / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors").is_file()
        and (path / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl").is_dir()
    )


def _local_model_base_ready(path: Path) -> bool:
    return _dw05_bundle_ready(path) or _legacy_model_cache_ready(path)


def _prepare_local_model_base(model_base_path: str | os.PathLike | None = None) -> str:
    """Pin DiffSynth/Wan loading to local files and disable implicit downloads."""
    candidates: list[Path] = []
    if model_base_path:
        candidates.append(Path(model_base_path).expanduser())
    env_base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    if env_base:
        candidates.append(Path(env_base).expanduser())
    candidates.extend(LOCAL_MODEL_BASE_CANDIDATES)

    for candidate in candidates:
        if _local_model_base_ready(candidate):
            os.environ["DW05_MODEL_BASE_PATH"] = str(candidate)
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(candidate)
            os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
            return str(candidate)

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Missing local DW05 runtime bundle. Refusing to download. "
        "Pass --model_base_path or set DW05_MODEL_BASE_PATH to a directory containing:\n"
        "  vae/Wan2.2_VAE.pth or vae/model.pth\n"
        "  text_encoder/models_t5_umt5-xxl-enc-bf16.pth or text_encoder/model.pth\n"
        "  tokenizer/ with tokenizer_config.json, tokenizer.json, or spiece.model\n"
        f"Checked:\n{checked}"
    )


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_numpy(value: Any, key: str | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    elif isinstance(value, dict):
        keys = [key] if key else []
        keys.extend(["action", "actions", "qpos", "joint_action", "state", "proprio", "arr_0"])
        for candidate in keys:
            if candidate and candidate in value:
                return _to_numpy(value[candidate])
        raise KeyError(f"No array key found in dict. Available keys: {sorted(value)}")
    elif isinstance(value, list):
        value = np.asarray(value, dtype=np.float32)
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def load_array(path: str | os.PathLike, key: str | None = None) -> np.ndarray:
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return _to_numpy(np.load(p, allow_pickle=False), key=key)
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            if key:
                return _to_numpy(data[key])
            if len(data.files) == 1:
                return _to_numpy(data[data.files[0]])
            for candidate in ("action", "actions", "qpos", "state", "proprio", "arr_0"):
                if candidate in data.files:
                    return _to_numpy(data[candidate])
            raise KeyError(f"No default array key found in {p}; keys={data.files}")
    if suffix in {".pt", ".pth"}:
        return _to_numpy(torch.load(p, map_location="cpu"), key=key)
    if suffix == ".json":
        return _to_numpy(_load_json(p), key=key)
    raise ValueError(f"Unsupported action/state file type: {p}")


def _as_2d(name: str, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [T,D] or [D], got shape {arr.shape}")
    if arr.shape[-1] != 14:
        raise ValueError(f"{name} must have 14 joint/gripper dims for the RobotWin online demo, got {arr.shape[-1]}")
    return arr


def _relative_action_from_abs(action_abs: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    action = _as_2d("action_abs", action_abs).copy()
    base = np.asarray(base_state, dtype=np.float32).reshape(-1)
    if base.shape[0] != action.shape[-1]:
        raise ValueError(f"base_state dim must match action dim {action.shape[-1]}, got {base.shape[0]}")
    action[:, RELATIVE_DIM_MASK] -= base[RELATIVE_DIM_MASK]
    return action.astype(np.float32)


def _read_image(path: str | os.PathLike) -> np.ndarray:
    return np.asarray(Image.open(Path(path).expanduser()).convert("RGB"))


def _load_prompt(data_root: Path, ep: int, prompt: str, instr_root: str | None) -> str:
    if prompt:
        return prompt
    roots = [Path(instr_root).expanduser()] if instr_root else [
        data_root / "instructions/fixed_scene_task",
        data_root / "instructions_1/fixed_scene_task",
        data_root / "instructions_2/fixed_scene_task",
    ]
    for root in roots:
        path = root / f"episode{int(ep)}.json"
        if path.exists():
            task = _load_json(path).get("instruction", "")
            if task:
                return DEFAULT_PROMPT.format(task=task)
    return "A robot arm completes a task."


def _load_init_image(args: argparse.Namespace, model, processor, ep: int) -> torch.Tensor:
    data_root = Path(args.data_root).expanduser()
    if args.head_image:
        head = _read_image(args.head_image)
    else:
        head = _read_image(data_root / "first_frame/fixed_scene_task" / f"episode{int(ep)}.png")

    left = _read_image(args.left_image) if args.left_image else read_first_frame(args.left_mp4)
    right = _read_image(args.right_image) if args.right_image else read_first_frame(args.right_mp4)
    return build_image_tensor(head, left, right, model.device, model.torch_dtype)


def _load_worldarena_condition(args: argparse.Namespace, ep: int) -> tuple[np.ndarray, np.ndarray]:
    data_root = Path(args.data_root).expanduser()
    hdf5_root = Path(args.hdf5_root).expanduser() if args.hdf5_root else data_root / "data/fixed_scene_task"
    with h5py.File(hdf5_root / f"episode{int(ep)}.hdf5", "r") as f:
        qpos_np = f[args.hdf5_action_key][:].astype(np.float32)
    state_np, action_abs_np = _split_worldarena_state_action(qpos_np)
    return _as_2d("state", state_np), _as_2d("action_abs", action_abs_np)


def _align_state_to_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    state = _as_2d("state_condition", state)
    action = _as_2d("action_condition", action)
    if len(state) == len(action):
        return state
    if len(state) == 1:
        return np.repeat(state, len(action), axis=0).astype(np.float32)
    if len(state) < len(action):
        pad = np.repeat(state[-1:], len(action) - len(state), axis=0)
        return np.concatenate([state, pad], axis=0).astype(np.float32)
    return state[:len(action)].astype(np.float32)


def _load_external_condition(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray, bool]:
    action = _as_2d("action_condition", load_array(args.action_condition, key=args.action_key))
    state = None
    if args.state_condition:
        state = _as_2d("state_condition", load_array(args.state_condition, key=args.state_key))

    mode = args.action_condition_mode
    if mode == "auto":
        mode = "qpos" if state is None else "raw_action"

    if mode == "qpos":
        state_np, action_abs_np = _split_worldarena_state_action(action)
        return _as_2d("state", state_np), _as_2d("action_abs", action_abs_np), False
    if mode == "raw_action":
        if state is None:
            state = action[:1].copy()
            print("No --state_condition was provided; reusing the first absolute action row as proprio/state.")
        return _align_state_to_action(state, action), action, False
    if mode == "normalized":
        return state, action, True
    raise ValueError(f"Unsupported --action_condition_mode={args.action_condition_mode!r}")


def _select_proprio(proprio_np: np.ndarray | None, start: int, device, dtype) -> torch.Tensor | None:
    if proprio_np is None:
        return None
    prop = np.asarray(proprio_np, dtype=np.float32)
    if prop.ndim == 2:
        prop = prop[min(int(start), len(prop) - 1)]
    elif prop.ndim != 1:
        raise ValueError(f"Normalized proprio/state must be [D] or [T,D], got {prop.shape}")
    return torch.as_tensor(prop, dtype=torch.float32, device=device).to(dtype=dtype)


def run_episode_with_normalized_action(
    model,
    action_np: np.ndarray,
    prompt: str,
    init_image_tensor: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    max_rollouts: int | None,
    proprio_np: np.ndarray | None = None,
) -> list[Image.Image]:
    action_np = _as_2d("normalized relative action", action_np)
    all_frames: list[Image.Image] = []
    cur_image = init_image_tensor
    action_offset = 0
    rollout = 0

    while action_offset < len(action_np):
        if max_rollouts is not None and rollout >= max_rollouts:
            break
        chunk = action_np[action_offset : action_offset + ACTION_STEPS_PER_ROLLOUT]
        if len(chunk) == 0:
            break
        if len(chunk) < ACTION_STEPS_PER_ROLLOUT:
            pad = np.tile(chunk[-1:], (ACTION_STEPS_PER_ROLLOUT - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)

        action_tensor = torch.as_tensor(chunk, dtype=torch.float32, device=model.device).to(dtype=model.torch_dtype)
        proprio_tensor = _select_proprio(proprio_np, action_offset, model.device, model.torch_dtype)
        with torch.no_grad():
            out = model.infer_joint(
                prompt=prompt,
                input_image=cur_image,
                num_video_frames=NUM_FRAMES,
                action_horizon=ACTION_STEPS_PER_ROLLOUT,
                action=action_tensor,
                proprio=proprio_tensor,
                num_inference_steps=num_inference_steps,
                seed=seed,
                rand_device="cpu",
                test_action_with_infer_action=False,
            )

        frames = out["video"]
        all_frames.extend(frames if rollout == 0 else frames[1:])
        cur_image = pil_to_tensor(frames[-1], model.device, model.torch_dtype)
        action_offset += ACTION_STEPS_PER_ROLLOUT
        rollout += 1
    return all_frames


def _target_video_path(output_root: Path, output_name: str, ep: int | None) -> Path:
    if ep is None:
        return output_root / f"{output_name}.mp4"
    return output_root / output_name / f"episode{int(ep)}.mp4"


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tensor_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor[0].detach().cpu().float()
    arr = ((t + 1.0) / 2.0 * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _joint_payload(base_state: np.ndarray, target_state: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target_state, dtype=np.float32)
    base = np.asarray(base_state, dtype=np.float32)
    return {
        "state": target.round(5).tolist(),
        "base": base.round(5).tolist(),
        "condition": target.round(5).tolist(),
        "condition_mode": "absolute",
        "joint_names": JOINT_NAMES,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RobotWin Online Demo</title>
<style>
  :root{--bg:#0f1115;--bar:#13161c;--panel:#171b22;--line:#2a313c;--line2:#343c48;--text:#e6e9ee;--muted:#9aa4b2;--soft:#c7ced8;--accent:#32a667;--accent2:#62d38f;--blue:#78a7ff;--danger:#f06458}
  html,body{height:100%;width:100%;overflow:hidden}
  body{margin:0;box-sizing:border-box;background:var(--bg);color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .appHeader{height:58px;padding:0 18px;box-sizing:border-box;background:var(--bar);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:18px}
  h1{margin:0;font-size:18px;font-weight:700;letter-spacing:0;color:#fff;white-space:nowrap}.appShell{height:calc(100vh - 58px);width:100%;padding:14px;box-sizing:border-box;overflow:hidden}.featureTabs{display:flex;gap:6px;align-items:center;justify-content:flex-end}.featureTab{height:34px;padding:0 16px;background:#1b2028;color:var(--soft);border:1px solid var(--line2);border-radius:6px;font-family:inherit;cursor:pointer}.featureTab.active{background:var(--accent);color:#fff;border-color:var(--accent)}.featurePanel{display:none;height:100%}.featurePanel.active{display:block}.featureLayout{height:100%;display:grid;grid-template-columns:430px minmax(520px,1fr) 420px;gap:14px;align-items:stretch}.infoCol,.visualCol,.controlCol{min-width:0;min-height:0;display:flex;flex-direction:column}.infoCol,.controlCol{overflow:auto;scrollbar-color:#3a4350 transparent;scrollbar-width:thin}.visualCol{overflow:hidden}.columnTitle{height:22px;margin:0 0 8px;font-size:12px;font-weight:700;color:#dce3ec;text-transform:uppercase;letter-spacing:0}.infoBlock,.controlBlock{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-sizing:border-box;padding:14px;margin-bottom:10px;flex-shrink:0}.controlBlock{padding:12px}.infoTitle{font-size:11px;color:var(--accent2);text-transform:uppercase;margin-bottom:7px}.primaryInfo{font-size:14px;font-weight:700;color:#69b4ff;line-height:1.35}.st{font-size:13px;color:var(--soft);min-height:16px;line-height:1.4}.inf{font-size:12px;color:var(--blue);line-height:1.4;overflow-wrap:anywhere}.metric,.muted,.batchInfo{font-size:12px;color:var(--muted);line-height:1.4;overflow-wrap:anywhere}.metric{font-variant-numeric:tabular-nums}
  .stageCard{flex:1;min-height:0;height:auto;position:relative;overflow:hidden;background:linear-gradient(180deg,#181c23 0%,#11151a 100%);border:1px solid var(--line);border-radius:8px;box-sizing:border-box}.stageTitle{position:absolute;top:12px;left:14px;z-index:4;padding:5px 9px;background:rgba(13,16,21,.78);border:1px solid rgba(105,119,137,.34);border-radius:6px;color:#eaf0f7;font-size:12px}.stageCard .robotCanvas{position:absolute;inset:0;width:100%;height:100%;max-height:none;border:0;border-radius:0;background:#15191f;cursor:grab;touch-action:none}.robotCanvas.dragging{cursor:grabbing}.mediaStack{position:absolute;z-index:5;top:12px;left:12px;display:flex;flex-direction:column;gap:10px;align-items:flex-start;justify-content:flex-start;pointer-events:auto;touch-action:none;cursor:grab;max-height:calc(100% - 24px);overflow:auto;scrollbar-width:thin;scrollbar-color:#3a4350 transparent}.mediaStack.dragging{cursor:grabbing}.floatVideo{width:clamp(230px,18vw,340px);padding:8px;background:rgba(17,22,29,.84);border:1px solid rgba(124,139,158,.36);border-radius:8px;box-shadow:0 18px 40px rgba(0,0,0,.32);backdrop-filter:blur(10px)}.floatVideo canvas{width:100%;height:auto;border-radius:5px;border:1px solid rgba(126,139,154,.38);background:#0d1117}.floatTitle{height:16px;margin-bottom:6px;color:#dce4ef;font-size:11px;cursor:grab}.stageTimeline{flex:0 0 auto;margin-top:10px;padding:11px 13px;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-sizing:border-box}.stageTimeline .infoTitle{margin-bottom:6px}.timelineRange{width:100%;accent-color:var(--accent);margin:2px 0 8px}.timelineMeta{display:flex;align-items:center;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px}canvas{box-sizing:border-box;max-width:100%;display:block}.jointChart{width:100%;height:128px;border:0;border-radius:5px;background:#10151c}.gripMeter{display:grid;grid-template-columns:48px 1fr 48px;gap:10px;align-items:center;color:#d6dde7;font-size:11px;margin:7px 0}.gripTrack{height:9px;background:#252c35;border:1px solid #3b4654;border-radius:999px;overflow:hidden}.gripFill{display:block;height:100%;width:0%;background:linear-gradient(90deg,#65b7ff,#35d07f)}.gripQuick{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px}
  .chartBlock{flex:1 1 420px;min-height:360px;display:flex;flex-direction:column}.chartStack{flex:1;min-height:0;display:grid;grid-template-rows:repeat(3,minmax(0,1fr));gap:10px}.chartStack>div{min-height:0;display:flex;flex-direction:column}.chartLabel{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;color:#cbd4df;font-size:11px;line-height:1;margin-bottom:5px}.chartLabel span:last-child{color:#7f8b99}.chartStack .jointChart{flex:1;width:100%;height:100%;min-height:118px}.helpList{margin:0;padding-left:16px;color:#c9d1dc;font-size:12px;line-height:1.55}.helpList li{margin:3px 0}.helpList strong{color:#eef3f9;font-weight:700}.buttonNote{margin-top:9px;color:#8793a1;font-size:11px;line-height:1.35}
  button{height:32px;padding:0 13px;font-family:inherit;cursor:pointer;background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:6px;margin:0}button:disabled{opacity:.45;cursor:default}.timelineMeta button,.gripQuick button,.tabBtn{background:#202832;border-color:var(--line2);color:#dce5ee}.tabBtn.active{background:var(--accent);color:#fff;border-color:var(--accent)}.dlBtn{display:inline-flex;height:32px;align-items:center;justify-content:center;padding:0 13px;background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:6px;text-decoration:none;font-family:inherit;opacity:.4;pointer-events:none}.demoActions{display:grid;grid-template-columns:1fr;gap:7px;align-items:stretch}.demoActions button,.demoActions .dlBtn{width:100%;box-sizing:border-box}.legendRow{display:flex;align-items:center;gap:8px;font-size:12px;margin:6px 0;color:var(--soft)}.sw{width:12px;height:12px;border-radius:3px;display:inline-block;border:1px solid #6c7480}.swL{background:var(--danger)}.swR{background:#25d978}.swBase{background:#7d858f}.controlTabs{display:flex;gap:7px;align-items:center;margin-bottom:10px}.jointCtl{display:grid;grid-template-columns:1fr;gap:8px 12px}.jointCol{display:grid;grid-template-columns:1fr;gap:6px}.sliderRow{display:grid;grid-template-columns:74px 1fr 58px 24px;gap:8px;align-items:center;font-size:12px;color:#d7dce4}.sliderRow span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sliderRow input{width:100%;accent-color:var(--accent)}.sliderRow output{text-align:right;color:#cbd2dc;font-variant-numeric:tabular-nums}.miniReset{width:24px;height:24px;padding:0;background:#252b34;color:#dce2ea;border:1px solid var(--line2);border-radius:5px;font-size:11px}.waSelector{font-size:12px;color:#dce2ea;margin-bottom:8px}.waSelector select{background:#20262f;color:#eef2f7;border:1px solid var(--line2);padding:6px 8px;border-radius:6px}.batchOpts{display:grid;grid-template-columns:1fr;gap:8px 10px;font-size:12px;color:#cbd2dc;margin-top:8px}.batchOpts label{display:inline-flex;align-items:center;gap:5px}.progWrap{width:100%;height:16px;border:1px solid var(--line2);background:#11151b;border-radius:5px;overflow:hidden;margin-top:8px}.progFill{height:100%;width:0%;background:var(--accent);transition:width .25s}.controlHint{font-size:12px;color:var(--muted);line-height:1.4;margin-top:10px}
  .demoActions{gap:9px}.demoActions .actionBtn,.demoActions .dlBtn{height:38px;border-radius:7px;font-weight:700;letter-spacing:0}.actionBtn,.downloadAction{transition:transform .12s ease,border-color .12s ease,filter .12s ease}.actionBtn:hover:not(:disabled),.dlBtn:hover{transform:translateY(-1px);filter:brightness(1.08)}.primaryAction{background:linear-gradient(180deg,#39bd77,#268b58);border-color:#48d991;color:#fff}.secondaryAction{background:#222a34;border-color:#414c5b;color:#edf2f7}.recordAction{background:linear-gradient(180deg,#d89234,#a96820);border-color:#e4a44d;color:#fff6e5}.dangerAction{background:#3a2427;border-color:#e26b62;color:#ffe0dd}.downloadAction{background:#1d3448;border-color:#62a6e7;color:#eef7ff}.evalAction{background:#292b43;border-color:#7771dc;color:#f0efff}.miniReset:hover,.timelineMeta button:hover:not(:disabled),.gripQuick button:hover:not(:disabled),.tabBtn:hover:not(:disabled){filter:brightness(1.1)}
  @media(max-width:1600px){.featureLayout{grid-template-columns:390px minmax(500px,1fr) 380px}.chartBlock{min-height:330px}.floatVideo{width:clamp(210px,16vw,300px)}}@media(max-width:1280px){.featureLayout{grid-template-columns:340px minmax(420px,1fr) 340px}.chartBlock{min-height:300px}.chartStack .jointChart{min-height:96px}.floatVideo{width:clamp(190px,15vw,260px)}}@media(max-width:1100px){html,body{overflow:auto}.appHeader{height:auto;min-height:58px;padding:12px;align-items:flex-start;flex-direction:column}.appShell{height:auto;min-height:calc(100vh - 58px);overflow:visible}.featureTabs{justify-content:flex-start}.featureLayout{height:auto;grid-template-columns:1fr}.stageCard{height:min(70vh,680px);min-height:500px}.chartBlock{flex:0 0 auto;min-height:360px}.controlCol{max-height:none}.demoActions{grid-template-columns:repeat(2,minmax(0,1fr))}.floatVideo{width:clamp(220px,36vw,340px)}}@media(max-width:720px){.appShell{padding:10px}.floatVideo{min-width:190px}.mediaStack{right:10px;overflow:auto}.stageCard{height:580px}.chartBlock{min-height:330px}.demoActions{grid-template-columns:1fr}.sliderRow{grid-template-columns:68px 1fr 50px 24px}}
</style></head>
<body>
<div class="appHeader">
  <h1>RobotWin Online Demo</h1>
  <div class="featureTabs" role="tablist">
    <button class="featureTab active" data-tab="interactive" type="button">Action-MoT</button>
    <button class="featureTab" data-tab="robotwin" type="button">RoboTwin</button>
    <button class="featureTab" data-tab="worldarena" type="button">WorldArena</button>
  </div>
</div>

<main class="appShell">
  <section class="featurePanel active" data-panel="interactive">
    <div class="featureLayout">
      <aside class="infoCol">
        <h2 class="columnTitle">Info</h2>
        <div class="infoBlock"><div class="infoTitle">Mode</div><div id="arm" class="primaryInfo">Absolute joint Action-MoT condition</div></div>
        <div class="infoBlock"><div class="infoTitle">Status</div><div id="status" class="st">Connecting...</div></div>
        <div class="infoBlock"><div class="infoTitle">Prompt</div><div id="prompt" class="inf">&nbsp;</div></div>
        <div class="infoBlock"><div class="infoTitle">Condition</div><div id="rel" class="metric">&nbsp;</div></div>
        <div class="infoBlock chartBlock"><div class="infoTitle">Joint curves</div><div class="chartStack"><div><div class="chartLabel"><span>Left arm</span><span>J0-J5</span></div><canvas id="jointChart" class="jointChart" width="720" height="160"></canvas></div><div><div class="chartLabel"><span>Right arm</span><span>J7-J12</span></div><canvas id="rightJointChart" class="jointChart" width="720" height="160"></canvas></div><div><div class="chartLabel"><span>Gripper</span><span>J6/J13</span></div><canvas id="gripJointChart" class="jointChart" width="720" height="160"></canvas></div></div></div>
        <div class="infoBlock"><div class="infoTitle">Gripper state</div><div class="gripMeter"><span>Left</span><div class="gripTrack"><span id="leftGripFill" class="gripFill"></span></div><span id="leftGripText">0.00</span></div><div class="gripMeter"><span>Right</span><div class="gripTrack"><span id="rightGripFill" class="gripFill"></span></div><span id="rightGripText">0.00</span></div><div class="gripQuick"><button id="openGrip" type="button">Open</button><button id="closeGrip" type="button">Close</button></div></div>
        <div class="infoBlock"><div class="infoTitle">Legend</div><div class="legendRow"><span class="sw swBase"></span><span>base joint state</span></div><div class="legendRow"><span class="sw swL"></span><span>left arm target</span></div><div class="legendRow"><span class="sw swR"></span><span>right arm target</span></div></div>
      </aside>
      <section class="visualCol">
        <h2 class="columnTitle">Visualization</h2>
        <div class="stageCard">
          <div class="stageTitle">3D joint condition</div>
          <canvas id="robot" class="robotCanvas" width="960" height="640"></canvas>
          <div class="mediaStack">
            <div class="floatVideo"><div class="floatTitle">Generated video</div><canvas id="video" width="320" height="384"></canvas></div>
            <div class="floatVideo"><div class="floatTitle">GT / reference</div><canvas id="videoGT" width="320" height="384"></canvas></div>
          </div>
        </div>
        <div class="stageTimeline"><div class="infoTitle">Playback</div><input id="timeline" class="timelineRange" type="range" min="0" max="0" value="0" disabled><div class="timelineMeta"><button id="timelineToggle" type="button" disabled>Play</button><span id="timelineText">0 / 0</span></div></div>
      </section>
      <aside class="controlCol">
        <h2 class="columnTitle">Controller</h2>
        <div class="controlBlock"><div class="demoActions"><button id="sampleInit" class="actionBtn secondaryAction">Sample Init</button><button id="reset" class="actionBtn secondaryAction">Reset State</button><button id="record" class="actionBtn recordAction">Record Trajectory</button><button id="stop" class="actionBtn dangerAction" disabled>Stop Recording</button><button id="infer" class="actionBtn primaryAction">Run Inference</button><a id="dl" href="/video.gif" download="dw05-v5.gif" class="dlBtn downloadAction">Download GIF</a></div><div class="buttonNote">Inference uses the recorded trajectory, or the current target state when no recording exists.</div></div>
        <div class="controlBlock"><div class="infoTitle">操作说明</div><ul class="helpList"><li><strong>视角</strong>: 左键拖拽旋转，Shift/中键/右键拖拽平移，滚轮缩放，双击复位。</li><li><strong>视频</strong>: 上层视频窗口可拖动，GT/参考与生成视频上下排列。</li><li><strong>动作</strong>: 先采样初始帧，再调节关节或 EEF，录制后运行推理。</li><li><strong>播放</strong>: 使用中间底部时间轴检查动作轨迹和生成视频。</li></ul></div>
        <div class="controlBlock"><div class="controlTabs"><button id="modeJoint" class="tabBtn active">Joint</button><button id="modeEef" class="tabBtn">EEF</button></div><div id="sliders" class="jointCtl"></div><div id="eefSliders" class="jointCtl" style="display:none"></div><div class="controlHint">Drag to rotate; Shift-drag, middle-drag, or right-drag to pan; double click resets view.</div></div>
      </aside>
    </div>
  </section>

  <section class="featurePanel" data-panel="robotwin">
    <div class="featureLayout">
      <aside class="infoCol">
        <h2 class="columnTitle">Info</h2>
        <div class="infoBlock"><div class="infoTitle">Status</div><div id="rwSt" class="st">Click Sample to begin</div></div>
        <div class="infoBlock"><div class="infoTitle">Prompt</div><div id="rwInf" class="inf">&nbsp;</div></div>
        <div class="infoBlock"><div class="infoTitle">Source</div><div class="muted">RoboTwin training sample with GT video and joint condition replay.</div></div>
        <div class="infoBlock chartBlock"><div class="infoTitle">Joint trajectory</div><div class="chartStack"><div><div class="chartLabel"><span>Left arm</span><span>J0-J5</span></div><canvas id="rwJointChart" class="jointChart" width="720" height="160"></canvas></div><div><div class="chartLabel"><span>Right arm</span><span>J7-J12</span></div><canvas id="rwRightJointChart" class="jointChart" width="720" height="160"></canvas></div><div><div class="chartLabel"><span>Gripper</span><span>J6/J13</span></div><canvas id="rwGripJointChart" class="jointChart" width="720" height="160"></canvas></div></div></div>
        <div class="infoBlock"><div class="infoTitle">Gripper replay</div><div class="gripMeter"><span>Left</span><div class="gripTrack"><span id="rwLeftGripFill" class="gripFill"></span></div><span id="rwLeftGripText">0.00</span></div><div class="gripMeter"><span>Right</span><div class="gripTrack"><span id="rwRightGripFill" class="gripFill"></span></div><span id="rwRightGripText">0.00</span></div></div>
      </aside>
      <section class="visualCol">
        <h2 class="columnTitle">Visualization</h2>
        <div class="stageCard">
          <div class="stageTitle">RoboTwin joint replay</div>
          <canvas id="cvRC" class="robotCanvas" width="960" height="640"></canvas>
          <div class="mediaStack">
            <div class="floatVideo"><div class="floatTitle">Generated</div><canvas id="cvRG" width="320" height="384"></canvas></div>
            <div class="floatVideo"><div class="floatTitle">GT 3-cam</div><canvas id="cvRT" width="320" height="384"></canvas></div>
          </div>
        </div>
        <div class="stageTimeline"><div class="infoTitle">Playback</div><input id="rwTimeline" class="timelineRange" type="range" min="0" max="0" value="0" disabled><div class="timelineMeta"><button id="rwTimelineToggle" type="button" disabled>Play</button><span id="rwTimelineText">0 / 0</span></div></div>
      </section>
      <aside class="controlCol">
        <h2 class="columnTitle">Controller</h2>
        <div class="controlBlock"><div class="demoActions"><button id="rwBS" class="actionBtn secondaryAction" onclick="rwSample()">Random Sample</button><button id="rwBI" class="actionBtn primaryAction" onclick="rwInfer()" disabled>Run Inference</button><a id="rwDl" href="/robotwin.gif" download="robotwin.gif" class="dlBtn downloadAction">Download GIF</a></div><div class="buttonNote">Generated and GT videos stay synchronized with the 3D replay and the timeline.</div></div>
        <div class="controlBlock"><div class="infoTitle">操作说明</div><ul class="helpList"><li><strong>采样</strong>: 随机载入 RoboTwin 片段，左侧显示 prompt 和动作曲线。</li><li><strong>视频</strong>: 上层视频窗口可拖动，生成与 GT 上下排列。</li><li><strong>推理</strong>: 运行后对比 GT 与生成结果，拖动中间底部时间轴逐帧检查。</li><li><strong>视角</strong>: 3D 区域支持旋转、平移、缩放和双击复位。</li></ul></div>
      </aside>
    </div>
  </section>

  <section class="featurePanel" data-panel="worldarena">
    <div class="featureLayout">
      <aside class="infoCol">
        <h2 class="columnTitle">Info</h2>
        <div class="infoBlock"><div class="infoTitle">Status</div><div id="waSt" class="st">Click Sample to begin</div></div>
        <div class="infoBlock"><div class="infoTitle">Prompt</div><div id="waInf" class="inf">&nbsp;</div></div>
        <div class="infoBlock"><div class="infoTitle">Batch</div><div class="progWrap"><div id="waProg" class="progFill"></div></div><div id="waBatchInfo" class="batchInfo">Full eval idle</div></div>
        <div class="infoBlock chartBlock"><div class="infoTitle">Joint trajectory</div><div class="chartStack"><div><div class="chartLabel"><span>Left arm</span><span>J0-J5</span></div><canvas id="waJointChart" class="jointChart" width="720" height="160"></canvas></div><div><div class="chartLabel"><span>Right arm</span><span>J7-J12</span></div><canvas id="waRightJointChart" class="jointChart" width="720" height="160"></canvas></div><div><div class="chartLabel"><span>Gripper</span><span>J6/J13</span></div><canvas id="waGripJointChart" class="jointChart" width="720" height="160"></canvas></div></div></div>
        <div class="infoBlock"><div class="infoTitle">Gripper replay</div><div class="gripMeter"><span>Left</span><div class="gripTrack"><span id="waLeftGripFill" class="gripFill"></span></div><span id="waLeftGripText">0.00</span></div><div class="gripMeter"><span>Right</span><div class="gripTrack"><span id="waRightGripFill" class="gripFill"></span></div><span id="waRightGripText">0.00</span></div></div>
      </aside>
      <section class="visualCol">
        <h2 class="columnTitle">Visualization</h2>
        <div class="stageCard">
          <div class="stageTitle">WorldArena joint replay</div>
          <canvas id="cvWC" class="robotCanvas" width="960" height="640"></canvas>
          <div class="mediaStack">
            <div class="floatVideo"><div class="floatTitle">Generated</div><canvas id="cvWG" width="320" height="384"></canvas></div>
            <div class="floatVideo"><div class="floatTitle">GT</div><canvas id="cvWT" width="320" height="384"></canvas></div>
          </div>
        </div>
        <div class="stageTimeline"><div class="infoTitle">Playback</div><input id="waTimeline" class="timelineRange" type="range" min="0" max="0" value="0" disabled><div class="timelineMeta"><button id="waTimelineToggle" type="button" disabled>Play</button><span id="waTimelineText">0 / 0</span></div></div>
      </section>
      <aside class="controlCol">
        <h2 class="columnTitle">Controller</h2>
        <div class="controlBlock"><div class="waSelector">Dataset <select id="waDataset" onchange="waDatasetChanged()"><option value="eval750">Eval-750</option><option value="test1000">Test-1000</option></select></div><div class="demoActions"><button id="waBS" class="actionBtn secondaryAction" onclick="waSample()">Random Sample</button><button id="waBI" class="actionBtn primaryAction" onclick="waInfer()" disabled>Run Inference</button><a id="waDl" href="/worldarena.gif" download="worldarena.gif" class="dlBtn downloadAction">Download GIF</a><button id="waBE" class="actionBtn evalAction" onclick="waBatchEval()" disabled>Run Full Eval</button></div><div class="buttonNote">Sample-level inference is available here; full evaluation remains gated by the backend runner.</div></div>
        <div class="controlBlock"><div class="infoTitle">操作说明</div><ul class="helpList"><li><strong>数据集</strong>: 选择 split 后采样 episode，GT 会显示在中间上层。</li><li><strong>视频</strong>: 上层视频窗口可拖动，GT 与生成视频上下排列。</li><li><strong>对比</strong>: 推理完成后同时播放 GT、生成视频和 3D 条件。</li><li><strong>导出</strong>: 按需选择输出内容，再下载或触发评测流程。</li></ul></div>
        <div class="controlBlock"><div class="infoTitle">Eval output</div><div class="batchOpts"><label><input id="waOptGt" type="checkbox" checked>GT video</label><label><input id="waOptGen" type="checkbox" checked>generated video</label><label><input id="waOptCond" type="checkbox" checked>condition video</label><label><input id="waOptCrop" type="checkbox">main view only</label><label><input id="waOptMerge" type="checkbox" checked>merge after eval</label></div></div>
      </aside>
    </div>
  </section>
</main>

<script>
function drawImg(ctx,b64,w,h){const img=new Image();img.src='data:image/jpeg;base64,'+b64;img.onload=()=>ctx.drawImage(img,0,0,w,h)}
function fitCanvasToBox(canvas){const rect=canvas.getBoundingClientRect();const dpr=window.devicePixelRatio||1;const w=Math.max(320,Math.round(rect.width*dpr));const h=Math.max(220,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;return true}return false}
const WS_ORIGIN=(location.protocol==='https:'?'wss://':'ws://')+location.host;
const WS_PATH_BASE=location.pathname.endsWith('/') ? location.pathname.slice(0,-1) : location.pathname.replace(/\/[^\/]*$/, '');
function wsUrl(path){return WS_ORIGIN+WS_PATH_BASE+path;}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}
const ALOHA_KIN=__ALOHA_KINEMATICS_JSON__;
const ZERO_ORIGIN={xyz:[0,0,0],rpy:[0,0,0]};
function conditionValues(q,base){return q.slice()}
function vAdd(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]}
function vSub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
function vDot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function vCross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function vNorm(a){const n=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/n,a[1]/n,a[2]/n]}
function matId(){return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}
function matMul(a,b){const r=new Array(16);for(let row=0;row<4;row++){for(let col=0;col<4;col++){let s=0;for(let k=0;k<4;k++)s+=a[row*4+k]*b[k*4+col];r[row*4+col]=s}}return r}
function matTranslate(v){return [1,0,0,v[0],0,1,0,v[1],0,0,1,v[2],0,0,0,1]}
function matRotX(a){const c=Math.cos(a),s=Math.sin(a);return [1,0,0,0,0,c,-s,0,0,s,c,0,0,0,0,1]}
function matRotY(a){const c=Math.cos(a),s=Math.sin(a);return [c,0,s,0,0,1,0,0,-s,0,c,0,0,0,0,1]}
function matRotZ(a){const c=Math.cos(a),s=Math.sin(a);return [c,-s,0,0,s,c,0,0,0,0,1,0,0,0,0,1]}
function matAxis(axis,angle){const u=vNorm(axis||[0,0,1]),x=u[0],y=u[1],z=u[2],c=Math.cos(angle),s=Math.sin(angle),t=1-c;return [t*x*x+c,t*x*y-s*z,t*x*z+s*y,0,t*x*y+s*z,t*y*y+c,t*y*z-s*x,0,t*x*z-s*y,t*y*z+s*x,t*z*z+c,0,0,0,0,1]}
function matRPY(rpy){rpy=rpy||[0,0,0];return matMul(matMul(matRotZ(rpy[2]||0),matRotY(rpy[1]||0)),matRotX(rpy[0]||0))}
function originMat(origin){origin=origin||ZERO_ORIGIN;return matMul(matTranslate(origin.xyz||[0,0,0]),matRPY(origin.rpy||[0,0,0]))}
function jointMat(j,value){let m=originMat(j.origin);const typ=j.type||'fixed';if(typ==='revolute'||typ==='continuous')m=matMul(m,matAxis(j.axis,value||0));else if(typ==='prismatic')m=matMul(m,matTranslate((j.axis||[0,0,1]).map(v=>v*(value||0))));return m}
function tfPoint(m,p){return [m[0]*p[0]+m[1]*p[1]+m[2]*p[2]+m[3],m[4]*p[0]+m[5]*p[1]+m[6]*p[2]+m[7],m[8]*p[0]+m[9]*p[1]+m[10]*p[2]+m[11]]}
function quatNorm(q){const n=Math.hypot(q[0],q[1],q[2],q[3])||1;return [q[0]/n,q[1]/n,q[2]/n,q[3]/n]}
function quatToEuler(q){q=quatNorm(q||[0,0,0,1]);const x=q[0],y=q[1],z=q[2],w=q[3];const sinr=2*(w*x+y*z),cosr=1-2*(x*x+y*y);const roll=Math.atan2(sinr,cosr);let sinp=2*(w*y-z*x);sinp=clamp(sinp,-1,1);const pitch=Math.asin(sinp);const siny=2*(w*z+x*y),cosy=1-2*(y*y+z*z);const yaw=Math.atan2(siny,cosy);return [roll,pitch,yaw]}
function eulerToQuat(rpy){const r=rpy[0]||0,p=rpy[1]||0,y=rpy[2]||0;const cr=Math.cos(r*0.5),sr=Math.sin(r*0.5),cp=Math.cos(p*0.5),sp=Math.sin(p*0.5),cy=Math.cos(y*0.5),sy=Math.sin(y*0.5);return quatNorm([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy])}
function eefQuatToEulerState(src){const q=(src&&src.length>=16)?src:new Array(16).fill(0);const out=new Array(14).fill(0);for(const pair of [[0,0],[8,7]]){const eoff=pair[0],ooff=pair[1];out[ooff]=q[eoff]||0;out[ooff+1]=q[eoff+1]||0;out[ooff+2]=q[eoff+2]||0;const rpy=quatToEuler([q[eoff+3]||0,q[eoff+4]||0,q[eoff+5]||0,(q[eoff+6]===undefined)?1:q[eoff+6]]);out[ooff+3]=rpy[0];out[ooff+4]=rpy[1];out[ooff+5]=rpy[2];out[ooff+6]=clamp(q[eoff+7]||0,0,1)}return out}
function eefEulerToQuatState(src){const e=(src&&src.length>=14)?src:new Array(14).fill(0);const out=new Array(16).fill(0);for(const pair of [[0,0],[7,8]]){const ooff=pair[0],eoff=pair[1];out[eoff]=e[ooff]||0;out[eoff+1]=e[ooff+1]||0;out[eoff+2]=e[ooff+2]||0;const q=eulerToQuat([e[ooff+3]||0,e[ooff+4]||0,e[ooff+5]||0]);out[eoff+3]=q[0];out[eoff+4]=q[1];out[eoff+5]=q[2];out[eoff+6]=q[3];out[eoff+7]=clamp(e[ooff+6]||0,0,1)}return out}
function eefStateToEuler(src){if(!src)return null;return src.length===14?src.slice():eefQuatToEulerState(src)}
const robotViews=new WeakMap();
function getRobotView(canvas){let v=robotViews.get(canvas);if(!v){v={yaw:0,pitch:0.22,zoom:1,target:[0.08,0,0.48],ctx:null,lastQ:null,lastBase:null,lastMetric:null,drag:false,pan:false,lastX:0,lastY:0,init:false};robotViews.set(canvas,v)}return v}
function cameraProjector(canvas,view){const w=canvas.width,h=canvas.height,cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);const dir=vNorm([cp*Math.cos(view.yaw),cp*Math.sin(view.yaw),sp]);let right=vNorm(vCross([0,0,1],dir));if(!isFinite(right[0]))right=[0,1,0];const up=vNorm(vCross(dir,right));const scale=Math.min(w,h)*0.72*view.zoom;return p=>{const d=vSub(p,view.target);return {x:w*0.5+vDot(d,right)*scale,y:h*0.56-vDot(d,up)*scale,z:vDot(d,dir)}}}
function cameraBasis(view){const cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),dir=vNorm([cp*Math.cos(view.yaw),cp*Math.sin(view.yaw),sp]);let right=vNorm(vCross([0,0,1],dir));if(!isFinite(right[0]))right=[0,1,0];return {right,up:vNorm(vCross(dir,right))}}
function drawLine2(ctx,a,b,color,width,alpha){ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.restore()}
function drawDot2(ctx,p,color,r,alpha){ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();ctx.restore()}
function drawBackground3D(ctx,project){const c=ctx.canvas,w=c.width,h=c.height;ctx.clearRect(0,0,w,h);const g=ctx.createLinearGradient(0,0,0,h);g.addColorStop(0,'#242424');g.addColorStop(1,'#181818');ctx.fillStyle=g;ctx.fillRect(0,0,w,h);for(let x=-0.45;x<=0.65;x+=0.1)drawLine2(ctx,project([x,-0.55,0]),project([x,0.55,0]),'#303030',1,1);for(let y=-0.55;y<=0.55;y+=0.1)drawLine2(ctx,project([-0.45,y,0]),project([0.65,y,0]),'#303030',1,1);drawLine2(ctx,project([0,0,0]),project([0.18,0,0]),'#b66',2,0.8);drawLine2(ctx,project([0,0,0]),project([0,0.18,0]),'#6b6',2,0.8);drawLine2(ctx,project([0,0,0]),project([0,0,0.18]),'#66b',2,0.8)}
const BOX_EDGES=[[0,1],[0,2],[0,4],[3,1],[3,2],[3,7],[5,1],[5,4],[5,7],[6,2],[6,4],[6,7]];
const BOX_FACES=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
function boxCorners(b){const mn=b.min||[-0.02,-0.02,-0.02],mx=b.max||[0.02,0.02,0.02];return [[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],[mn[0],mx[1],mn[2]],[mx[0],mx[1],mn[2]],[mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],[mn[0],mx[1],mx[2]],[mx[0],mx[1],mx[2]]]}
function hexRgb(hex){hex=(hex||'#888').replace('#','');if(hex.length===3)hex=hex.split('').map(c=>c+c).join('');const n=parseInt(hex,16);return [(n>>16)&255,(n>>8)&255,n&255]}
function rgbStr(rgb,k){return `rgb(${Math.round(rgb[0]*k)},${Math.round(rgb[1]*k)},${Math.round(rgb[2]*k)})`}
function drawLinkMesh(ctx,project,M,link,color,alpha,width){const tris=link.triangles||[];if(!tris.length)return false;const base=hexRgb(color),light=vNorm([-0.35,-0.45,0.82]);const faces=[];for(const tri of tris){const p0=tfPoint(M,[tri[0],tri[1],tri[2]]),p1=tfPoint(M,[tri[3],tri[4],tri[5]]),p2=tfPoint(M,[tri[6],tri[7],tri[8]]);const pp0=project(p0),pp1=project(p1),pp2=project(p2);const n=vNorm(vCross(vSub(p1,p0),vSub(p2,p0)));const shade=0.36+0.58*Math.max(0,vDot(n,light));faces.push({pts:[pp0,pp1,pp2],z:(pp0.z+pp1.z+pp2.z)/3,shade})}faces.sort((a,b)=>a.z-b.z);const fillAlpha=link.static_mesh?Math.min(0.28,alpha):alpha*0.95,edgeAlpha=link.static_mesh?Math.min(0.10,alpha*0.34):alpha*0.72,edgeEvery=link.static_mesh?42:8;ctx.save();ctx.globalAlpha=fillAlpha;for(const f of faces){ctx.fillStyle=rgbStr(base,f.shade);ctx.beginPath();ctx.moveTo(f.pts[0].x,f.pts[0].y);ctx.lineTo(f.pts[1].x,f.pts[1].y);ctx.lineTo(f.pts[2].x,f.pts[2].y);ctx.closePath();ctx.fill()}ctx.globalAlpha=edgeAlpha;ctx.strokeStyle=rgbStr(base,0.92);ctx.lineWidth=Math.max(0.35,(width||1)*0.35);for(let i=0;i<faces.length;i+=edgeEvery){const f=faces[i];ctx.beginPath();ctx.moveTo(f.pts[0].x,f.pts[0].y);ctx.lineTo(f.pts[1].x,f.pts[1].y);ctx.lineTo(f.pts[2].x,f.pts[2].y);ctx.closePath();ctx.stroke()}ctx.restore();return true}
function fillBoxFaces(ctx,pts,color,alpha){ctx.save();ctx.fillStyle=color;ctx.globalAlpha=alpha;BOX_FACES.map(face=>({face,z:face.reduce((s,i)=>s+pts[i].z,0)/4})).sort((a,b)=>a.z-b.z).forEach(f=>{ctx.beginPath();ctx.moveTo(pts[f.face[0]].x,pts[f.face[0]].y);for(let i=1;i<f.face.length;i++)ctx.lineTo(pts[f.face[i]].x,pts[f.face[i]].y);ctx.closePath();ctx.fill()});ctx.restore()}
function strokeBoxEdges(ctx,pts,color,alpha,width){ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width||1.2;BOX_EDGES.forEach(e=>{ctx.beginPath();ctx.moveTo(pts[e[0]].x,pts[e[0]].y);ctx.lineTo(pts[e[1]].x,pts[e[1]].y);ctx.stroke()});ctx.restore()}
function drawLinkBox(ctx,project,T,link,color,alpha,width){if(!link)return;const M=matMul(T,originMat(link.origin||ZERO_ORIGIN));if(drawLinkMesh(ctx,project,M,link,color,alpha,width))return;const bbox=link.bbox||{min:[-0.018,-0.018,-0.018],max:[0.018,0.018,0.018]},pts=boxCorners(bbox).map(p=>project(tfPoint(M,p)));fillBoxFaces(ctx,pts,color,alpha*0.14);strokeBoxEdges(ctx,pts,color,alpha*0.75,width||1.2)}
function drawStaticModel(ctx,project){(ALOHA_KIN.static_parts||[]).forEach(part=>{let T=matId();(part.joints||[]).forEach(j=>{T=matMul(T,jointMat(j,0))});drawLinkBox(ctx,project,T,part.link,part.color||'#888',0.24,0.75)})}
function jointValueForArm(j,vals){if(typeof j.value_index!=='number')return 0;let v=vals[j.value_index]||0;if(j.type==='prismatic'){const limit=j.limit||{},lo=Number.isFinite(limit.lower)?limit.lower:0,hi=Number.isFinite(limit.upper)?limit.upper:(ALOHA_KIN.gripper_upper||0.04765);v=clamp(v,0,1)*Math.max(Math.abs(hi-lo),ALOHA_KIN.gripper_upper||0.04765)+Math.min(lo,hi);v=clamp(v,Math.min(lo,hi),Math.max(lo,hi))}return v}
function armPose3D(side,vals){const arm=ALOHA_KIN.arms[side];const frames=[],pts=[],edges=[];function visit(linkName,T,parentPt){const link=arm.links[linkName];if(link)frames.push({T,link});const here=tfPoint(T,[0,0,0]);pts.push(here);if(parentPt)edges.push([parentPt,here]);(arm.children[linkName]||[]).forEach(j=>{const JT=matMul(T,jointMat(j,jointValueForArm(j,vals)));if(j.child)visit(j.child,JT,here)})}const rootT=matMul(matId(),jointMat(arm.base_joint,0));visit(arm.base_joint.child,rootT,null);return {frames,pts,edges}}
function drawArm3D(ctx,project,side,vals,color,ghost){const pose=armPose3D(side,vals);const alpha=ghost?0.58:0.98,width=ghost?1.25:1.75;pose.frames.forEach(f=>drawLinkBox(ctx,project,f.T,f.link,color,alpha,width));pose.edges.forEach(e=>drawLine2(ctx,project(e[0]),project(e[1]),color,ghost?1.9:2.8,ghost?0.38:0.62));pose.pts.forEach((p,i)=>{if(i%2===0)drawDot2(ctx,project(p),color,ghost?2.4:3.1,ghost?0.48:0.78)})}
function drawRobotCanvas(ctx,q,base,metricEl){q=q||new Array(14).fill(0);base=base||q;const view=getRobotView(ctx.canvas);view.ctx=ctx;view.lastQ=q.slice();view.lastBase=base.slice();view.lastMetric=metricEl||null;const project=cameraProjector(ctx.canvas,view);drawBackground3D(ctx,project);drawStaticModel(ctx,project);drawArm3D(ctx,project,'left',base.slice(0,7),'#d9dde3',true);drawArm3D(ctx,project,'right',base.slice(7,14),'#d9dde3',true);drawArm3D(ctx,project,'left',q.slice(0,7),'#ff2a1f',false);drawArm3D(ctx,project,'right',q.slice(7,14),'#20f070',false);if(metricEl){const vals=conditionValues(q,base);metricEl.textContent='absolute joint action condition  '+vals.map(v=>(v||0).toFixed(2)).join('  ')}}
function redrawRobotCanvas(canvas){if(canvas.getClientRects().length)fitCanvasToBox(canvas);const view=getRobotView(canvas);if(view.ctx&&view.lastQ)drawRobotCanvas(view.ctx,view.lastQ,view.lastBase,view.lastMetric)}
function initRobotCanvas(canvas){const view=getRobotView(canvas);if(view.init)return;view.init=true;canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('pointerdown',e=>{view.drag=true;view.pan=e.shiftKey||e.button===1||e.button===2;view.lastX=e.clientX;view.lastY=e.clientY;canvas.classList.add('dragging');try{canvas.setPointerCapture(e.pointerId)}catch(_){}e.preventDefault()});canvas.addEventListener('pointermove',e=>{if(!view.drag)return;const dx=e.clientX-view.lastX,dy=e.clientY-view.lastY;view.lastX=e.clientX;view.lastY=e.clientY;if(view.pan){const basis=cameraBasis(view),scale=Math.min(canvas.width,canvas.height)*0.72*view.zoom,worldScale=1/Math.max(80,scale);view.target=vAdd(view.target,vAdd(basis.right.map(v=>-dx*worldScale*v),basis.up.map(v=>dy*worldScale*v)))}else{view.yaw+=dx*0.01;view.pitch=clamp(view.pitch+dy*0.008,-1.15,1.15)}redrawRobotCanvas(canvas)});function endDrag(e){view.drag=false;view.pan=false;canvas.classList.remove('dragging');try{canvas.releasePointerCapture(e.pointerId)}catch(_){}}canvas.addEventListener('pointerup',endDrag);canvas.addEventListener('pointercancel',endDrag);canvas.addEventListener('pointerleave',()=>{view.drag=false;view.pan=false;canvas.classList.remove('dragging')});canvas.addEventListener('wheel',e=>{view.zoom=clamp(view.zoom*Math.exp(-e.deltaY*0.001),0.55,2.2);redrawRobotCanvas(canvas);e.preventDefault()},{passive:false});canvas.addEventListener('dblclick',()=>{view.yaw=0;view.pitch=0.22;view.zoom=1;view.target=[0.08,0,0.48];redrawRobotCanvas(canvas)})}
function clampMediaStack(stack){const parent=stack.closest('.stageCard');if(!parent||!stack.getClientRects().length)return;const pr=parent.getBoundingClientRect(),sr=stack.getBoundingClientRect();const maxX=Math.max(12,pr.width-sr.width-12),maxY=Math.max(12,pr.height-sr.height-12);const x=clamp(Number(stack.dataset.x||12),12,maxX),y=clamp(Number(stack.dataset.y||12),12,maxY);stack.dataset.x=String(x);stack.dataset.y=String(y);stack.style.left=x+'px';stack.style.top=y+'px'}
function initMediaStack(stack){if(stack.dataset.init)return;stack.dataset.init='1';stack.dataset.x=stack.dataset.x||'12';stack.dataset.y=stack.dataset.y||'12';clampMediaStack(stack);stack.addEventListener('pointerdown',e=>{if(e.button!==0)return;stack.dataset.drag='1';stack.dataset.startX=String(e.clientX);stack.dataset.startY=String(e.clientY);stack.dataset.baseX=stack.dataset.x||'12';stack.dataset.baseY=stack.dataset.y||'12';stack.classList.add('dragging');try{stack.setPointerCapture(e.pointerId)}catch(_){}e.stopPropagation();e.preventDefault()});stack.addEventListener('pointermove',e=>{if(stack.dataset.drag!=='1')return;const dx=e.clientX-Number(stack.dataset.startX||e.clientX),dy=e.clientY-Number(stack.dataset.startY||e.clientY);stack.dataset.x=String(Number(stack.dataset.baseX||12)+dx);stack.dataset.y=String(Number(stack.dataset.baseY||12)+dy);clampMediaStack(stack);e.stopPropagation();e.preventDefault()});function end(e){if(stack.dataset.drag!=='1')return;stack.dataset.drag='0';stack.classList.remove('dragging');try{stack.releasePointerCapture(e.pointerId)}catch(_){}e.stopPropagation();e.preventDefault()}stack.addEventListener('pointerup',end);stack.addEventListener('pointercancel',end);stack.addEventListener('lostpointercapture',()=>{stack.dataset.drag='0';stack.classList.remove('dragging')})}
function resizeRobotCanvases(){document.querySelectorAll('.stageCard .robotCanvas').forEach(canvas=>{if(canvas.getClientRects().length&&fitCanvasToBox(canvas))redrawRobotCanvas(canvas)})}
function clampMediaStacks(){document.querySelectorAll('.mediaStack').forEach(clampMediaStack)}
function telemetryId(prefix,name){return prefix?prefix+name:name.charAt(0).toLowerCase()+name.slice(1)}
function chartId(prefix,name){return prefix?prefix+name:name.charAt(0).toLowerCase()+name.slice(1)}
function nearestStateIndex(rows,current,dims){if(!rows||!rows.length||!current)return -1;let best=0,bestErr=Infinity;rows.forEach((row,idx)=>{let err=0;(dims||[]).forEach(dim=>{const a=Number(row&&row[dim])||0,b=Number(current&&current[dim])||0;err+=Math.abs(a-b)});if(err<bestErr){bestErr=err;best=idx}});return best}
function resizeChartCanvas(canvas){if(!canvas||!canvas.getClientRects().length)return;const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(320,Math.round(rect.width*dpr)),h=Math.max(100,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}}
function drawJointChart(canvas,states,current,dims,label){if(!canvas)return;resizeChartCanvas(canvas);const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);const g=ctx.createLinearGradient(0,0,0,h);g.addColorStop(0,'#141b23');g.addColorStop(1,'#0d1218');ctx.fillStyle=g;ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(132,148,168,.18)';ctx.lineWidth=1;for(let i=1;i<4;i++){const y=h*i/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}for(let i=1;i<8;i++){const x=w*i/8;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}const rows=(states&&states.length)?states:[current||new Array(14).fill(0)];const useDims=dims&&dims.length?dims:[0,1,2,3,4,5];const colors=['#ff665e','#ff9c63','#ffd166','#64d98a','#5fc0ff','#b58cff','#57f070','#46c2ff','#9bd35d','#ff7ac8','#8ea5ff','#dce3ec'];useDims.forEach((dim,ci)=>{ctx.strokeStyle=colors[ci%colors.length];ctx.globalAlpha=useDims.length<=2?.9:.72;ctx.lineWidth=useDims.length<=2?2.2:1.65;ctx.beginPath();rows.forEach((row,idx)=>{const raw=Number(row&&row[dim])||0;const lim=jointLimit(dim);const denom=Math.max(1e-6,lim[1]-lim[0]);const norm=clamp((raw-lim[0])/denom,0,1);const x=rows.length>1?idx*(w-1)/(rows.length-1):w-1;const y=h-10-norm*(h-22);if(idx===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke()});const mark=nearestStateIndex(rows,current,useDims);if(mark>=0&&rows.length>1){const x=mark*(w-1)/(rows.length-1);ctx.globalAlpha=1;ctx.strokeStyle='rgba(255,255,255,.42)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}ctx.globalAlpha=1;ctx.fillStyle='rgba(220,228,238,.78)';ctx.font='12px ui-monospace,monospace';ctx.fillText(label||'joint trajectory',12,18);useDims.slice(0,6).forEach((dim,ci)=>{ctx.fillStyle=colors[ci%colors.length];ctx.fillText('J'+dim,12+ci*46,h-11)})}
function updateGrip(prefix,state){const s=state||new Array(14).fill(0);const l=clamp(Number(s[6])||0,0,1),r=clamp(Number(s[13])||0,0,1);const lf=document.getElementById(telemetryId(prefix,'LeftGripFill')),rf=document.getElementById(telemetryId(prefix,'RightGripFill')),lt=document.getElementById(telemetryId(prefix,'LeftGripText')),rt=document.getElementById(telemetryId(prefix,'RightGripText'));if(lf)lf.style.width=Math.round(l*100)+'%';if(rf)rf.style.width=Math.round(r*100)+'%';if(lt)lt.textContent=l.toFixed(2);if(rt)rt.textContent=r.toFixed(2)}
function updateTelemetry(prefix,state,states){drawJointChart(document.getElementById(chartId(prefix,'JointChart')),states,state,[0,1,2,3,4,5],'left arm joints');drawJointChart(document.getElementById(chartId(prefix,'RightJointChart')),states,state,[7,8,9,10,11,12],'right arm joints');drawJointChart(document.getElementById(chartId(prefix,'GripJointChart')),states,state,[6,13],'gripper aperture');updateGrip(prefix,state)}
function redrawTelemetry(){updateTelemetry('',q,[base,q]);if(typeof rwStates!=='undefined')updateTelemetry('rw',rwStates&&rwStates[0],rwStates);if(typeof waStates!=='undefined')updateTelemetry('wa',waStates&&waStates[0],waStates)}
function timelineId(prefix,name){return prefix?prefix+name:name.charAt(0).toLowerCase()+name.slice(1)}
function timelineEls(prefix){return {range:document.getElementById(timelineId(prefix,'Timeline')),toggle:document.getElementById(timelineId(prefix,'TimelineToggle')),text:document.getElementById(timelineId(prefix,'TimelineText'))}}
function stopPlayback(loop){if(loop&&loop.tmr){clearInterval(loop.tmr);loop.tmr=null}if(loop){loop.playing=false;const els=timelineEls(loop.prefix||'');if(els.toggle)els.toggle.textContent='Play'}}
function setTimeline(loop,idx){const els=timelineEls(loop.prefix||''),total=Math.max(0,loop.total||0);const val=total?clamp(idx,0,total-1):0;loop.index=val;if(els.range)els.range.value=String(val);if(els.text)els.text.textContent=total?`${val+1} / ${total}`:'0 / 0'}
function renderPlayback(loop,idx){if(!loop||!loop.render)return;setTimeline(loop,idx);loop.render(loop.index)}
function configurePlayback(loop,prefix,total,fps,render){stopPlayback(loop);loop.prefix=prefix||'';loop.total=Math.max(0,total||0);loop.fps=fps||8;loop.render=render;loop.index=0;const els=timelineEls(loop.prefix);if(els.range){els.range.disabled=loop.total<=1;els.range.min='0';els.range.max=String(Math.max(0,loop.total-1));els.range.value='0';els.range.oninput=()=>renderPlayback(loop,parseInt(els.range.value||'0',10)||0)}if(els.toggle){els.toggle.disabled=loop.total<=1;els.toggle.textContent='Play';els.toggle.onclick=()=>{if(loop.playing)stopPlayback(loop);else startPlayback(loop)}}setTimeline(loop,0);if(loop.total)renderPlayback(loop,0)}
function startPlayback(loop){if(!loop||loop.total<=1)return;stopPlayback(loop);loop.playing=true;const els=timelineEls(loop.prefix||'');if(els.toggle)els.toggle.textContent='Pause';loop.tmr=setInterval(()=>{const next=((loop.index||0)+1)%loop.total;renderPlayback(loop,next)},1000/(loop.fps||8))}
function playImageLoop(loop, ctxA, wA, hA, framesA, fps, ctxB, wB, hB, framesB){let i=0;if(loop.tmr)clearInterval(loop.tmr);loop.tmr=setInterval(()=>{const n=Math.max((framesA||[]).length,(framesB||[]).length,1);const k=i++%n;if(framesA&&framesA.length)drawImg(ctxA,framesA[k%framesA.length],wA,hA);if(ctxB&&framesB&&framesB.length)drawImg(ctxB,framesB[k%framesB.length],wB,hB);},1000/(fps||8))}
function playRobotLoop(loop, ctxA, wA, hA, framesA, robotCtx, states, bases, fps, prefix){const n=Math.max((framesA||[]).length,(states||[]).length,1);updateTelemetry(prefix||'',states&&states[0],states);configurePlayback(loop,prefix||'',n,fps||8,k=>{if(framesA&&framesA.length)drawImg(ctxA,framesA[k%framesA.length],wA,hA);if(states&&states.length){const s=states[k%states.length],b=(bases&&bases.length)?bases[k%bases.length]:s;drawRobotCanvas(robotCtx,s,b);updateTelemetry(prefix||'',s,states)}});startPlayback(loop)}
function playTwoImageRobotLoop(loop, ctxA, wA, hA, framesA, ctxB, wB, hB, framesB, robotCtx, states, bases, fps, prefix){const n=Math.max((framesA||[]).length,(framesB||[]).length,(states||[]).length,1);updateTelemetry(prefix||'',states&&states[0],states);configurePlayback(loop,prefix||'',n,fps||8,k=>{if(framesA&&framesA.length)drawImg(ctxA,framesA[k%framesA.length],wA,hA);if(framesB&&framesB.length)drawImg(ctxB,framesB[k%framesB.length],wB,hB);if(states&&states.length){const s=states[k%states.length],b=(bases&&bases.length)?bases[k%bases.length]:s;drawRobotCanvas(robotCtx,s,b);updateTelemetry(prefix||'',s,states)}});startPlayback(loop)}
function playInteractiveReplay(loop, videoCtx, w, h, frames, states, bases, eefStates, fps){const n=Math.max((frames||[]).length,(states||[]).length,1);updateTelemetry('',states&&states[0],states);configurePlayback(loop,'',n,fps||8,k=>{if(frames&&frames.length)drawImg(videoCtx,frames[k%frames.length],w,h);if(states&&states.length){const s=states[k%states.length],b=(bases&&bases.length)?bases[k%bases.length]:s,e=(eefStates&&eefStates.length)?eefStates[k%eefStates.length]:null;setJointState({state:s,base:b,eef_state:e,recording:false,steps:states.length,chart_states:states})}});startPlayback(loop)}

function activateFeatureTab(name){document.querySelectorAll('.featureTab').forEach(btn=>btn.classList.toggle('active',btn.dataset.tab===name));document.querySelectorAll('.featurePanel').forEach(panel=>panel.classList.toggle('active',panel.dataset.panel===name));setTimeout(()=>{resizeRobotCanvases();clampMediaStacks();document.querySelectorAll('.robotCanvas').forEach(redrawRobotCanvas)},0)}
document.querySelectorAll('.featureTab').forEach(btn=>btn.addEventListener('click',()=>activateFeatureTab(btn.dataset.tab)));

// Interactive joint demo
const video=document.getElementById('video'), vctx=video.getContext('2d'), videoGT=document.getElementById('videoGT'), gtctx=videoGT.getContext('2d');
const robot=document.getElementById('robot'), rctx=robot.getContext('2d');
const statusEl=document.getElementById('status'), promptEl=document.getElementById('prompt'), relEl=document.getElementById('rel'), slidersEl=document.getElementById('sliders');
const btnSampleInit=document.getElementById('sampleInit'), btnReset=document.getElementById('reset'), btnRec=document.getElementById('record'), btnStop=document.getElementById('stop'), btnInfer=document.getElementById('infer'), dl=document.getElementById('dl');
let q=new Array(14).fill(0), base=new Array(14).fill(0), initQ=new Array(14).fill(0), eef=new Array(14).fill(0), initEef=new Array(14).fill(0), names=[], eefNames=[], recording=false, steps=0, kbLoop={tmr:null}, eefTimer=null, controlMode='joint', sid='';
let rwSid='', waSid='';
const eefSlidersEl=document.getElementById('eefSliders'), modeJoint=document.getElementById('modeJoint'), modeEef=document.getElementById('modeEef');
const btnOpenGrip=document.getElementById('openGrip'), btnCloseGrip=document.getElementById('closeGrip');
function sendState(){if(ws.readyState===1)ws.send(JSON.stringify({action:'set_joint',state:q}))}
function updateGifLinks(){if(sid)dl.href='/video.gif?sid='+encodeURIComponent(sid);if(rwSid)rwDl.href='/robotwin.gif?sid='+encodeURIComponent(rwSid);if(waSid)waDl.href='/worldarena.gif?sid='+encodeURIComponent(waSid)}
function limitedSliderValue(input,prev,cap){const raw=parseFloat(input.value);const old=Number.isFinite(prev)?prev:raw;const lim=clamp(raw,old-cap,old+cap);if(Math.abs(lim-raw)>1e-9)input.value=lim;return lim}
function sendEefState(){if(eefTimer)clearTimeout(eefTimer);eefTimer=setTimeout(()=>{statusEl.textContent='[solving IK...]';if(ws.readyState===1)ws.send(JSON.stringify({action:'set_eef',state:eef}))},90)}
function addReset(row,onReset){const btn=document.createElement('button');btn.type='button';btn.className='miniReset';btn.textContent='R';btn.title='Reset this dimension';btn.addEventListener('click',e=>{e.preventDefault();onReset()});row.appendChild(btn)}
function jointLimit(i){const lim=(ALOHA_KIN.joint_limits||[])[i];return lim&&lim.length===2?lim:((i===6||i===13)?[0,1]:[-3.14,3.14])}
function clampToJointLimit(i,v){const lim=jointLimit(i);return clamp(v,lim[0],lim[1])}
function addSlider(i,parent){const row=document.createElement('label');row.className='sliderRow';const span=document.createElement('span');span.textContent=names[i]||('J'+i);const input=document.createElement('input');input.type='range';input.step='0.01';input.dataset.idx=i;const lim=jointLimit(i);input.min=String(lim[0]);input.max=String(lim[1]);const out=document.createElement('output');out.dataset.idx=i;row.append(span,input,out);addReset(row,()=>{q[i]=clampToJointLimit(i,initQ[i]||0);syncSliders();drawRobotCanvas(rctx,q,base,relEl);updateTelemetry('',q,[base,q]);sendState()});parent.appendChild(row);input.addEventListener('input',()=>{q[i]=clampToJointLimit(i,limitedSliderValue(input,q[i],(i===6||i===13)?0.05:0.08));input.value=q[i];out.value=q[i].toFixed(2);drawRobotCanvas(rctx,q,base,relEl);updateTelemetry('',q,[base,q]);sendState()});}
function addEefSlider(i,parent){const row=document.createElement('label');row.className='sliderRow';const span=document.createElement('span');span.textContent=eefNames[i]||('E'+i);const input=document.createElement('input');input.type='range';input.dataset.idx=i;const out=document.createElement('output');out.dataset.idx=i;row.append(span,input,out);addReset(row,()=>{eef[i]=initEef[i]||0;syncEefSliders();sendEefState()});parent.appendChild(row);input.addEventListener('input',()=>{const k=i%7;const cap=(k===6)?0.05:(k>=3&&k<=5?0.08:0.015);eef[i]=limitedSliderValue(input,eef[i],cap);out.value=eefValueText(i,eef[i]);sendEefState()});}
function makeSliders(){slidersEl.innerHTML='';const col=document.createElement('div');col.className='jointCol';for(let i=0;i<14;i++)addSlider(i,col);slidersEl.append(col);}
function makeEefSliders(){eefSlidersEl.innerHTML='';const col=document.createElement('div');col.className='jointCol';for(let i=0;i<14;i++)addEefSlider(i,col);eefSlidersEl.append(col);}
function eefValueText(i,v){const k=i%7;return k===6?(v||0).toFixed(2):(k>=3&&k<=5?(v||0).toFixed(2):(v||0).toFixed(3))}
function syncSliders(){document.querySelectorAll('#sliders input[type=range]').forEach(inp=>{const i=+inp.dataset.idx,lim=jointLimit(i);inp.min=String(lim[0]);inp.max=String(lim[1]);q[i]=clampToJointLimit(i,q[i]||0);inp.value=q[i];const out=inp.parentElement.querySelector('output');out.value=(q[i]||0).toFixed(2)})}
function syncEefSliders(){document.querySelectorAll('#eefSliders input[type=range]').forEach(inp=>{const i=+inp.dataset.idx,k=i%7;if(k===6){inp.min='0';inp.max='1';inp.step='0.01'}else if(k>=3&&k<=5){const c=initEef[i]||0;inp.min=(c-1.57).toFixed(3);inp.max=(c+1.57).toFixed(3);inp.step='0.01'}else{const c=initEef[i]||0;inp.min=(c-0.25).toFixed(3);inp.max=(c+0.25).toFixed(3);inp.step='0.005'}inp.value=eef[i]||0;const out=inp.parentElement.querySelector('output');out.value=eefValueText(i,eef[i])})}
function setControlMode(mode){controlMode=mode;slidersEl.style.display=mode==='joint'?'grid':'none';eefSlidersEl.style.display=mode==='eef'?'grid':'none';modeJoint.classList.toggle('active',mode==='joint');modeEef.classList.toggle('active',mode==='eef')}
modeJoint.onclick=()=>setControlMode('joint');modeEef.onclick=()=>setControlMode('eef');
document.querySelectorAll('.robotCanvas').forEach(initRobotCanvas);
document.querySelectorAll('.mediaStack').forEach(initMediaStack);
function setGripperPair(value){q[6]=clampToJointLimit(6,value);q[13]=clampToJointLimit(13,value);syncSliders();drawRobotCanvas(rctx,q,base,relEl);updateTelemetry('',q,[base,q]);sendState()}
btnOpenGrip.onclick=()=>setGripperPair(1);btnCloseGrip.onclick=()=>setGripperPair(0);
window.addEventListener('resize',()=>requestAnimationFrame(()=>{resizeRobotCanvases();clampMediaStacks();redrawTelemetry()}));
requestAnimationFrame(()=>{resizeRobotCanvases();clampMediaStacks();redrawTelemetry()});
function setJointState(d){if(d.base)base=d.base.slice();if(d.state)q=d.state.slice();if(d.initial)initQ=d.initial.slice();else if(d.type==='init'&&d.base)initQ=d.base.slice();if(d.eef_euler_state)eef=d.eef_euler_state.slice();else if(d.eef_state){const v=eefStateToEuler(d.eef_state);if(v)eef=v}if(d.eef_euler_initial)initEef=d.eef_euler_initial.slice();else if(d.eef_initial){const v=eefStateToEuler(d.eef_initial);if(v)initEef=v}else if(d.type==='init'&&d.eef_state){const v=eefStateToEuler(d.eef_state);if(v)initEef=v}if(d.joint_names)names=d.joint_names.slice();if(d.eef_names)eefNames=d.eef_names.slice();if(!slidersEl.children.length)makeSliders();if(!eefSlidersEl.children.length)makeEefSliders();syncSliders();syncEefSliders();drawRobotCanvas(rctx,q,base,relEl);updateTelemetry('',q,d.chart_states||[base,q]);if(typeof d.steps==='number')steps=d.steps;recording=!!d.recording;btnRec.disabled=recording;btnStop.disabled=!recording;statusEl.textContent=(recording?`[REC ${steps} states]`:'ready');}
const ws=new WebSocket(wsUrl('/ws'));
ws.onopen=()=>statusEl.textContent='Connected';ws.onerror=()=>statusEl.textContent='WebSocket error';ws.onclose=e=>statusEl.textContent='Disconnected '+e.code+(e.reason?' '+e.reason:'');
ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.sid){sid=d.sid;updateGifLinks()}if(d.type==='init'){configurePlayback(kbLoop,'',0,8,()=>{});drawImg(gtctx,d.video,320,384);drawImg(vctx,d.video,320,384);promptEl.textContent=d.prompt||'';setJointState(d);btnSampleInit.disabled=false;btnReset.disabled=false;btnInfer.disabled=false;dl.style.opacity='.4';dl.style.pointerEvents='none'}else if(d.type==='joint_state'){setJointState(d)}else if(d.type==='sampling'){statusEl.textContent='[sampling train init...]';btnSampleInit.disabled=true;btnReset.disabled=true;btnInfer.disabled=true}else if(d.type==='inferring'){statusEl.textContent='[inferring...]';btnInfer.disabled=true}else if(d.type==='frames'){if(d.prompt)promptEl.textContent=d.prompt;const states=d.cond_states||d.cond_values||[],bases=d.cond_bases||[],eefs=d.cond_eef_euler_states||d.cond_eef_states||[];playInteractiveReplay(kbLoop,vctx,320,384,d.frames||[],states,bases,eefs,d.fps||8);statusEl.textContent=`looping ${(d.frames||[]).length} frames`;btnInfer.disabled=false;dl.style.opacity='1';dl.style.pointerEvents='auto';if(!states.length)setJointState(d)}else if(d.type==='error'){statusEl.textContent=d.message||'Error';btnSampleInit.disabled=false;btnReset.disabled=false;btnInfer.disabled=false}};
btnSampleInit.onclick=()=>{if(ws.readyState===1)ws.send(JSON.stringify({action:'sample_init'}))};btnReset.onclick=()=>{if(ws.readyState===1)ws.send(JSON.stringify({action:'reset'}))};btnRec.onclick=()=>{if(ws.readyState===1)ws.send(JSON.stringify({action:'record_start'}))};btnStop.onclick=()=>{if(ws.readyState===1)ws.send(JSON.stringify({action:'record_stop'}))};btnInfer.onclick=()=>{btnInfer.disabled=true;if(ws.readyState===1)ws.send(JSON.stringify({action:'infer'}));else btnInfer.disabled=false};

// RoboTwin sampler
const rwSt=document.getElementById('rwSt'), rwInfEl=document.getElementById('rwInf'), rwBI=document.getElementById('rwBI'), rwBS=document.getElementById('rwBS'), rwDl=document.getElementById('rwDl');
const gRG=document.getElementById('cvRG').getContext('2d'), gRT=document.getElementById('cvRT').getContext('2d'), gRC=document.getElementById('cvRC').getContext('2d');let rwLoop={tmr:null}, rwGt=[], rwStates=[], rwBases=[];
const wsRW=new WebSocket(wsUrl('/ws/robotwin'));
wsRW.onmessage=e=>{const d=JSON.parse(e.data);if(d.sid){rwSid=d.sid;updateGifLinks()}if(d.type==='session'){return}else if(d.type==='episode'){rwSt.textContent='Ep '+d.ep+' - click Run Inference';rwInfEl.textContent=(d.prompt||'').substring(0,160);rwGt=d.gt_frames||[];rwStates=d.cond_states||[];rwBases=d.cond_bases||[];if(rwGt.length)drawImg(gRT,rwGt[0],320,384);if(rwStates.length)drawRobotCanvas(gRC,rwStates[0],rwBases[0]||rwStates[0]);updateTelemetry('rw',rwStates[0],rwStates);rwBI.disabled=false;rwBS.disabled=false;rwDl.style.opacity='.4';rwDl.style.pointerEvents='none';playRobotLoop(rwLoop,gRT,320,384,rwGt,gRC,rwStates,rwBases,8,'rw')}else if(d.type==='inferring'){rwSt.textContent='[Inferring...]';rwBI.disabled=true}else if(d.type==='result'){playTwoImageRobotLoop(rwLoop,gRG,320,384,d.generated||[],gRT,320,384,rwGt,gRC,d.cond_states||rwStates,d.cond_bases||rwBases,d.fps||8,'rw');rwSt.textContent='Done - '+(d.generated||[]).length+' frames';rwBI.disabled=false;rwDl.style.opacity='1';rwDl.style.pointerEvents='auto'}else if(d.type==='error'){rwSt.textContent=d.message||'Error';rwBI.disabled=false;rwBS.disabled=false}};
wsRW.onerror=()=>rwSt.textContent='WebSocket error';wsRW.onclose=e=>rwSt.textContent='Disconnected '+e.code+(e.reason?' '+e.reason:'');
function rwSample(){rwBS.disabled=true;rwSt.textContent='Sampling...';wsRW.send(JSON.stringify({action:'sample'}))}function rwInfer(){wsRW.send(JSON.stringify({action:'infer'}))}

// WorldArena sampler
const waSt=document.getElementById('waSt'), waInfEl=document.getElementById('waInf'), waBI=document.getElementById('waBI'), waBS=document.getElementById('waBS'), waDl=document.getElementById('waDl'), waBE=document.getElementById('waBE'), waDataset=document.getElementById('waDataset');
const waProg=document.getElementById('waProg'), waBatchInfo=document.getElementById('waBatchInfo'), waOptGt=document.getElementById('waOptGt'), waOptGen=document.getElementById('waOptGen'), waOptCond=document.getElementById('waOptCond'), waOptCrop=document.getElementById('waOptCrop'), waOptMerge=document.getElementById('waOptMerge');
const waGT=document.getElementById('cvWT').getContext('2d'), waGG=document.getElementById('cvWG').getContext('2d'), waGC=document.getElementById('cvWC').getContext('2d');let waLoop={tmr:null}, waBatchTimer=null, waGt=[], waStates=[], waBases=[];
function waSetBatch(d){const done=d.completed||0,total=d.total||0;const pct=total?Math.min(100,Math.round(done*100/total)):0;waProg.style.width=pct+'%';waBatchInfo.textContent=`${d.dataset||waDataset.value} ${d.status||'idle'} ${done}/${total} (${pct}%)`+(d.gpus?` GPUs=${d.gpus}`:'')+(d.output_root?` output=${d.output_root}`:'')+(d.merged_root?` merged=${d.merged_root}`:'')+(d.error?` error=${d.error}`:'');waBE.disabled=true;if(d.status==='running'||d.status==='starting'||d.status==='merging'){if(!waBatchTimer)waBatchTimer=setInterval(()=>{if(wsWA.readyState===1)wsWA.send(JSON.stringify({action:'batch_status'}));},2000)}else if(waBatchTimer){clearInterval(waBatchTimer);waBatchTimer=null}}
const wsWA=new WebSocket(wsUrl('/ws/worldarena'));
wsWA.onmessage=e=>{const d=JSON.parse(e.data);if(d.sid){waSid=d.sid;updateGifLinks()}if(d.type==='session'){return}else if(d.type==='episode'){waSt.textContent=(d.dataset||waDataset.value)+' Ep '+d.ep+' - click Run Inference';waInfEl.textContent=(d.prompt||'').substring(0,160);waGt=d.gt_frames||[];waStates=d.cond_states||[];waBases=d.cond_bases||[];if(d.first_frame)drawImg(waGG,d.first_frame,320,384);if(waGt.length)drawImg(waGT,waGt[0],320,384);if(waStates.length)drawRobotCanvas(waGC,waStates[0],waBases[0]||waStates[0]);updateTelemetry('wa',waStates[0],waStates);waBI.disabled=false;waBS.disabled=false;waDl.style.opacity='.4';waDl.style.pointerEvents='none';playRobotLoop(waLoop,waGT,320,384,waGt,waGC,waStates,waBases,8,'wa')}else if(d.type==='inferring'){waSt.textContent='[Inferring...]';waBI.disabled=true}else if(d.type==='result'){if(d.gt_frames&&d.gt_frames.length)waGt=d.gt_frames;playTwoImageRobotLoop(waLoop,waGG,320,384,d.generated||[],waGT,320,384,waGt,waGC,d.cond_states||waStates,d.cond_bases||waBases,d.fps||8,'wa');waSt.textContent='Done - '+(d.generated||[]).length+' frames';waBI.disabled=false;waDl.style.opacity='1';waDl.style.pointerEvents='auto'}else if(d.type==='batch_status'){waSetBatch(d)}else if(d.type==='error'){waSt.textContent=d.message||'Error';waBI.disabled=false;waBS.disabled=false}};
wsWA.onerror=()=>waSt.textContent='WebSocket error';wsWA.onclose=e=>waSt.textContent='Disconnected '+e.code+(e.reason?' '+e.reason:'');
function waDatasetChanged(){waBatchInfo.textContent='Dataset '+waDataset.value+' selected';waProg.style.width='0%';waBI.disabled=true;waDl.style.opacity='.4';waDl.style.pointerEvents='none'}
function waSample(){waBS.disabled=true;waSt.textContent='Sampling...';wsWA.send(JSON.stringify({action:'sample',dataset:waDataset.value}))}function waInfer(){wsWA.send(JSON.stringify({action:'infer'}))}
function waBatchEval(){waBE.disabled=true;waBatchInfo.textContent='Full eval disabled'}
</script></body></html>"""


def _html() -> str:
    return HTML_TEMPLATE.replace("__ALOHA_KINEMATICS_JSON__", json.dumps(_aloha_kinematics_payload(), separators=(",", ":")))


def _mat_id_py() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _mat_translate_py(v: Any) -> np.ndarray:
    m = _mat_id_py()
    m[:3, 3] = np.asarray(v, dtype=np.float64).reshape(3)
    return m


def _mat_rot_x_py(a: float) -> np.ndarray:
    c, ss = np.cos(float(a)), np.sin(float(a))
    m = _mat_id_py()
    m[:3, :3] = [[1, 0, 0], [0, c, -ss], [0, ss, c]]
    return m


def _mat_rot_y_py(a: float) -> np.ndarray:
    c, ss = np.cos(float(a)), np.sin(float(a))
    m = _mat_id_py()
    m[:3, :3] = [[c, 0, ss], [0, 1, 0], [-ss, 0, c]]
    return m


def _mat_rot_z_py(a: float) -> np.ndarray:
    c, ss = np.cos(float(a)), np.sin(float(a))
    m = _mat_id_py()
    m[:3, :3] = [[c, -ss, 0], [ss, c, 0], [0, 0, 1]]
    return m


def _mat_axis_py(axis: Any, angle: float) -> np.ndarray:
    axis_np = np.asarray(axis if axis is not None else [0, 0, 1], dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_np))
    if norm < 1e-9:
        axis_np = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis_np = axis_np / norm
    x, y, z = axis_np.tolist()
    c, ss, t = np.cos(float(angle)), np.sin(float(angle)), 1.0 - np.cos(float(angle))
    m = _mat_id_py()
    m[:3, :3] = [
        [t * x * x + c, t * x * y - ss * z, t * x * z + ss * y],
        [t * x * y + ss * z, t * y * y + c, t * y * z - ss * x],
        [t * x * z - ss * y, t * y * z + ss * x, t * z * z + c],
    ]
    return m


def _origin_mat_py(origin: dict[str, Any] | None) -> np.ndarray:
    origin = origin or {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
    rpy = origin.get("rpy") or [0, 0, 0]
    return _mat_translate_py(origin.get("xyz") or [0, 0, 0]) @ _mat_rot_z_py(rpy[2] if len(rpy) > 2 else 0) @ _mat_rot_y_py(rpy[1] if len(rpy) > 1 else 0) @ _mat_rot_x_py(rpy[0] if len(rpy) > 0 else 0)


def _joint_mat_py(joint: dict[str, Any], value: float = 0.0) -> np.ndarray:
    m = _origin_mat_py(joint.get("origin"))
    typ = joint.get("type", "fixed")
    if typ in {"revolute", "continuous"}:
        m = m @ _mat_axis_py(joint.get("axis"), float(value))
    elif typ == "prismatic":
        axis = np.asarray(joint.get("axis") or [0, 0, 1], dtype=np.float64).reshape(3)
        m = m @ _mat_translate_py(axis * float(value))
    return m


def _arm_joint_bounds(side: str) -> tuple[np.ndarray, np.ndarray]:
    arm = _aloha_kinematics_payload()["arms"][side]
    lows, highs = [], []
    for joint in arm["joints"][:6]:
        limit = joint.get("limit") or {}
        lo, hi = float(limit.get("lower", -3.14)), float(limit.get("upper", 3.14))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = -3.14, 3.14
        lows.append(max(lo, -3.14))
        highs.append(min(hi, 3.14))
    return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)


def _fk_arm_pose(side: str, vals: Any) -> tuple[np.ndarray, np.ndarray]:
    arm = _aloha_kinematics_payload()["arms"][side]
    vals_np = np.asarray(vals, dtype=np.float64).reshape(-1)
    T = _mat_id_py() @ _joint_mat_py(arm["base_joint"], 0.0)
    for i, joint in enumerate(arm["joints"][:6]):
        T = T @ _joint_mat_py(joint, float(vals_np[i]) if i < len(vals_np) else 0.0)
    return T[:3, 3].copy(), T[:3, :3].copy()


def _normalize_quat_xyzw(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-9 or not np.isfinite(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _rot_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat().astype(np.float32)


def _eef_quat_to_euler_state(eef_quat: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    e = np.asarray(eef_quat, dtype=np.float32).reshape(16)
    out = np.zeros(14, dtype=np.float32)
    for eoff, ooff in ((0, 0), (8, 7)):
        out[ooff:ooff + 3] = e[eoff:eoff + 3]
        try:
            out[ooff + 3:ooff + 6] = Rotation.from_quat(_normalize_quat_xyzw(e[eoff + 3:eoff + 7])).as_euler("xyz", degrees=False).astype(np.float32)
        except Exception:
            out[ooff + 3:ooff + 6] = 0.0
        out[ooff + 6] = float(np.clip(e[eoff + 7], 0.0, 1.0))
    return out


def _eef_euler_to_quat_state(eef_euler: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    e = np.asarray(eef_euler, dtype=np.float32).reshape(14)
    out = np.zeros(16, dtype=np.float32)
    for ooff, eoff in ((0, 0), (7, 8)):
        out[eoff:eoff + 3] = e[ooff:ooff + 3]
        try:
            out[eoff + 3:eoff + 7] = Rotation.from_euler("xyz", e[ooff + 3:ooff + 6], degrees=False).as_quat().astype(np.float32)
        except Exception:
            out[eoff + 3:eoff + 7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        out[eoff + 7] = float(np.clip(e[ooff + 6], 0.0, 1.0))
    return out


def _default_eef_from_joint(qpos: Any) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float32).reshape(14)
    out = np.zeros(16, dtype=np.float32)
    for side, qoff, eoff, grip_idx in (("left", 0, 0, 6), ("right", 7, 8, 13)):
        pos, rot = _fk_arm_pose(side, q[qoff:qoff + 6])
        out[eoff:eoff + 3] = pos.astype(np.float32)
        out[eoff + 3:eoff + 7] = _rot_to_quat_xyzw(rot)
        out[eoff + 7] = float(np.clip(q[grip_idx], 0.0, 1.0))
    return out


def _eef_from_joint_delta(qpos: Any, initial_qpos: Any, initial_eef: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(qpos, dtype=np.float32).reshape(14)
    q0 = np.asarray(initial_qpos, dtype=np.float32).reshape(14)
    e0 = np.asarray(initial_eef, dtype=np.float32).reshape(16)
    out = e0.copy()
    for side, qoff, eoff, grip_idx in (("left", 0, 0, 6), ("right", 7, 8, 13)):
        pos0, rot0 = _fk_arm_pose(side, q0[qoff:qoff + 6])
        pos, rot = _fk_arm_pose(side, q[qoff:qoff + 6])
        out[eoff:eoff + 3] = e0[eoff:eoff + 3] + (pos - pos0).astype(np.float32)
        try:
            delta = Rotation.from_matrix(rot) * Rotation.from_matrix(rot0).inv()
            out[eoff + 3:eoff + 7] = (delta * Rotation.from_quat(_normalize_quat_xyzw(e0[eoff + 3:eoff + 7]))).as_quat().astype(np.float32)
        except Exception:
            pass
        out[eoff + 7] = float(np.clip(q[grip_idx], 0.0, 1.0))
    return out.astype(np.float32)


def _solve_joint_from_eef(initial_qpos: Any, current_qpos: Any, initial_eef: Any, target_eef: Any) -> np.ndarray:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    q0 = np.asarray(initial_qpos, dtype=np.float32).reshape(14)
    q = np.asarray(current_qpos, dtype=np.float32).reshape(14).copy()
    e0 = np.asarray(initial_eef, dtype=np.float32).reshape(16)
    et = np.asarray(target_eef, dtype=np.float32).reshape(16).copy()
    et[3:7] = _normalize_quat_xyzw(et[3:7]).astype(np.float32)
    et[11:15] = _normalize_quat_xyzw(et[11:15]).astype(np.float32)

    for side, qoff, eoff, grip_idx in (("left", 0, 0, 6), ("right", 7, 8, 13)):
        init_pos, init_rot = _fk_arm_pose(side, q0[qoff:qoff + 6])
        target_pos = init_pos + (et[eoff:eoff + 3] - e0[eoff:eoff + 3]).astype(np.float64)
        try:
            delta_train = Rotation.from_quat(_normalize_quat_xyzw(et[eoff + 3:eoff + 7])) * Rotation.from_quat(_normalize_quat_xyzw(e0[eoff + 3:eoff + 7])).inv()
            target_rot = delta_train * Rotation.from_matrix(init_rot)
        except Exception:
            target_rot = Rotation.from_matrix(init_rot)
        start = q[qoff:qoff + 6].astype(np.float64)
        lo, hi = _arm_joint_bounds(side)
        start = np.clip(start, lo, hi)

        def residual(x: np.ndarray) -> np.ndarray:
            pos, rot = _fk_arm_pose(side, x)
            pos_res = (pos - target_pos) * 9.0
            rot_res = (target_rot.inv() * Rotation.from_matrix(rot)).as_rotvec() * 0.65
            reg_res = (x - start) * 0.025
            return np.concatenate([pos_res, rot_res, reg_res])

        result = least_squares(residual, start, bounds=(lo, hi), max_nfev=90, xtol=1e-4, ftol=1e-4, gtol=1e-4)
        q[qoff:qoff + 6] = result.x.astype(np.float32)
        q[grip_idx] = float(np.clip(et[eoff + 7], 0.0, 1.0))
    return q.astype(np.float32)


class V5JointDemo:
    def __init__(
        self,
        model,
        processor,
        init_image: torch.Tensor,
        prompt: str,
        init_state: np.ndarray,
        num_steps: int,
        seed: int,
        sample_label: str = "",
        init_eef_state: np.ndarray | None = None,
    ):
        self.model = model
        self.processor = processor
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.sample_label = ""
        self.last_frames: list[Image.Image] = []
        self.last_cond_states: list[list[float]] = []
        self.last_cond_bases: list[list[float]] = []
        self.set_initial(init_image, prompt, init_state, sample_label=sample_label, init_eef_state=init_eef_state)

    def use_worker(self, worker: ModelWorker) -> None:
        _activate_worker(worker)
        self.model = worker.model
        self.processor = worker.processor
        self.init_image = self.init_image.to(device=self.model.device, dtype=self.model.torch_dtype)
        self.image = self.image.to(device=self.model.device, dtype=self.model.torch_dtype)

    def set_initial(
        self,
        init_image: torch.Tensor,
        prompt: str,
        init_state: np.ndarray,
        sample_label: str = "",
        init_eef_state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self.init_image = init_image
        self.image = init_image
        self.prompt = prompt
        self.sample_label = sample_label
        self.initial_state = np.asarray(init_state, dtype=np.float32).reshape(14)
        if init_eef_state is None:
            init_eef_state = _default_eef_from_joint(self.initial_state)
        self.initial_eef_state = np.asarray(init_eef_state, dtype=np.float32).reshape(16)
        self.initial_eef_state[3:7] = _normalize_quat_xyzw(self.initial_eef_state[3:7]).astype(np.float32)
        self.initial_eef_state[11:15] = _normalize_quat_xyzw(self.initial_eef_state[11:15]).astype(np.float32)
        self.base_state = self.initial_state.copy()
        self.target_state = self.initial_state.copy()
        self.target_eef_state = self.initial_eef_state.copy()
        self.sequence: list[np.ndarray] = []
        self.recording = False
        self.last_frames = []
        self.last_cond_states = []
        self.last_cond_bases = []
        return self.payload()

    def payload(self) -> dict[str, Any]:
        return {
            **_joint_payload(self.base_state, self.target_state),
            "initial": self.initial_state.round(5).tolist(),
            "eef_state": self.target_eef_state.round(5).tolist(),
            "eef_initial": self.initial_eef_state.round(5).tolist(),
            "eef_euler_state": _eef_quat_to_euler_state(self.target_eef_state).round(5).tolist(),
            "eef_euler_initial": _eef_quat_to_euler_state(self.initial_eef_state).round(5).tolist(),
            "eef_names": EEF_NAMES,
            "recording": self.recording,
            "steps": len(self.sequence),
            "sample_label": self.sample_label,
        }

    def reset(self) -> dict[str, Any]:
        self.image = self.init_image
        self.base_state = self.initial_state.copy()
        self.target_state = self.initial_state.copy()
        self.target_eef_state = self.initial_eef_state.copy()
        self.sequence.clear()
        self.recording = False
        return self.payload()

    def set_joint(self, state: Any) -> dict[str, Any]:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 14:
            raise ValueError(f"Expected 14 joint values, got {arr.shape[0]}")
        self.target_state = arr
        self.target_eef_state = _eef_from_joint_delta(self.target_state, self.initial_state, self.initial_eef_state)
        if self.recording:
            self.sequence.append(arr.copy())
        return self.payload()

    def set_eef(self, state: Any) -> dict[str, Any]:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 14:
            eef = _eef_euler_to_quat_state(arr)
        elif arr.shape[0] == 16:
            eef = arr.copy()
        else:
            raise ValueError(f"Expected 14 Euler EEF values or 16 quaternion EEF values, got {arr.shape[0]}")
        eef[3:7] = _normalize_quat_xyzw(eef[3:7]).astype(np.float32)
        eef[11:15] = _normalize_quat_xyzw(eef[11:15]).astype(np.float32)
        self.target_state = _solve_joint_from_eef(self.initial_state, self.target_state, self.initial_eef_state, eef)
        self.target_eef_state = eef.copy()
        if self.recording:
            self.sequence.append(self.target_state.copy())
        return self.payload()

    def record_start(self) -> dict[str, Any]:
        self.sequence.clear()
        self.recording = True
        self.sequence.append(self.target_state.copy())
        return self.payload()

    def record_stop(self) -> dict[str, Any]:
        self.recording = False
        return self.payload()

    def infer(self) -> tuple[list[Image.Image], dict[str, Any]]:
        seq = [s.copy() for s in self.sequence] or [self.target_state.copy() for _ in range(ACTION_STEPS_PER_ROLLOUT)]
        action_abs = np.stack(seq, axis=0).astype(np.float32)
        prev = np.concatenate([self.base_state[None], action_abs[:-1]], axis=0).astype(np.float32)
        frames, _ = run_episode(
            model=self.model,
            processor=self.processor,
            action_np=action_abs,
            prompt=self.prompt,
            init_image_tensor=self.image,
            num_inference_steps=self.num_steps,
            seed=self.seed,
            max_rollouts=None,
            state_np=prev,
        )
        self.last_frames = frames
        cond = _qpos_to_condition_payload(action_abs, prev, n=len(frames) or len(action_abs))
        self.last_cond_states = cond.get("cond_states", [])
        self.last_cond_bases = cond.get("cond_bases", [])
        cond["cond_eef_states"] = [_eef_from_joint_delta(row, self.initial_state, self.initial_eef_state).round(5).tolist() for row in action_abs[np.linspace(0, len(action_abs) - 1, len(self.last_cond_states)).astype(int)]] if self.last_cond_states else []
        cond["cond_eef_euler_states"] = [_eef_quat_to_euler_state(row).round(5).tolist() for row in cond["cond_eef_states"]]
        if frames:
            self.image = pil_to_tensor(frames[-1], self.model.device, self.model.torch_dtype)
        self.base_state = action_abs[-1].copy()
        self.target_state = self.base_state.copy()
        self.target_eef_state = _eef_from_joint_delta(self.target_state, self.initial_state, self.initial_eef_state)
        self.sequence.clear()
        self.recording = False
        return frames, {**self.payload(), **cond}


# -- sampler helpers -----------------------------------------------------------
_WEB_ARGS: argparse.Namespace | None = None
_WEB_POOL: ModelWorkerPool | None = None
_WEB_BOOT_SAMPLE: dict[str, Any] | None = None
_SESSION_OUTPUTS: dict[str, dict[str, dict[str, Any]]] = {}
_PROXY_WORKERS: list[dict[str, Any]] = []
_PROXY_SID_TO_WORKER: dict[str, dict[str, Any]] = {}
_PROXY_LOCK = threading.Lock()
_WA_BATCH_STATE: dict[str, Any] = {"status": "idle", "completed": 0, "total": 0}
_WA_BATCH_LOCK = threading.Lock()


def _resize_head_to_320x240(head: np.ndarray) -> np.ndarray:
    head = np.asarray(head, dtype=np.uint8)
    if head.shape[:2] == (240, 320):
        return head
    return np.asarray(Image.fromarray(head).resize((320, 240), Image.BILINEAR), dtype=np.uint8)


def _build_3cam_pil(head: np.ndarray, left: np.ndarray, right: np.ndarray, normalize_head_320x240: bool = False) -> Image.Image:
    if normalize_head_320x240:
        head = _resize_head_to_320x240(head)
    head = np.asarray(Image.fromarray(np.asarray(head, dtype=np.uint8)).resize((320, 256), Image.BILINEAR))
    left = np.asarray(Image.fromarray(np.asarray(left, dtype=np.uint8)).resize((160, 128), Image.BILINEAR))
    right = np.asarray(Image.fromarray(np.asarray(right, dtype=np.uint8)).resize((160, 128), Image.BILINEAR))
    return Image.fromarray(np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0))


def _train_video_to_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if _WEB_ARGS is not None and bool(getattr(_WEB_ARGS, "train_video_bgr", False)):
        return arr[..., ::-1].copy()
    return arr


def _train_video_to_gt_display_rgb(frame: np.ndarray) -> np.ndarray:
    return _train_video_to_rgb(frame)


def _read_3cam_pils(p0: str, p1: str, p2: str, n: int) -> tuple[list[Image.Image], int]:
    import imageio.v3 as iio
    fH = iio.imread(p0, plugin="pyav")
    fL = iio.imread(p1, plugin="pyav")
    fR = iio.imread(p2, plugin="pyav")
    T = min(len(fH), len(fL), len(fR))
    idx = np.linspace(0, T - 1, int(n)).astype(int)
    return [
        _build_3cam_pil(
            _train_video_to_gt_display_rgb(fH[i]),
            _train_video_to_gt_display_rgb(fL[i]),
            _train_video_to_gt_display_rgb(fR[i]),
        )
        for i in idx
    ], T


def _main_view_frame_to_3cam_canvas(frame: np.ndarray) -> Image.Image:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    return _build_3cam_pil(arr.astype(np.uint8, copy=False), black, black, normalize_head_320x240=True)


def _read_worldarena_gt_video_path(path: Path, n: int | None = None) -> list[Image.Image]:
    if not path.exists():
        return []
    import imageio.v3 as iio
    frames = iio.imread(str(path), plugin="pyav")
    if frames is None or len(frames) == 0:
        return []
    idx = np.linspace(0, len(frames) - 1, int(n)).astype(int) if n else np.arange(len(frames))
    return [_main_view_frame_to_3cam_canvas(frames[int(i)]) for i in idx]


def _read_worldarena_gt_frames(data_root: Path, ep: int, n: int | None = None) -> tuple[list[Image.Image], Path | None]:
    candidates = []
    gt_dirs = (_WEB_ARGS.worldarena_gt_video_dir or "").split(os.pathsep) if _WEB_ARGS else []
    candidates.extend(Path(p) for p in gt_dirs if p)
    candidates.append(data_root / "real_gt_videos")
    for root in candidates:
        path = root / f"episode{int(ep)}.mp4"
        if path.exists():
            return _read_worldarena_gt_video_path(path, n=n), path
    return [], None


def _qpos_to_condition_payload(action_abs: np.ndarray, state_np: np.ndarray | None = None, n: int = 33) -> dict[str, Any]:
    action = _as_2d("action_abs", action_abs)
    if len(action) == 0:
        states = np.zeros((1, 14), dtype=np.float32)
    else:
        idx = np.linspace(0, len(action) - 1, int(n)).astype(int)
        states = action[idx].astype(np.float32)
    if state_np is not None and len(state_np):
        base0 = _as_2d("state", state_np)[0]
    else:
        base0 = states[0]
    bases = np.repeat(base0.reshape(1, -1), len(states), axis=0).astype(np.float32)
    return {
        "cond_states": states.round(5).tolist(),
        "cond_bases": bases.round(5).tolist(),
        "cond_values": states.round(5).tolist(),
        "cond_mode": "absolute",
    }


def _draw_joint_condition_pil(state: np.ndarray, base: np.ndarray | None = None, size: tuple[int, int] = (640, 384)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (28, 28, 28))
    draw = ImageDraw.Draw(img)
    state = np.asarray(state, dtype=np.float32).reshape(14)
    base = state if base is None else np.asarray(base, dtype=np.float32).reshape(14)
    yaw, pitch, zoom = 0.0, 0.22, min(w, h) * 0.72
    target = np.array([0.08, 0.0, 0.48], dtype=np.float64)
    cp, sp = np.cos(pitch), np.sin(pitch)
    direction = np.array([cp * np.cos(yaw), cp * np.sin(yaw), sp], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    right = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.array([0.0, 1.0, 0.0])
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(direction, right)
    up /= max(float(np.linalg.norm(up)), 1e-9)

    def project(point: Any) -> tuple[float, float, float]:
        pnt = np.asarray(point, dtype=np.float64).reshape(3) - target
        return (
            w * 0.5 + float(np.dot(pnt, right)) * zoom,
            h * 0.56 - float(np.dot(pnt, up)) * zoom,
            float(np.dot(pnt, direction)),
        )

    def line3(a: Any, b: Any, color: tuple[int, int, int], width: int = 3):
        ax, ay, _ = project(a)
        bx, by, _ = project(b)
        draw.line((ax, ay, bx, by), fill=color, width=width)

    def dot3(a: Any, color: tuple[int, int, int], r: int = 4):
        x, y, _ = project(a)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    for gx in np.arange(-0.45, 0.66, 0.10):
        line3([gx, -0.55, 0.0], [gx, 0.55, 0.0], (48, 48, 48), 1)
    for gy in np.arange(-0.55, 0.56, 0.10):
        line3([-0.45, gy, 0.0], [0.65, gy, 0.0], (48, 48, 48), 1)
    line3([0, 0, 0], [0.18, 0, 0], (150, 80, 80), 2)
    line3([0, 0, 0], [0, 0.18, 0], (80, 150, 80), 2)
    line3([0, 0, 0], [0, 0, 0.18], (80, 80, 150), 2)

    def arm_points(side: str, vals: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        arm = _aloha_kinematics_payload()["arms"][side]
        T = _mat_id_py() @ _joint_mat_py(arm["base_joint"], 0.0)
        pts = [T[:3, 3].copy()]
        for i, joint in enumerate(arm["joints"][:6]):
            T = T @ _joint_mat_py(joint, float(vals[i]))
            pts.append(T[:3, 3].copy())
        grip = float(np.clip(vals[6], 0.0, 1.0)) * float(_aloha_kinematics_payload().get("gripper_upper", 0.04765))
        fingers = []
        for joint in arm["joints"][6:8]:
            FT = T @ _joint_mat_py(joint, grip)
            fingers.append(FT[:3, 3].copy())
        return pts, fingers

    def draw_arm(side: str, vals: np.ndarray, color: tuple[int, int, int], width: int):
        pts, fingers = arm_points(side, vals)
        for a, b in zip(pts, pts[1:]):
            line3(a, b, color, width)
        for f in fingers:
            line3(pts[-1], f, color, max(2, width - 2))
        for idx, point in enumerate(pts):
            dot3(point, (235, 235, 235) if idx == len(pts) - 1 else color, 3 if width <= 4 else 4)

    draw_arm("left", base[:7], (105, 105, 105), 4)
    draw_arm("right", base[7:], (105, 105, 105), 4)
    draw_arm("left", state[:7], (255, 51, 51), 6)
    draw_arm("right", state[7:], (51, 221, 85), 6)
    return img


def _joint_condition_frames(states: list | np.ndarray, bases: list | np.ndarray | None = None) -> list[Image.Image]:
    states_np = np.asarray(states, dtype=np.float32)
    if states_np.ndim != 2 or states_np.shape[-1] != 14:
        return []
    bases_np = np.asarray(bases, dtype=np.float32) if bases is not None else states_np
    if bases_np.ndim != 2 or bases_np.shape[-1] != 14:
        bases_np = states_np
    return [_draw_joint_condition_pil(states_np[i], bases_np[min(i, len(bases_np) - 1)]) for i in range(len(states_np))]


def _hconcat_pils(images: list[Image.Image], height: int = 384) -> Image.Image:
    arrays = []
    for img in images:
        pil = img.convert("RGB")
        if pil.height != height:
            width = max(1, int(round(pil.width * height / max(pil.height, 1))))
            pil = pil.resize((width, height), Image.BILINEAR)
        arrays.append(np.asarray(pil))
    return Image.fromarray(np.concatenate(arrays, axis=1))


def _gif_response(series: list[list[Image.Image]], filename: str):
    from fastapi.responses import Response
    series = [s for s in series if s]
    if not series:
        return Response(status_code=404, content=b"No video yet")
    n = max(len(s) for s in series)
    frames = []
    for i in range(n):
        frames.append(_hconcat_pils([s[min(i, len(s) - 1)] for s in series]))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=125)
    return Response(content=buf.getvalue(), media_type="image/gif", headers={"Content-Disposition": f"attachment; filename={filename}"})


def _episode_ids_from_worldarena(data_root: Path) -> list[int]:
    head_root = data_root / "first_frame/fixed_scene_task"
    ids = []
    for p in head_root.glob("episode*.png"):
        stem = p.stem.replace("episode", "")
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def _worldarena_dataset_key(raw: str | None = None) -> str:
    key = str(raw or "eval750").strip().lower()
    return {"eval": "eval750", "750": "eval750", "test": "test1000", "1000": "test1000"}.get(key, key)


def _worldarena_root_for(key: str | None = None) -> Path:
    if _WEB_ARGS is None:
        raise RuntimeError("Web args are not initialized.")
    key = _worldarena_dataset_key(key)
    root = Path(_WEB_ARGS.worldarena_test_data_root if key == "test1000" else _WEB_ARGS.data_root).expanduser()
    if not root.exists():
        raise RuntimeError(f"WorldArena dataset {key} path does not exist: {root}")
    return root


def _load_worldarena_sample(data_root: Path, ep: int, model) -> tuple[torch.Tensor, str, np.ndarray, np.ndarray, list[Image.Image], Path | None, Image.Image]:
    head = np.asarray(Image.open(data_root / "first_frame/fixed_scene_task" / f"episode{int(ep)}.png").convert("RGB"))
    left = read_first_frame(_WEB_ARGS.left_mp4)
    right = read_first_frame(_WEB_ARGS.right_mp4)
    init_image = build_image_tensor(head, left, right, model.device, model.torch_dtype)
    with h5py.File(data_root / "data/fixed_scene_task" / f"episode{int(ep)}.hdf5", "r") as f:
        qpos = f[_WEB_ARGS.hdf5_action_key][:].astype(np.float32)
    state_np, action_np = _split_worldarena_state_action(qpos)
    prompt = _load_prompt(data_root, ep, _WEB_ARGS.prompt, _WEB_ARGS.instr_root or None)
    gt_frames, gt_path = _read_worldarena_gt_frames(data_root, ep, n=33)
    first_frame = _build_3cam_pil(head, left, right, normalize_head_320x240=True)
    return init_image, prompt, state_np, action_np, gt_frames, gt_path, first_frame


def _joint_from_train_annotation(ann: dict[str, Any]) -> np.ndarray:
    if "action.joint_position" in ann:
        arr = np.asarray(ann["action.joint_position"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[-1] == 14:
            return arr
    joint = np.asarray(ann["observation.state.joint_position"], dtype=np.float32)
    grip = np.asarray(ann["observation.state.gripper_position"], dtype=np.float32)
    if joint.ndim != 2 or joint.shape[-1] < 12 or grip.ndim != 2 or grip.shape[-1] < 2:
        raise ValueError("Training annotation does not contain 12D joint + 2D gripper state.")
    return np.concatenate([joint[:, :6], grip[:, 0:1], joint[:, 6:12], grip[:, 1:2]], axis=1).astype(np.float32)


def _train_prompt_from_annotation(ann: dict[str, Any]) -> str:
    prompt_text = ann.get("texts", [""])[0] if ann.get("texts") else "A robot arm completes a task."
    return DEFAULT_PROMPT.format(task=prompt_text) if not prompt_text.startswith("A video recorded") else prompt_text


def _eef_from_train_annotation(ann: dict[str, Any], qpos_row: np.ndarray, raw_idx: int) -> np.ndarray:
    cart = ann.get("observation.state.cartesian_position")
    if cart is None:
        return _default_eef_from_joint(qpos_row)
    arr = np.asarray(cart, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] < 14:
        return _default_eef_from_joint(qpos_row)
    idx = int(np.clip(raw_idx, 0, len(arr) - 1))
    q = np.asarray(qpos_row, dtype=np.float32).reshape(14)
    eef = np.zeros(16, dtype=np.float32)
    eef[:7] = arr[idx, :7]
    eef[7] = float(np.clip(q[6], 0.0, 1.0))
    eef[8:15] = arr[idx, 7:14]
    eef[15] = float(np.clip(q[13], 0.0, 1.0))
    eef[3:7] = _normalize_quat_xyzw(eef[3:7]).astype(np.float32)
    eef[11:15] = _normalize_quat_xyzw(eef[11:15]).astype(np.float32)
    return eef


def _train_annotation_files(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    train_root = Path(args.train_root).expanduser()
    ann_dir = train_root / "annotation/train"
    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise RuntimeError(f"No training annotations found under {ann_dir}")
    return train_root, ann_files


def _train_camera_paths(train_root: Path, ann: dict[str, Any], ann_path: Path) -> tuple[Path, Path, Path]:
    videos = ann.get("videos") or []
    if len(videos) < 3:
        raise RuntimeError(f"Training annotation {ann_path} does not contain 3 camera videos.")
    cam_left = train_root / videos[0]["video_path"]
    cam_high = train_root / videos[1]["video_path"]
    cam_right = train_root / videos[2]["video_path"]
    return cam_left, cam_high, cam_right


def _video_frame_count(path: Path, fallback: int = 0) -> int:
    try:
        import imageio.v3 as iio
        props = iio.improps(str(path), plugin="pyav")
        if props.shape and len(props.shape) >= 1 and int(props.shape[0]) > 0:
            return int(props.shape[0])
    except Exception:
        pass
    return int(fallback or 0)


def _read_video_frame(path: Path, frame_idx: int) -> np.ndarray:
    import imageio
    reader = imageio.get_reader(str(path))
    try:
        return np.asarray(reader.get_data(int(frame_idx)))
    finally:
        reader.close()


def _raw_index_for_video_frame(video_idx: int, video_len: int, raw_len: int) -> int:
    if raw_len <= 1 or video_len <= 1:
        return 0
    return int(np.clip(round(float(video_idx) * float(raw_len - 1) / float(video_len - 1)), 0, raw_len - 1))


def _sample_train_frame_init(model, args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _WEB_ARGS
    if args is None:
        raise RuntimeError("Web args are not initialized.")
    train_root, ann_files = _train_annotation_files(args)
    ann_path = random.choice(ann_files)
    ann = _load_json(ann_path)
    cam_left, cam_high, cam_right = _train_camera_paths(train_root, ann, ann_path)
    qpos = _joint_from_train_annotation(ann)
    meta_video_len = int(ann.get("video_length") or ann.get("state_length") or 0)
    video_len = min(
        _video_frame_count(cam_left, meta_video_len),
        _video_frame_count(cam_high, meta_video_len),
        _video_frame_count(cam_right, meta_video_len),
    )
    if video_len <= 0:
        raise RuntimeError(f"Could not determine video length for training annotation {ann_path}")
    frame_idx = random.randrange(video_len)
    raw_idx = _raw_index_for_video_frame(frame_idx, video_len, len(qpos))
    head = _train_video_to_rgb(_read_video_frame(cam_high, frame_idx))
    left = _train_video_to_rgb(_read_video_frame(cam_left, frame_idx))
    right = _train_video_to_rgb(_read_video_frame(cam_right, frame_idx))
    init_image = build_image_tensor(head, left, right, model.device, model.torch_dtype)
    prompt = _train_prompt_from_annotation(ann)
    ep = ann.get("episode_id", ann_path.stem)
    label = f"train annotation {ann_path.name} frame {frame_idx}/{video_len - 1} state {raw_idx}/{len(qpos) - 1}"
    return {
        "ep": int(ep) if str(ep).isdigit() else ep,
        "ann_path": str(ann_path),
        "frame_idx": int(frame_idx),
        "video_len": int(video_len),
        "raw_idx": int(raw_idx),
        "raw_len": int(len(qpos)),
        "init_image": init_image,
        "prompt": prompt,
        "state": qpos[raw_idx].astype(np.float32),
        "eef_state": _eef_from_train_annotation(ann, qpos[raw_idx], raw_idx),
        "sample_label": label,
    }


def _do_sample_rw(worker: ModelWorker, state: dict[str, Any]) -> dict[str, Any]:
    if _WEB_ARGS is None:
        raise RuntimeError("Demo is not initialized.")
    _activate_worker(worker)
    train_root, ann_files = _train_annotation_files(_WEB_ARGS)
    ann_path = random.choice(ann_files)
    ann = _load_json(ann_path)
    cam_left, cam_high, cam_right = _train_camera_paths(train_root, ann, ann_path)
    head0 = _train_video_to_rgb(read_first_frame(str(cam_high)))
    left0 = _train_video_to_rgb(read_first_frame(str(cam_left)))
    right0 = _train_video_to_rgb(read_first_frame(str(cam_right)))
    init_image = build_image_tensor(head0, left0, right0, worker.model.device, worker.model.torch_dtype)
    qpos = _joint_from_train_annotation(ann)
    state_np, action_np = _split_worldarena_state_action(qpos)
    prompt = _train_prompt_from_annotation(ann)
    gt_frames, _ = _read_3cam_pils(str(cam_high), str(cam_left), str(cam_right), 33)
    payload = _qpos_to_condition_payload(action_np, state_np, n=33)
    state.clear()
    state.update({
        "ep": int(ann.get("episode_id", ann_path.stem)),
        "init_image": init_image,
        "prompt": prompt,
        "action_np": action_np,
        "state_np": state_np,
        "gt_frames": gt_frames,
        **payload,
    })
    return {
        "ep": state["ep"],
        "prompt": f"{prompt}  [train annotation {ann_path.name}]",
        "gt_frames": [_to_b64(f) for f in gt_frames],
        **payload,
    }


def _do_infer_rw(worker: ModelWorker, state: dict[str, Any], num_steps: int, seed: int) -> dict[str, Any]:
    _activate_worker(worker)
    if not state:
        raise RuntimeError("No RoboTwin training sample selected.")
    frames, _ = run_episode(
        model=worker.model,
        processor=worker.processor,
        action_np=state["action_np"],
        prompt=state["prompt"],
        init_image_tensor=state["init_image"].to(device=worker.model.device, dtype=worker.model.torch_dtype),
        num_inference_steps=num_steps,
        seed=seed,
        max_rollouts=None,
        state_np=state["state_np"],
    )
    cond = _qpos_to_condition_payload(state["action_np"], state["state_np"], n=len(frames))
    state["last_generated"] = frames
    state["last_cond_states"] = cond["cond_states"]
    state["last_cond_bases"] = cond["cond_bases"]
    return {"generated": [_to_b64(f) for f in frames], "fps": 8, **cond}


def _do_sample_wa(worker: ModelWorker, state: dict[str, Any], dataset_key: str | None = None) -> dict[str, Any]:
    _activate_worker(worker)
    key = _worldarena_dataset_key(dataset_key)
    data_root = _worldarena_root_for(key)
    episodes = _episode_ids_from_worldarena(data_root)
    if not episodes:
        raise RuntimeError(f"No WorldArena episodes found under {data_root}")
    ep = random.choice(episodes)
    init_image, prompt, state_np, action_np, gt_frames, gt_path, first_frame = _load_worldarena_sample(data_root, ep, worker.model)
    payload = _qpos_to_condition_payload(action_np, state_np, n=33)
    state.clear()
    state.update({
        "dataset_key": key,
        "data_root": str(data_root),
        "ep": ep,
        "init_image": init_image,
        "prompt": prompt,
        "action_np": action_np,
        "state_np": state_np,
        "gt_frames": gt_frames,
        "gt_video_path": str(gt_path) if gt_path is not None else "",
        **payload,
    })
    return {
        "ep": ep,
        "dataset": key,
        "prompt": prompt,
        "first_frame": _to_b64(first_frame),
        "gt_frames": [_to_b64(f) for f in gt_frames],
        "gt_path": str(gt_path) if gt_path is not None else "",
        **payload,
    }


def _do_infer_wa(worker: ModelWorker, state: dict[str, Any], num_steps: int, seed: int) -> dict[str, Any]:
    _activate_worker(worker)
    if not state:
        raise RuntimeError("No WorldArena sample selected.")
    frames, _ = run_episode(
        model=worker.model,
        processor=worker.processor,
        action_np=state["action_np"],
        prompt=state["prompt"],
        init_image_tensor=state["init_image"].to(device=worker.model.device, dtype=worker.model.torch_dtype),
        num_inference_steps=num_steps,
        seed=seed,
        max_rollouts=None,
        state_np=state["state_np"],
    )
    gt_frames = state.get("gt_frames", [])
    gt_path = state.get("gt_video_path")
    if gt_path:
        gt_frames = _read_worldarena_gt_video_path(Path(gt_path), n=len(frames))
    cond = _qpos_to_condition_payload(state["action_np"], state["state_np"], n=len(frames))
    state["last_generated"] = frames
    state["last_gt"] = gt_frames
    state["last_cond_states"] = cond["cond_states"]
    state["last_cond_bases"] = cond["cond_bases"]
    return {"generated": [_to_b64(f) for f in frames], "gt_frames": [_to_b64(f) for f in gt_frames], "fps": 8, **cond}


def _visible_cuda_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if ids:
            return ids
    return [str(i) for i in range(torch.cuda.device_count())]


def _episode_chunks(episodes: list[int], n: int) -> list[list[int]]:
    n = max(1, min(int(n), len(episodes)))
    q, r = divmod(len(episodes), n)
    chunks, off = [], 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        chunks.append(episodes[off:off + size])
        off += size
    return [c for c in chunks if c]


def _format_episode_spec(episodes: list[int]) -> str:
    if not episodes:
        return ""
    parts, start, prev = [], episodes[0], episodes[0]
    for ep in episodes[1:]:
        if ep == prev + 1:
            prev = ep
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = ep
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _wa_batch_payload_locked() -> dict[str, Any]:
    output_root = _WA_BATCH_STATE.get("output_root", "")
    if output_root:
        _WA_BATCH_STATE["completed"] = len(list(Path(output_root).glob("**/episode*.mp4")))
    procs = _WA_BATCH_STATE.get("processes") or []
    if _WA_BATCH_STATE.get("status") == "running" and procs:
        codes = [p.poll() for p in procs]
        if all(code is not None for code in codes):
            _WA_BATCH_STATE["status"] = "done" if all(code == 0 for code in codes) else "failed"
            _WA_BATCH_STATE["finished_at"] = time.time()
    return {
        "type": "batch_status",
        "status": _WA_BATCH_STATE.get("status", "idle"),
        "completed": int(_WA_BATCH_STATE.get("completed", 0)),
        "total": int(_WA_BATCH_STATE.get("total", 0)),
        "gpus": ",".join(_WA_BATCH_STATE.get("gpus", [])),
        "dataset": str(_WA_BATCH_STATE.get("dataset", _worldarena_dataset_key())),
        "output_root": str(_WA_BATCH_STATE.get("output_root", "")),
        "merged_root": "",
        "logs": _WA_BATCH_STATE.get("logs", []),
    }


def _wa_batch_status() -> dict[str, Any]:
    with _WA_BATCH_LOCK:
        return _wa_batch_payload_locked()


def _start_wa_batch_eval(options: dict[str, Any] | None = None) -> dict[str, Any]:
    if _WEB_ARGS is None:
        raise RuntimeError("Web args are not initialized.")
    options = dict(options or {})
    dataset_key = _worldarena_dataset_key(options.get("dataset"))
    data_root = _worldarena_root_for(dataset_key)
    episodes = _episode_ids_from_worldarena(data_root)
    if not episodes:
        raise RuntimeError(f"No WorldArena episodes found under {data_root}")
    gpus = _visible_cuda_ids()
    if not gpus:
        raise RuntimeError("No visible CUDA GPUs found for WorldArena full eval")
    with _WA_BATCH_LOCK:
        payload = _wa_batch_payload_locked()
        if payload.get("status") == "running":
            return payload
        job_id = time.strftime("v5_worldarena_%Y%m%d_%H%M%S")
        output_root = Path(_WEB_ARGS.worldarena_eval_output_root).expanduser() / job_id
        output_root.mkdir(parents=True, exist_ok=True)
        chunks = _episode_chunks(episodes, len(gpus))
        logs, procs = [], []
        for shard_idx, chunk in enumerate(chunks):
            gpu_id = gpus[shard_idx]
            log_path = output_root / f"shard_{shard_idx:02d}.log"
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--ckpt", _WEB_ARGS.ckpt or _default_ckpt_for_step(_WEB_ARGS.step),
                "--stats", _WEB_ARGS.stats,
                "--cfg_name", _WEB_ARGS.cfg_name,
                "--sim_task", _WEB_ARGS.sim_task,
                "--normalization_mode", _WEB_ARGS.normalization_mode,
                "--image_layout", _WEB_ARGS.image_layout,
                "--policy_action_condition_mode", _WEB_ARGS.policy_action_condition_mode,
                "--data_root", str(data_root),
                "--left_mp4", _WEB_ARGS.left_mp4,
                "--right_mp4", _WEB_ARGS.right_mp4,
                "--hdf5_action_key", _WEB_ARGS.hdf5_action_key,
                "--num_inference_steps", str(_WEB_ARGS.num_inference_steps),
                "--episodes", _format_episode_spec(chunk),
                "--output_root", str(output_root),
                "--output_name", f"shard_{shard_idx:02d}",
                "--skip_existing",
                "--continue_on_error",
            ]
            if bool(options.get("crop_main_view", False)):
                cmd.append("--crop_main_view")
            if bool(options.get("save_gt", True)):
                gt_dir = data_root / "real_gt_videos"
                if gt_dir.is_dir():
                    cmd.extend(["--save_gt", "--gt_video_dir", str(gt_dir)])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
            log_f = open(log_path, "w", buffering=1)
            procs.append(subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT))
            logs.append(str(log_path))
        _WA_BATCH_STATE.clear()
        _WA_BATCH_STATE.update({
            "status": "running",
            "completed": 0,
            "total": len(episodes),
            "dataset": dataset_key,
            "gpus": gpus[:len(chunks)],
            "output_root": str(output_root),
            "logs": logs,
            "processes": procs,
            "started_at": time.time(),
        })
        return _wa_batch_payload_locked()


app = FastAPI()


@app.get("/")
def index():
    return HTMLResponse(_html())


@app.get("/worker_status")
def worker_status():
    return {"pid": os.getpid(), "pool": _WEB_POOL.status() if _WEB_POOL is not None else None}


@app.get("/video.gif")
def video_gif(sid: str = ""):
    data = _session_output(sid, "video")
    frames = data.get("frames", [])
    if not frames:
        from fastapi.responses import Response
        return Response(status_code=404, content=b"No video yet")
    cond = _joint_condition_frames(data.get("cond_states", []), data.get("cond_bases", []))
    return _gif_response([frames, cond], "dw05-v5.gif")


@app.get("/robotwin.gif")
def robotwin_gif(sid: str = ""):
    data = _session_output(sid, "robotwin")
    generated = data.get("generated", [])
    gt = data.get("gt", [])
    cond = _joint_condition_frames(data.get("cond_states", []), data.get("cond_bases", []))
    return _gif_response([generated, gt, cond], "robotwin.gif")


@app.get("/worldarena.gif")
def worldarena_gif(sid: str = ""):
    data = _session_output(sid, "worldarena")
    gt = data.get("gt", [])
    generated = data.get("generated", [])
    cond = _joint_condition_frames(data.get("cond_states", []), data.get("cond_bases", []))
    return _gif_response([gt, generated, cond], "worldarena.gif")


async def _require_pool() -> ModelWorkerPool:
    if _WEB_POOL is None:
        raise RuntimeError("Worker pool is not initialized.")
    return _WEB_POOL


@app.websocket("/ws/robotwin")
async def ws_robotwin(ws: WebSocket):
    await ws.accept()
    sid = _session_id()
    state: dict[str, Any] = {}
    await ws.send_text(json.dumps({"type": "session", "sid": sid}))
    try:
        while True:
            data = json.loads(await ws.receive_text())
            try:
                pool = await _require_pool()
                if data.get("action") == "sample":
                    await ws.send_text(json.dumps({"type": "inferring"}))
                    async with pool.lease() as worker:
                        result = await asyncio.to_thread(_do_sample_rw, worker, state)
                    _cache_session_output(sid, "robotwin", gt=state.get("gt_frames", []), cond_states=state.get("cond_states", []), cond_bases=state.get("cond_bases", []))
                    await ws.send_text(json.dumps({"type": "episode", "sid": sid, **result}))
                elif data.get("action") == "infer":
                    await ws.send_text(json.dumps({"type": "inferring"}))
                    async with pool.lease() as worker:
                        result = await asyncio.to_thread(_do_infer_rw, worker, state, _WEB_ARGS.num_inference_steps, _WEB_ARGS.seed)
                    _cache_session_output(sid, "robotwin", generated=state.get("last_generated", []), gt=state.get("gt_frames", []), cond_states=state.get("last_cond_states", state.get("cond_states", [])), cond_bases=state.get("last_cond_bases", state.get("cond_bases", [])))
                    await ws.send_text(json.dumps({"type": "result", "sid": sid, **result}))
            except Exception as exc:
                await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/worldarena")
async def ws_worldarena(ws: WebSocket):
    await ws.accept()
    sid = _session_id()
    state: dict[str, Any] = {}
    await ws.send_text(json.dumps({"type": "session", "sid": sid}))
    try:
        while True:
            data = json.loads(await ws.receive_text())
            try:
                pool = await _require_pool()
                if data.get("action") == "sample":
                    async with pool.lease() as worker:
                        result = await asyncio.to_thread(_do_sample_wa, worker, state, data.get("dataset"))
                    _cache_session_output(sid, "worldarena", gt=state.get("gt_frames", []), cond_states=state.get("cond_states", []), cond_bases=state.get("cond_bases", []))
                    await ws.send_text(json.dumps({"type": "episode", "sid": sid, **result}))
                elif data.get("action") == "infer":
                    await ws.send_text(json.dumps({"type": "inferring"}))
                    async with pool.lease() as worker:
                        result = await asyncio.to_thread(_do_infer_wa, worker, state, _WEB_ARGS.num_inference_steps, _WEB_ARGS.seed)
                    _cache_session_output(sid, "worldarena", generated=state.get("last_generated", []), gt=state.get("last_gt", state.get("gt_frames", [])), cond_states=state.get("last_cond_states", state.get("cond_states", [])), cond_bases=state.get("last_cond_bases", state.get("cond_bases", [])))
                    await ws.send_text(json.dumps({"type": "result", "sid": sid, **result}))
                elif data.get("action") == "batch_start":
                    result = await asyncio.to_thread(_start_wa_batch_eval, data)
                    await ws.send_text(json.dumps(result))
                elif data.get("action") == "batch_status":
                    await ws.send_text(json.dumps(_wa_batch_status()))
            except Exception as exc:
                if data.get("action", "").startswith("batch"):
                    await ws.send_text(json.dumps({"type": "batch_status", "status": "failed", "completed": 0, "total": 0, "dataset": data.get("dataset", ""), "error": str(exc)}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sid = _session_id()
    try:
        pool = await _require_pool()
        async with pool.lease() as worker:
            if _WEB_BOOT_SAMPLE is None:
                sample = await asyncio.to_thread(_sample_train_frame_init, worker.model, _WEB_ARGS)
            else:
                sample = _WEB_BOOT_SAMPLE
            demo = V5JointDemo(
                worker.model,
                worker.processor,
                sample["init_image"].to(device=worker.model.device, dtype=worker.model.torch_dtype),
                sample["prompt"],
                sample["state"],
                _WEB_ARGS.num_inference_steps,
                _WEB_ARGS.seed,
                sample_label=sample["sample_label"],
                init_eef_state=sample.get("eef_state"),
            )
        await ws.send_text(json.dumps({
            "type": "init",
            "sid": sid,
            "video": _to_b64(_tensor_image_to_pil(demo.image)),
            "prompt": f'{demo.prompt}  [{demo.sample_label}]' if demo.sample_label else demo.prompt,
            **demo.payload(),
        }))
        try:
            while True:
                data = json.loads(await ws.receive_text())
                action = data.get("action")
                try:
                    if action == "set_joint":
                        payload = demo.set_joint(data.get("state", []))
                        await ws.send_text(json.dumps({"type": "joint_state", "sid": sid, **payload}))
                    elif action == "set_eef":
                        payload = demo.set_eef(data.get("state", []))
                        await ws.send_text(json.dumps({"type": "joint_state", "sid": sid, **payload}))
                    elif action == "sample_init":
                        await ws.send_text(json.dumps({"type": "sampling", "sid": sid}))
                        async with pool.lease() as worker:
                            sample = await asyncio.to_thread(_sample_train_frame_init, worker.model, _WEB_ARGS)
                            demo.use_worker(worker)
                        payload = demo.set_initial(sample["init_image"].to(device=demo.model.device, dtype=demo.model.torch_dtype), sample["prompt"], sample["state"], sample_label=sample["sample_label"], init_eef_state=sample.get("eef_state"))
                        await ws.send_text(json.dumps({
                            "type": "init",
                            "sid": sid,
                            "video": _to_b64(_tensor_image_to_pil(demo.image)),
                            "prompt": f'{demo.prompt}  [{demo.sample_label}]',
                            **payload,
                        }))
                    elif action == "reset":
                        payload = demo.reset()
                        await ws.send_text(json.dumps({
                            "type": "init",
                            "sid": sid,
                            "video": _to_b64(_tensor_image_to_pil(demo.image)),
                            "prompt": f'{demo.prompt}  [{demo.sample_label}]' if demo.sample_label else demo.prompt,
                            **payload,
                        }))
                    elif action == "record_start":
                        await ws.send_text(json.dumps({"type": "joint_state", "sid": sid, **demo.record_start()}))
                    elif action == "record_stop":
                        await ws.send_text(json.dumps({"type": "joint_state", "sid": sid, **demo.record_stop()}))
                    elif action == "infer":
                        await ws.send_text(json.dumps({"type": "inferring", "sid": sid}))
                        async with pool.lease() as worker:
                            demo.use_worker(worker)
                            frames, payload = await asyncio.to_thread(demo.infer)
                        _cache_session_output(sid, "video", frames=demo.last_frames, cond_states=demo.last_cond_states, cond_bases=demo.last_cond_bases)
                        await ws.send_text(json.dumps({
                            "type": "frames",
                            "sid": sid,
                            "frames": [_to_b64(f) for f in frames],
                            "fps": 8,
                            "prompt": f'{demo.prompt}  [{demo.sample_label}]' if demo.sample_label else demo.prompt,
                            **payload,
                        }))
                except Exception as exc:
                    await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except WebSocketDisconnect:
            pass
    except Exception as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))


proxy_app = FastAPI()


def _proxy_worker_url(worker: dict[str, Any], path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"http://127.0.0.1:{worker['port']}{path}"


def _proxy_worker_ws_url(worker: dict[str, Any], path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"ws://127.0.0.1:{worker['port']}{path}"


def _proxy_alive(worker: dict[str, Any]) -> bool:
    proc = worker.get("process")
    return proc is not None and proc.poll() is None


def _choose_proxy_worker() -> dict[str, Any]:
    with _PROXY_LOCK:
        alive = [w for w in _PROXY_WORKERS if _proxy_alive(w)]
        if not alive:
            raise RuntimeError("No live DW05 worker processes are available.")
        worker = min(alive, key=lambda w: (int(w.get("active", 0)), int(w.get("jobs", 0)), int(w.get("id", 0))))
        worker["active"] = int(worker.get("active", 0)) + 1
        return worker


def _release_proxy_worker(worker: dict[str, Any]) -> None:
    with _PROXY_LOCK:
        worker["active"] = max(0, int(worker.get("active", 0)) - 1)


def _remember_proxy_sid(sid: str, worker: dict[str, Any]) -> None:
    if not sid:
        return
    with _PROXY_LOCK:
        _PROXY_SID_TO_WORKER[sid] = worker


def _worker_for_sid(sid: str) -> dict[str, Any] | None:
    with _PROXY_LOCK:
        worker = _PROXY_SID_TO_WORKER.get(sid)
        if worker and _proxy_alive(worker):
            return worker
    return None


def _first_live_proxy_worker() -> dict[str, Any] | None:
    with _PROXY_LOCK:
        for worker in _PROXY_WORKERS:
            if _proxy_alive(worker):
                return worker
    return None


def _http_get_bytes(url: str, timeout: float = 10.0) -> tuple[bytes, str, int]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            ctype = resp.headers.get("content-type", "application/octet-stream")
            return resp.read(), ctype, int(resp.status)
    except Exception as exc:
        return str(exc).encode("utf-8"), "text/plain", 502


@proxy_app.get("/")
def proxy_index():
    worker = _first_live_proxy_worker()
    if worker is None:
        return Response(status_code=503, content=b"No live DW05 worker processes are available")
    body, ctype, status = _http_get_bytes(_proxy_worker_url(worker, "/"), timeout=30.0)
    return Response(content=body, media_type=ctype, status_code=status)


@proxy_app.get("/video.gif")
def proxy_video_gif(sid: str = ""):
    worker = _worker_for_sid(sid) or _first_live_proxy_worker()
    if worker is None:
        return Response(status_code=503, content=b"No live DW05 worker processes are available")
    body, ctype, status = _http_get_bytes(_proxy_worker_url(worker, f"/video.gif?sid={sid}"), timeout=60.0)
    return Response(content=body, media_type=ctype, status_code=status)


@proxy_app.get("/robotwin.gif")
def proxy_robotwin_gif(sid: str = ""):
    worker = _worker_for_sid(sid) or _first_live_proxy_worker()
    if worker is None:
        return Response(status_code=503, content=b"No live DW05 worker processes are available")
    body, ctype, status = _http_get_bytes(_proxy_worker_url(worker, f"/robotwin.gif?sid={sid}"), timeout=60.0)
    return Response(content=body, media_type=ctype, status_code=status)


@proxy_app.get("/worldarena.gif")
def proxy_worldarena_gif(sid: str = ""):
    worker = _worker_for_sid(sid) or _first_live_proxy_worker()
    if worker is None:
        return Response(status_code=503, content=b"No live DW05 worker processes are available")
    body, ctype, status = _http_get_bytes(_proxy_worker_url(worker, f"/worldarena.gif?sid={sid}"), timeout=60.0)
    return Response(content=body, media_type=ctype, status_code=status)


@proxy_app.get("/worker_status")
def proxy_worker_status():
    with _PROXY_LOCK:
        return {
            "mode": "proxy",
            "workers": [
                {"id": w["id"], "gpu": w["gpu"], "port": w["port"], "active": w.get("active", 0), "jobs": w.get("jobs", 0), "alive": _proxy_alive(w)}
                for w in _PROXY_WORKERS
            ],
        }


async def _proxy_websocket(ws: WebSocket, path: str) -> None:
    await ws.accept()
    worker = _choose_proxy_worker()
    try:
        async with websockets.connect(_proxy_worker_ws_url(worker, path), max_size=2 ** 28) as upstream:
            async def client_to_worker():
                while True:
                    msg = await ws.receive_text()
                    await upstream.send(msg)

            async def worker_to_client():
                while True:
                    msg = await upstream.recv()
                    try:
                        data = json.loads(msg)
                        sid = data.get("sid")
                        if sid:
                            _remember_proxy_sid(str(sid), worker)
                    except Exception:
                        pass
                    await ws.send_text(msg)

            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_worker()), asyncio.create_task(worker_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        with _PROXY_LOCK:
            worker["jobs"] = int(worker.get("jobs", 0)) + 1
        _release_proxy_worker(worker)


@proxy_app.websocket("/ws")
async def proxy_ws(ws: WebSocket):
    await _proxy_websocket(ws, "/ws")


@proxy_app.websocket("/ws/robotwin")
async def proxy_ws_robotwin(ws: WebSocket):
    await _proxy_websocket(ws, "/ws/robotwin")


@proxy_app.websocket("/ws/worldarena")
async def proxy_ws_worldarena(ws: WebSocket):
    await _proxy_websocket(ws, "/ws/worldarena")


def _wait_worker_ready(port: int, timeout_s: float = 900.0) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/worker_status"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return int(resp.status) == 200
        except Exception:
            time.sleep(1.0)
    return False


def run_web_proxy(args: argparse.Namespace, ckpt: str) -> None:
    global _WEB_ARGS
    _WEB_ARGS = args
    devices = _web_worker_devices(args)
    base_port = int(args.port) + 1
    workers = []
    env_base = os.environ.copy()
    env_base.setdefault("TOKENIZERS_PARALLELISM", "false")
    env_base.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    for idx, device in enumerate(devices):
        gpu = device.split(":", 1)[1] if device.startswith("cuda:") else str(idx)
        port = base_port + idx
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--web",
            "--step", str(args.step),
            "--episode", str(args.episode),
            "--port", str(port),
            "--web_gpu_count", "1",
            "--device", "cuda" if device.startswith("cuda") else device,
            "--ckpt", ckpt,
            "--stats", args.stats,
            "--cfg_name", args.cfg_name,
            "--sim_task", args.sim_task,
            "--normalization_mode", args.normalization_mode,
            "--image_layout", args.image_layout,
            "--policy_action_condition_mode", args.policy_action_condition_mode,
            "--model_base_path", args.model_base_path,
            "--data_root", args.data_root,
            "--worldarena_test_data_root", args.worldarena_test_data_root,
            "--worldarena_gt_video_dir", args.worldarena_gt_video_dir,
            "--worldarena_eval_output_root", args.worldarena_eval_output_root,
            "--train_root", args.train_root,
            "--hdf5_action_key", args.hdf5_action_key,
            "--num_inference_steps", str(args.num_inference_steps),
            "--seed", str(args.seed),
            "--left_mp4", args.left_mp4,
            "--right_mp4", args.right_mp4,
        ]
        cmd.append("--train_video_bgr" if args.train_video_bgr else "--no-train_video_bgr")
        env = env_base.copy()
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = Path("/tmp") / f"dw05_v5_worker_{idx}_{port}.log"
        log_f = open(log_path, "w", buffering=1)
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        worker = {"id": idx, "gpu": gpu, "port": port, "process": proc, "active": 0, "jobs": 0, "log": str(log_path)}
        workers.append(worker)
        print(f"spawned_worker={idx} gpu={gpu} port={port} log={log_path}", flush=True)
    _PROXY_WORKERS[:] = workers

    def cleanup_workers():
        for worker in workers:
            proc = worker.get("process")
            if proc is not None and proc.poll() is None:
                proc.terminate()
    atexit.register(cleanup_workers)

    print("waiting_for_workers=1", flush=True)
    for worker in workers:
        if not _wait_worker_ready(int(worker["port"])):
            raise RuntimeError(f"Worker {worker['id']} did not become ready. See {worker['log']}")
        print(f"ready_worker={worker['id']} gpu={worker['gpu']} port={worker['port']}", flush=True)
    print("worker_proxy=" + json.dumps(proxy_worker_status(), separators=(",", ":")), flush=True)
    print(f"Open http://localhost:{args.port}", flush=True)
    uvicorn.run(proxy_app, host="0.0.0.0", port=args.port, log_level="warning")


def _init_state_for_episode(args: argparse.Namespace, ep: int) -> np.ndarray:
    state_np, _ = _load_worldarena_condition(args, ep)
    return state_np[0].astype(np.float32)


def run_web(args: argparse.Namespace) -> None:
    global _WEB_POOL, _WEB_ARGS, _WEB_BOOT_SAMPLE
    ep = int(args.episode)
    ckpt = args.ckpt or _default_ckpt_for_step(args.step)
    args.ckpt = ckpt
    _WEB_ARGS = args
    print("=== DW05 RobotWin online joint demo ===")
    print(f"ckpt={ckpt}")
    print(f"stats={args.stats}")
    print(f"sim_task={args.sim_task}")
    print(f"normalization_mode={args.normalization_mode}")
    print(f"image_layout={args.image_layout}")
    print(f"policy_action_condition_mode={args.policy_action_condition_mode}")
    print(f"episode={ep}")
    print(f"web_gpu_count={args.web_gpu_count}")
    model_base = _prepare_local_model_base(args.model_base_path)
    print(f"model_base_path={model_base}")
    if int(args.web_gpu_count) > 1:
        run_web_proxy(args, ckpt)
        return
    _WEB_POOL = _load_web_worker_pool(args, ckpt)
    first_worker = _WEB_POOL.workers[0]
    _WEB_BOOT_SAMPLE = _sample_train_frame_init(first_worker.model, args)
    print(f"initial_sample={_WEB_BOOT_SAMPLE['sample_label']}")
    print("worker_pool=" + json.dumps(_WEB_POOL.status(), separators=(",", ":")))
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def run_cli(args: argparse.Namespace) -> int:
    ckpt = args.ckpt or _default_ckpt_for_step(args.step)
    episodes = parse_episodes(args.episodes) if args.episodes else [int(args.episode)]
    max_rollouts = args.max_rollouts if args.max_rollouts > 0 else None
    output_root = Path(args.output_root).expanduser()

    print("=== DW05 RobotWin online inference ===")
    print(f"ckpt={ckpt}")
    print(f"stats={args.stats}")
    print(f"sim_task={args.sim_task}")
    print(f"normalization_mode={args.normalization_mode}")
    print(f"image_layout={args.image_layout}")
    print(f"policy_action_condition_mode={args.policy_action_condition_mode}")
    print(f"episodes={episodes}")
    print(f"action_mode=DW05 policy absolute qpos; relative_mask_hint={RELATIVE_DIM_MASK.astype(int).tolist()}")
    print(f"action_condition={args.action_condition or 'WorldArena HDF5 joint_action/vector'}")

    model, processor, _ = _load_model_and_processor_for_web_worker(
        ckpt,
        args.stats,
        args.device,
        args.cfg_name,
        sim_task=args.sim_task,
        normalization_mode=args.normalization_mode,
        image_layout=args.image_layout,
        policy_action_condition_mode=args.policy_action_condition_mode,
    )

    saved: list[str] = []
    skipped: list[str] = []
    errors: list[dict[str, Any]] = []
    external_condition = bool(args.action_condition)
    if external_condition and len(episodes) != 1:
        raise ValueError("External --action_condition currently supports exactly one episode/first-frame context.")

    for ep in episodes:
        try:
            target_path = _target_video_path(output_root, args.output_name, ep if not external_condition else None)
            if args.skip_existing and target_path.exists() and target_path.stat().st_size > 0:
                print(f"Episode {ep}: skip existing {target_path}")
                skipped.append(str(target_path))
                continue

            prompt = _load_prompt(Path(args.data_root).expanduser(), ep, args.prompt, args.instr_root or None)
            init_image = _load_init_image(args, model, processor, ep)
            if external_condition:
                state_np, action_np, normalized = _load_external_condition(args)
            else:
                state_np, action_np = _load_worldarena_condition(args, ep)
                normalized = False

            print(
                f"Episode {ep}: action_abs_shape={tuple(action_np.shape)} "
                f"state_shape={None if state_np is None else tuple(state_np.shape)} "
                f"normalized_relative={normalized}"
            )
            if not normalized and state_np is not None and len(state_np):
                print("  first absolute action:", np.array2string(action_np[0], precision=4, suppress_small=True))
                print("  first state/proprio:", np.array2string(state_np[0], precision=4, suppress_small=True))

            if normalized:
                frames = run_episode_with_normalized_action(
                    model=model,
                    action_np=action_np,
                    prompt=prompt,
                    init_image_tensor=init_image,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    max_rollouts=max_rollouts,
                    proprio_np=state_np,
                )
                cond_frames = []
            else:
                frames, cond_frames = run_episode(
                    model=model,
                    processor=processor,
                    action_np=action_np,
                    prompt=prompt,
                    init_image_tensor=init_image,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    max_rollouts=max_rollouts,
                    state_np=state_np,
                )

            gt_frames = read_gt_frames(args.gt_video_dir, ep, n=len(frames)) if args.save_gt else []
            out_dir = output_root if external_condition else output_root / args.output_name
            name = args.output_name if external_condition else f"episode{int(ep)}"
            saved_path = save_video(
                frames,
                out_dir,
                name,
                gt_frames=gt_frames or None,
                cond_frames=cond_frames or None,
                save_gt=args.save_gt,
                save_generated=True,
                save_condition=False,
                crop_main_view=args.crop_main_view,
            )
            saved.append(str(saved_path))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"Episode {ep} failed: {exc}")
            traceback.print_exc()
            errors.append({"episode": int(ep), "error": repr(exc), "traceback": traceback.format_exc()})

    summary = {
        "ckpt": ckpt,
        "stats": args.stats,
        "sim_task": args.sim_task,
        "cfg_name": args.cfg_name,
        "episodes": episodes,
        "relative_dim_mask_hint": RELATIVE_DIM_MASK.astype(bool).tolist(),
        "action_condition": args.action_condition,
        "action_condition_mode": args.action_condition_mode,
        "state_condition": args.state_condition,
        "saved_videos": saved,
        "skipped_videos": skipped,
        "errors": errors,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "max_rollouts": max_rollouts,
        "crop_main_view": args.crop_main_view,
    }
    summary_path = output_root / f"{args.output_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {summary_path}")
    print("All done.")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--web", action="store_true", help="Start the interactive joint-condition web demo.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8898)
    p.add_argument("--web_gpu_count", type=int, default=1, help="Number of CUDA worker model replicas to load for concurrent web inference. With --device cuda, --web_gpu_count 8 loads cuda:0..cuda:7.")
    p.add_argument("--ckpt", default="", help="V5 checkpoint path. Defaults to the DW05 V5 checkpoint alias when available.")
    p.add_argument("--step", default=V5_DEFAULT_STEP, help="V5 ckpt step, used when --ckpt is empty.")
    p.add_argument("--model_base_path", default="", help="Local DW05 runtime bundle root. Defaults to DW05_MODEL_BASE_PATH or known development caches.")
    p.add_argument("--stats", default=V5_STATS_PATH)
    p.add_argument("--cfg_name", default="runtime/sim_robotwin")
    p.add_argument("--sim_task", default=V5_SIM_TASK)
    p.add_argument("--normalization_mode", default="zscore_14d", choices=["zscore_14d", "dw05_quantile"])
    p.add_argument("--image_layout", default="robotwin_resize")
    p.add_argument("--policy_action_condition_mode", default="absolute", choices=["absolute", "delta_first_frame"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--worldarena_test_data_root", default=os.environ.get(
        "WORLDARENA_TEST_DATA_ROOT", "",
    ))
    p.add_argument("--worldarena_gt_video_dir", default=os.environ.get(
        "WORLDARENA_GT_VIDEO_DIR", "",
    ))
    p.add_argument("--worldarena_eval_output_root", default="/tmp/dw05-worldarena-eval")
    p.add_argument("--train_root", default=os.environ.get("DW05_TRAIN_ROOT", ""))
    p.add_argument("--train_video_bgr", action=argparse.BooleanOptionalAction, default=True, help="Treat RoboTwin training annotation mp4 frames as BGR-coded and swap to RGB for display/model input. Use --no-train_video_bgr if a dataset is already RGB.")
    p.add_argument("--hdf5_root", default="")
    p.add_argument("--hdf5_action_key", default="joint_action/vector")
    p.add_argument("--instr_root", default="")
    p.add_argument("--episodes", default="", help="Comma/range spec. If empty, uses --episode.")
    p.add_argument("--episode", type=int, default=576)
    p.add_argument("--prompt", default="")
    p.add_argument("--head_image", default="", help="Optional RGB image for the main camera first frame.")
    p.add_argument("--left_image", default="", help="Optional RGB image for the left wrist first frame.")
    p.add_argument("--right_image", default="", help="Optional RGB image for the right wrist first frame.")
    p.add_argument("--left_mp4", default=DEFAULT_LEFT_MP4 or "")
    p.add_argument("--right_mp4", default=DEFAULT_RIGHT_MP4 or "")
    p.add_argument("--action_condition", default="", help="Optional .npy/.npz/.pt/.json action condition file.")
    p.add_argument("--state_condition", default="", help="Optional state/proprio file for external action condition.")
    p.add_argument("--action_key", default="", help="Array key for dict/npz action condition files.")
    p.add_argument("--state_key", default="", help="Array key for dict/npz state condition files.")
    p.add_argument(
        "--action_condition_mode",
        choices=["auto", "qpos", "raw_action", "normalized"],
        default="auto",
        help="qpos splits [T,D] absolute qpos into state_t/action_t+1; raw_action is absolute target joints; normalized is already-normalized relative action.",
    )
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rollouts", type=int, default=0)
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--output_name", default="robotwin_online_demo")
    p.add_argument("--gt_video_dir", default="", help="Optional directory containing episodeN.mp4 GT videos.")
    p.add_argument("--save_gt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--crop_main_view", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=False)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.web:
        run_web(args)
        return 0
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
