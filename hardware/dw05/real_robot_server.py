#!/usr/bin/env python
"""HTTP service for DW05 real-robot action inference.

Environment variables are intentionally simple and mirror the previous DW05
server where possible:

  CKPT_PATH=/path/to/checkpoint.pt
  DW05_NORM_STATS_PATH=/path/to/norm_stats.json
  DW05_MODEL_BASE_PATH=/path/to/Wan2.2/cache
  SERVER_PORT=8000
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import socket
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from dexbotic.data.dataset.dw05.data_source import ROBOTWIN_META
from dexbotic.policy.dw05_policy import DW05RobotWinPolicy, DW05RobotWinPolicyConfig, parse_bool, parse_optional_float, parse_optional_int


app = FastAPI(title="Dexbotic DW05 Real-Robot HTTP Server")
_policy: Optional[DW05RobotWinPolicy] = None
_policy_lock = asyncio.Lock()
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_runtime: dict[str, Any] = {}


def _env(name: str, default: Any = None) -> Any:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _parse_hw(value: Optional[str], default: tuple[int, int]) -> tuple[int, int]:
    if value is None or not str(value).strip():
        return default
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected H,W image size, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0 or height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"Image size must be positive multiples of 16, got {height}x{width}")
    return height, width


def _decode_image(raw: bytes, decode_mode: str) -> np.ndarray:
    mode = decode_mode.strip().lower()
    if mode == "standard":
        with Image.open(BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if mode == "cv2_passthrough":
        import cv2

        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image with cv2.")
        return decoded
    raise ValueError("REAL_ROBOT_IMAGE_DECODE must be standard or cv2_passthrough.")


def _parse_state(state_text: Optional[str], state_dim: int) -> np.ndarray:
    if state_text is None or not str(state_text).strip():
        return np.zeros(state_dim, dtype=np.float32)
    raw = str(state_text).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [float(x) for x in raw.replace(",", " ").split()]
    if isinstance(parsed, dict):
        parsed = parsed.get("vector", parsed.get("state"))
    if parsed is None:
        raise ValueError("State JSON object must contain `vector` or `state`.")
    state = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if state.shape[0] != state_dim:
        raise ValueError(f"Expected state_dim={state_dim}, got {state.shape[0]}")
    return state


def _decode_request(raw_images: list[bytes], state_text: Optional[str], *, decode_mode: str, state_dim: int) -> dict[str, Any]:
    images = [_decode_image(raw, decode_mode=decode_mode) for raw in raw_images]
    state = _parse_state(state_text, state_dim=state_dim)
    return {"observation": {"images": images}, "joint_action": {"vector": state}}


def _load_policy_from_env() -> DW05RobotWinPolicy:
    checkpoint_path = str(_env("CKPT_PATH"))
    if not checkpoint_path:
        raise ValueError("CKPT_PATH is required.")
    norm_stats_path = str(_env("DW05_NORM_STATS_PATH", _env("DATASET_STATS_PATH", ROBOTWIN_META["norm_stats_path"])))
    device = str(_env("DEVICE", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[real_robot_server] CUDA unavailable; falling back to CPU.", flush=True)
        device = "cpu"
    action_horizon = parse_optional_int(_env("ACTION_HORIZON")) or 32
    replan_steps = parse_optional_int(_env("REPLAN_STEPS")) or min(8, action_horizon)
    image_size = _parse_hw(_env("REAL_ROBOT_IMAGE_SIZE"), (384, 320))
    policy = DW05RobotWinPolicy(
        DW05RobotWinPolicyConfig(
            checkpoint_path=checkpoint_path,
            norm_stats_path=norm_stats_path,
            model_base_path=_env("DW05_MODEL_BASE_PATH"),
            device=device,
            mixed_precision=str(_env("MIXED_PRECISION", "bf16")),
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            num_inference_steps=parse_optional_int(_env("NUM_INFERENCE_STEPS")) or 10,
            num_video_frames=parse_optional_int(_env("NUM_VIDEO_FRAMES")) or 9,
            sigma_shift=parse_optional_float(_env("SIGMA_SHIFT")),
            seed=parse_optional_int(_env("SEED")),
            text_cfg_scale=float(_env("TEXT_CFG_SCALE", 1.0)),
            negative_prompt=str(_env("NEGATIVE_PROMPT", "")),
            rand_device=str(_env("RAND_DEVICE", "cpu")),
            tiled=parse_bool(_env("TILED"), False),
            delta_action=parse_bool(_env("DEPLOY_DELTA_ACTION"), False),
            delta_first_frame=parse_bool(_env("DEPLOY_DELTA_FIRST_FRAME"), True),
            image_layout=str(_env("REAL_ROBOT_IMAGE_LAYOUT", "auto")),
            normalization_mode=str(_env("DW05_NORMALIZATION_MODE", "auto")),
            action_condition_mode=str(_env("DW05_ACTION_CONDITION_MODE", "auto")),
            image_size_hw=image_size,
            raw_state_dim=parse_optional_int(_env("REAL_ROBOT_STATE_DIM", _env("RAW_STATE_DIM"))) or 14,
            raw_action_dim=parse_optional_int(_env("REAL_ROBOT_ACTION_DIM", _env("RAW_ACTION_DIM"))) or 14,
            load_text_encoder=parse_bool(_env("LOAD_TEXT_ENCODER"), True),
        )
    )
    _runtime.update(
        {
            "ckpt_path": checkpoint_path,
            "norm_stats_path": norm_stats_path,
            "device": str(policy.model.device),
            "model_class": type(policy.model).__name__,
            "checkpoint_step": policy.checkpoint_payload.get("step"),
            "action_dim": policy.action_dim,
            "proprio_dim": policy.proprio_dim,
            "action_horizon": action_horizon,
            "replan_steps": replan_steps,
            "num_inference_steps": policy.config.num_inference_steps,
            "image_layout": policy.config.image_layout,
            "normalization_mode": policy.normalization_mode,
            "action_condition_mode": policy.action_condition_mode,
            "image_size_hw": list(policy.config.image_size_hw),
            "image_decode": str(_env("REAL_ROBOT_IMAGE_DECODE", "standard")),
            "state_dim": policy.raw_state_dim,
        }
    )
    return policy


def _warmup(policy: DW05RobotWinPolicy) -> None:
    if not parse_bool(_env("REAL_ROBOT_WARMUP"), False):
        return
    num_images = 3 if policy.config.image_layout.strip().lower() in {"auto", "robotwin", "robotwin_resize", "robotwin_direct"} else 1
    height, width = policy.config.image_size_hw
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    obs = {"observation": {"images": [dummy.copy() for _ in range(num_images)]}, "joint_action": {"vector": np.zeros(policy.raw_state_dim, dtype=np.float32)}}
    t0 = time.perf_counter()
    policy.infer_action_chunk(obs, str(_env("REAL_ROBOT_WARMUP_PROMPT", "warmup")))
    print(f"[real_robot_server] warmup took {time.perf_counter() - t0:.2f}s", flush=True)


@app.on_event("startup")
async def startup() -> None:
    global _policy, _executor
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(_env("REAL_ROBOT_EXECUTOR_THREADS", 4)),
        thread_name_prefix="dw05-real-robot",
    )
    loop = asyncio.get_event_loop()
    _policy = await loop.run_in_executor(_executor, _load_policy_from_env)
    await loop.run_in_executor(_executor, _warmup, _policy)
    ip = _env("MLP_PRIMARY_HOST", _env("MLP_WORKER_0_HOST", socket.gethostbyname(socket.gethostname())))
    _runtime["ip"] = ip
    _runtime["port"] = int(_env("SERVER_PORT", 8000))
    print(f"[real_robot_server] ready: {_runtime}", flush=True)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    if _policy is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"loaded": _policy is not None, **_runtime})


@app.post("/process_frame")
async def process_frame(
    image: list[UploadFile] = File(...),
    prompt: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    states: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    temperature: float = Form(1.0),
) -> JSONResponse:
    del temperature
    if _policy is None or _executor is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    instruction = prompt if prompt is not None else text
    if instruction is None or not instruction.strip():
        raise HTTPException(status_code=400, detail="Form field `prompt` or `text` is required.")

    t0 = time.perf_counter()
    raw_images = await asyncio.gather(*[file.read() for file in image])
    loop = asyncio.get_event_loop()
    try:
        observation = await loop.run_in_executor(
            _executor,
            lambda: _decode_request(
                list(raw_images),
                states if states is not None else state,
                decode_mode=str(_env("REAL_ROBOT_IMAGE_DECODE", "standard")),
                state_dim=int(_runtime.get("state_dim", 14)),
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    t_decode = time.perf_counter()
    generate_video = parse_bool(_env("REAL_ROBOT_GENERATE_VIDEO"), False)
    async with _policy_lock:
        t_infer0 = time.perf_counter()
        if generate_video:
            action_chunk, video_frames = await loop.run_in_executor(_executor, _policy.infer_action_chunk_with_video, observation, instruction.strip())
        else:
            action_chunk = await loop.run_in_executor(_executor, _policy.infer_action_chunk, observation, instruction.strip())
            video_frames = None
        infer_s = time.perf_counter() - t_infer0

    debug_dir = _env("REAL_ROBOT_DEBUG_DIR")
    if debug_dir:
        req_dir = Path(debug_dir) / time.strftime("%Y%m%d_%H%M%S")
        req_dir.mkdir(parents=True, exist_ok=True)
        for idx, raw in enumerate(raw_images):
            (req_dir / f"image_{idx:02d}.jpg").write_bytes(raw)
        if _policy.last_composed_image is not None:
            Image.fromarray(_policy.last_composed_image.astype("uint8")).save(req_dir / "composed.jpg")
        if video_frames is not None:
            from dexbotic.exp.utils import save_mp4

            save_mp4(video_frames, str(req_dir / "video.mp4"), fps=8)
        (req_dir / "action.json").write_text(json.dumps(action_chunk.tolist(), indent=2), encoding="utf-8")

    total_s = time.perf_counter() - t0
    print(
        f"[real_robot_server] images={len(image)} shape={list(action_chunk.shape)} "
        f"decode={t_decode - t0:.3f}s infer={infer_s:.3f}s total={total_s:.3f}s",
        flush=True,
    )
    payload = {
        "action": action_chunk.tolist(),
        "shape": list(action_chunk.shape),
        "timing": {"decode_s": t_decode - t0, "infer_s": infer_s, "total_s": total_s},
    }
    if video_frames is not None:
        payload["num_video_frames"] = len(video_frames)
    return JSONResponse(payload)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(_env("SERVER_PORT", 8000)), workers=1)


if __name__ == "__main__":
    main()
