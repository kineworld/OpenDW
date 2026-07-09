"""Small experiment helpers used by DW05 training and inference.

Dexbotic keeps utilities close to the layer that owns them.  These helpers are
only shared by experiment entrypoints and trainers, so they live under
``dexbotic.exp`` rather than a package-wide ``utils`` namespace.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from typing import Optional

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from rich.logging import RichHandler
from torch.utils.data import Sampler


def _is_main_process() -> bool:
    """Best-effort main-process check without synchronizing workers."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")
    return True


def setup_logging(
    log_level: int | str = logging.INFO,
    is_main_process: Optional[bool] = None,
    rich_handler_kwargs: Optional[dict] = None,
    formatter_kwargs: Optional[dict] = None,
    preserve_file_handlers: bool = True,
) -> None:
    """Configure standard Python logging for single-process or distributed runs."""
    if is_main_process is None:
        is_main_process = _is_main_process()

    root_logger = logging.getLogger()
    if not is_main_process:
        root_logger.setLevel(logging.ERROR)
        return

    existing_file_handlers = []
    if preserve_file_handlers:
        existing_file_handlers = [
            handler for handler in root_logger.handlers if isinstance(handler, logging.FileHandler)
        ]
    root_logger.handlers.clear()

    rich_kwargs = {
        "markup": True,
        "rich_tracebacks": True,
        "show_level": True,
        "show_path": True,
        "show_time": True,
    }
    if rich_handler_kwargs:
        rich_kwargs.update(rich_handler_kwargs)

    formatter_config = {
        "fmt": "| >> %(message)s",
        "datefmt": "%m/%d [%H:%M:%S]",
    }
    if formatter_kwargs:
        formatter_config.update(formatter_kwargs)

    handler = RichHandler(**rich_kwargs)
    handler.setFormatter(logging.Formatter(**formatter_config))
    root_logger.addHandler(handler)
    for file_handler in existing_file_handlers:
        root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)


def normalize_mixed_precision(mixed_precision: str) -> str:
    """Normalize an accelerate-style mixed precision string."""
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """Map mixed precision mode to the dtype used for model construction."""
    precision = normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0"))))


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Seed Python, NumPy, and PyTorch with a rank-specific offset."""
    if not np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max:
        raise ValueError(f"Seed outside the np.uint32 bounds: {seed}.")
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)

    process_seed = seed + _resolve_global_rank()
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """Seed dataloader workers deterministically across ranks."""
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, _resolve_global_rank()])

    np.random.seed(seed_seq.generate_state(4))
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


class ResumableEpochSampler(Sampler[int]):
    """Epoch sampler that can skip already-seen samples after trainer resume."""

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self._resume_sample_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_sample_offset(self, sample_offset: int):
        self._resume_sample_offset = int(sample_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self._resume_sample_offset = int(batch_in_epoch) * self.batch_size * self.num_processes

    def clear_resume_batch_offset(self):
        self._resume_sample_offset = 0

    def _shuffled_indices(self, dataset_len: int) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        return torch.randperm(dataset_len, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        indices = self._shuffled_indices(len(self.dataset))
        if self.epoch == 0 and self._resume_sample_offset > 0:
            indices = indices[self._resume_sample_offset:]

        local_batch = max(self.batch_size, 1)
        cursor = 0
        while cursor < len(indices):
            yield from indices[cursor: cursor + local_batch]
            cursor += local_batch

    def __len__(self) -> int:
        return len(self.dataset)


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_mp4(frames: Iterable[Image.Image], path: str, fps: int = 8):
    """Save PIL frames to an H.264 mp4 with even frame dimensions."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=max(fps, 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(_to_even_frame(np.array(frame.convert("RGB"))))
    finally:
        writer.close()


def pil_frames_to_video_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    """Convert PIL frames to a ``[3, T, H, W]`` tensor in ``[0, 1]``."""
    if len(frames) == 0:
        raise ValueError("`frames` must be non-empty.")
    frame_tensors = []
    for frame in frames:
        arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
        frame_tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return torch.stack(frame_tensors, dim=1)


def _gaussian_kernel_2d(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel_2d.repeat(channels, 1, 1, 1)


def video_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    """Compute average PSNR over ``[3, T, H, W]`` video tensors."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    mse = (pred.float() - target.float()).pow(2).mean(dim=(0, 2, 3))
    psnr = 10.0 * torch.log10((data_range * data_range) / (mse + eps))
    return float(psnr.mean().item())


def video_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute average SSIM over ``[3, T, H, W]`` video tensors."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError("`kernel_size` must be odd.")

    pred = pred.float().permute(1, 0, 2, 3).contiguous()
    target = target.float().permute(1, 0, 2, 3).contiguous()
    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(kernel_size, sigma, channels, pred.device, pred.dtype)

    pad = kernel_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float((numerator / (denominator + 1e-12)).mean().item())


__all__ = [
    "ResumableEpochSampler",
    "mixed_precision_to_model_dtype",
    "normalize_mixed_precision",
    "pil_frames_to_video_tensor",
    "save_mp4",
    "set_global_seed",
    "setup_logging",
    "video_psnr",
    "video_ssim",
]
