"""DW05 dataset implementation.

This module owns dataset indexing, JSONL access, and sample orchestration for
DW05 world-action training. Pure data transformations live under
``dexbotic.data.dataset.dw05.transform`` so the dataset class stays small and
focused.
"""

from __future__ import annotations

import copy
import gc
import glob
import hashlib
import json
import math
import os
import random
import signal
from functools import lru_cache
from typing import Optional

from loguru import logger
import megfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from dexbotic.data.dataset.dw05.data_source_utils import resolve_datasets_info
from dexbotic.data.dataset.dw05.transform.action import (
    ActionNormMultiDataset,
    AddAction,
    AddProprioTrajectory,
    AddTerminationState,
    AddTrajectory,
    ArrangeState,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.dw05.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.dw05.transform.multimodal import (
    LoadImagesWithFutureClip,
    concat_views,
    prepare_future_images,
)
from dexbotic.data.dataset.dw05.transform.output import DW05OutputBuilder, DW05OutputConfig
from dexbotic.data.dataset.dw05.transform.text import PromptSelector, TextEmbeddingCache, refine_text


_SAMPLE_TIMEOUT_S = int(os.getenv("DW_SAMPLE_TIMEOUT", "90"))
_JSONL_OFFSET_CACHE: dict[str, np.ndarray] = {}


class _SampleTimeout(Exception):
    """Raised when a sample load exceeds ``DW_SAMPLE_TIMEOUT`` seconds."""


@lru_cache(maxsize=4096)
def _local_mirror_prefixes() -> tuple[str, ...]:
    value = os.getenv("DW_LOCAL_MIRROR_PREFIXES", "")
    return tuple(prefix.strip().rstrip("/") for prefix in value.split(os.pathsep) if prefix.strip())


@lru_cache(maxsize=4096)
def _resolve_mirrored_path(file_path: str) -> str:
    """Prefer an explicitly configured local mirror for remote S3 paths."""
    if not file_path.startswith("s3://"):
        return file_path
    s3_path = file_path[len("s3://") :]
    for prefix in _local_mirror_prefixes():
        candidate = f"{prefix}/{s3_path}"
        if megfile.smart_exists(candidate):
            return candidate
    return file_path


def _jsonl_offset_cache_dir() -> str:
    return os.getenv("DW_JSONL_OFFSET_CACHE_DIR", "/tmp/dw_jsonl_offsets")


def _jsonl_offset_cache_key(file_path: str) -> tuple[str, str]:
    stat = os.stat(file_path)
    source = f"{file_path}\0{stat.st_size}\0{stat.st_mtime_ns}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return source, digest


def _get_jsonl_offsets(file_path: str) -> np.ndarray:
    cache_key, digest = _jsonl_offset_cache_key(file_path)
    cached = _JSONL_OFFSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_dir = _jsonl_offset_cache_dir()
    cache_path = os.path.join(cache_dir, f"{digest}.npy")
    if os.path.exists(cache_path):
        try:
            offsets = np.load(cache_path, mmap_mode="r")
            _JSONL_OFFSET_CACHE[cache_key] = offsets
            return offsets
        except Exception:
            pass

    offsets = []
    with open(file_path, "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    offsets = np.asarray(offsets, dtype=np.uint64)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as handle:
            np.save(handle, offsets)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass

    max_items = int(os.getenv("DW_JSONL_OFFSET_CACHE_MAX_ITEMS", "64"))
    if max_items > 0 and len(_JSONL_OFFSET_CACHE) >= max_items:
        _JSONL_OFFSET_CACHE.pop(next(iter(_JSONL_OFFSET_CACHE)))
    _JSONL_OFFSET_CACHE[cache_key] = offsets
    return offsets


def _load_jsonl_range(file_path: str, *, start_line: int, end_line: int, parse: bool = False) -> list:
    """Load a contiguous range from a JSONL file, using local offsets when possible."""
    file_path = _resolve_mirrored_path(file_path)
    start_line = max(0, int(start_line))
    end_line = max(start_line, int(end_line))
    rows: list = []

    if os.path.exists(file_path):
        offsets = _get_jsonl_offsets(file_path)
        start_line = min(start_line, len(offsets))
        end_line = min(end_line, len(offsets))
        with open(file_path, "rb") as handle:
            if start_line < end_line:
                handle.seek(int(offsets[start_line]))
            line_index = start_line
            while line_index < end_line:
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                rows.append(json.loads(line) if parse else line.decode("utf-8"))
                line_index += 1
    else:
        line_index = 0
        with megfile.smart_open(file_path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if line_index >= end_line:
                    break
                if line_index >= start_line:
                    rows.append(json.loads(line) if parse else line)
                line_index += 1

    return rows


def _load_jsonl(file_path: str, parse: bool = False) -> list:
    """Load a JSONL file through megfile, preferring configured local mirrors."""
    file_path = _resolve_mirrored_path(file_path)
    with megfile.smart_open(file_path, "r") as handle:
        if parse:
            return [json.loads(line) for line in handle if line.strip()]
        return [line for line in handle if line.strip()]


class DWDataset(Dataset):
    """Dataset for DW05 robot-action and world-model training samples.

    The class resolves a static data recipe once during initialization. Hot data
    recipe creation, polling, rank synchronization, and rollback are outside the
    scope of this clean DW05 dataset implementation.
    """

    _worker_gc_enabled_pids: set[int] = set()
    _jsonl_window_log_pids: set[int] = set()
    _jsonl_fallback_log_pids: set[int] = set()

    def __init__(
        self,
        num_frames: int = 33,
        video_size: tuple[int, int] = (384, 320),
        action_video_freq_ratio: int = 4,
        concat_multi_camera: str = "robotwin",
        processor=None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        enc_id: str = "wan22ti2v5b",
        action_type: str = "state",
        images_keys: Optional[list[str]] = None,
        check_index_cache: bool = True,
        recipe: str = "robotwin_baseline",
        recipe_entries: Optional[list[dict] | dict] = None,
        pretrained_norm_stats=None,
        missing_text_embedding: str = "error",
        text_embedding_dim: int = 4096,
        empty_text_embed_path: Optional[str] = None,
        prompt_add_prob: float = 0.3,
        episode_split_prob: float = 0.0,
        aug_policy: Optional[str | list[str]] = None,
        max_sample_retries: int = 10,
        norm_stats_mode: bool = False,
        **kwargs,
    ) -> None:
        del processor, concat_multi_camera, action_type, pretrained_norm_stats
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DWDataset kwargs: {unknown}")

        self.future_image_size = (int(video_size[0]), int(video_size[1]))
        self.future_frame_count = (int(num_frames) - 1) // int(action_video_freq_ratio)
        self.chunk_size = int(num_frames) - 1
        self.future_frame_stride = int(action_video_freq_ratio)
        self.video_sample_indices = list(range(0, int(num_frames), int(action_video_freq_ratio)))
        self.future_source_key = images_keys or ["images_1", "images_2", "images_3"]
        self.images_keys = images_keys or ["images_1", "images_2", "images_3"]
        self.recipe = recipe
        self.norm_stats_mode = bool(norm_stats_mode)
        self.episode_split_prob = float(episode_split_prob)
        self.check_index_cache = bool(check_index_cache)
        self.max_sample_retries = int(max_sample_retries)
        if self.max_sample_retries < 0:
            raise ValueError("max_sample_retries must be >= 0")

        self.augmentations = self._build_augmentations(aug_policy)
        self.prompt_selector = PromptSelector(
            prompt_add_prob=prompt_add_prob,
            use_rewrite_prompt_and_subtask=os.getenv("USE_REWRITE_PROMPT_AND_SUBTASK", "0").strip() == "1",
        )
        self.text_embedding_cache = TextEmbeddingCache(
            cache_dir=text_embedding_cache_dir,
            context_len=context_len,
            enc_id=enc_id,
            missing=missing_text_embedding,
            embedding_dim=text_embedding_dim,
            empty_text_embed_path=empty_text_embed_path,
        )
        self.output_builder = DW05OutputBuilder(
            DW05OutputConfig(
                action_horizon=self.chunk_size,
                action_dim=32,
                future_frame_count=self.future_frame_count,
                future_frame_stride=self.future_frame_stride,
                future_image_size=self.future_image_size,
            )
        )

        recipe_spec = recipe_entries if recipe_entries is not None else recipe
        self.datasets_info = resolve_datasets_info(
            recipe_spec,
            require_text_embed_dir=not self.norm_stats_mode,
            require_norm_stats_path=not self.norm_stats_mode,
        )
        self._validate_datasets_info(self.datasets_info)
        self.data_transforms = self.build_data_transforms()
        self._per_dataset_index: dict = {}
        self._build_dataset_index()
        self.jsonl_frame_window = self._get_jsonl_frame_window()
        self._log_jsonl_window_state()

    @classmethod
    def _maybe_enable_gc_in_dataloader_worker(cls) -> None:
        value = os.getenv("DW_ENABLE_DATALOADER_WORKER_GC", "")
        if value.lower() not in {"1", "true", "yes", "on"}:
            return
        worker_info = get_worker_info()
        if worker_info is None:
            return
        pid = os.getpid()
        if pid in cls._worker_gc_enabled_pids:
            return
        was_enabled = gc.isenabled()
        if not was_enabled:
            gc.enable()
        cls._worker_gc_enabled_pids.add(pid)
        logger.info(
            "DW DataLoader worker Python GC state: worker_id={}, pid={}, was_enabled={}, now_enabled={}",
            worker_info.id,
            pid,
            was_enabled,
            gc.isenabled(),
        )

    @staticmethod
    def _build_augmentations(aug_policy):
        if aug_policy is None:
            return None
        from dexbotic.data.dataset.dw05.transform.augmentations import PixelAug

        if isinstance(aug_policy, str):
            return PixelAug(policy=aug_policy)
        if isinstance(aug_policy, list):
            return [PixelAug(policy=policy) if policy else None for policy in aug_policy]
        return None

    def _apply_augmentations(self, images: list[Image.Image]) -> list[Image.Image]:
        if self.augmentations is None:
            return images
        if isinstance(self.augmentations, list):
            return [
                augmentation(image) if augmentation is not None and image is not None else image
                for augmentation, image in zip(self.augmentations, images)
            ]
        return [self.augmentations(image) if image is not None else image for image in images]

    def _build_dataset_index(self) -> None:
        current_datasets = {}
        for dataset_info in self.datasets_info:
            data_paths = self._resolve_annotation_paths(dataset_info["annotations"])
            data_path_prefix = dataset_info.get("data_path_prefix", "")
            media_path_resolver = dataset_info.get("media_path_resolver")
            index_path_prefix = dataset_info.get("index_path_prefix", "")
            frequency = dataset_info["frequency"]
            meta_data = dataset_info["meta_data"]
            for data_path in data_paths:
                dataset_key = (data_path, data_path_prefix, media_path_resolver, index_path_prefix)
                current_datasets[dataset_key] = (frequency, meta_data)

        old_keys = set(self._per_dataset_index.keys())
        new_keys = set(current_datasets.keys())
        for key in old_keys - new_keys:
            del self._per_dataset_index[key]

        rebuild_keys = set()
        for key in new_keys:
            if key not in self._per_dataset_index:
                rebuild_keys.add(key)
                continue
            old_freq, old_meta = self._per_dataset_index[key][2], self._per_dataset_index[key][3]
            new_freq, new_meta = current_datasets[key]
            if old_freq != new_freq or old_meta != new_meta:
                rebuild_keys.add(key)

        for key in rebuild_keys:
            data_path, data_path_prefix, media_path_resolver, index_path_prefix = key
            frequency, meta_data = current_datasets[key]
            data_index = self._get_index_cache(data_path)["data"]
            data_index = self._apply_index_path_prefix(data_index, index_path_prefix)
            data_index = list(data_index.items())
            data_index = self._deterministic_shuffle_data_index(data_index)

            sampled_data_index = []
            freq = frequency
            while freq > 0:
                if freq >= 1:
                    sampled_data_index.extend(copy.deepcopy(data_index))
                else:
                    sampled_data_index.extend(copy.deepcopy(data_index[: math.ceil(len(data_index) * freq)]))
                freq -= 1

            self._per_dataset_index[key] = (
                data_path,
                data_path_prefix,
                frequency,
                meta_data,
                media_path_resolver,
                index_path_prefix,
                sampled_data_index,
            )

        self._reassemble_from_per_dataset_index()

    @staticmethod
    def _apply_index_path_prefix(data_index: dict, index_path_prefix: str) -> dict:
        if not index_path_prefix:
            return data_index
        prefix = str(index_path_prefix).rstrip("/")
        prefixed = {}
        for jsonl_file, num_samples in data_index.items():
            jsonl_file = str(jsonl_file)
            if jsonl_file.startswith(("s3://", "http://", "https://")) or os.path.isabs(jsonl_file):
                prefixed[jsonl_file] = num_samples
            else:
                prefixed[f"{prefix}/{jsonl_file.lstrip('/')}"] = num_samples
        return prefixed

    def _reassemble_from_per_dataset_index(self) -> None:
        global_index = []
        file_name_map = {}
        file_sample_count_map = {}
        dataset_map = {}
        file_id = 0
        dataset_id = 0

        for key in sorted(self._per_dataset_index.keys()):
            (
                data_path,
                data_path_prefix,
                _frequency,
                meta_data,
                media_path_resolver,
                index_path_prefix,
                sampled_data_index,
            ) = self._per_dataset_index[key]
            dataset_map[dataset_id] = {
                "data_path": data_path,
                "meta_data": meta_data,
                "data_path_prefix": data_path_prefix,
                "media_path_resolver": media_path_resolver,
                "index_path_prefix": index_path_prefix,
            }
            dataset_index = dataset_id
            dataset_id += 1

            for jsonl_file, num_samples in sampled_data_index:
                if jsonl_file not in file_name_map:
                    file_name_map[jsonl_file] = file_id
                    file_id += 1
                file_index = file_name_map[jsonl_file]
                file_sample_count_map[file_index] = int(num_samples)
                for frame_index in range(int(num_samples)):
                    global_index.append((dataset_index, file_index, frame_index))

        self.global_index = global_index
        self.file_name_map = {value: key for key, value in file_name_map.items()}
        self.file_sample_count_map = file_sample_count_map
        self.dataset_map = dataset_map
        self.total_samples = len(global_index)

    @classmethod
    def _resolve_annotation_paths(cls, annotations) -> list[str]:
        if isinstance(annotations, (str, os.PathLike)):
            annotation_items = [str(annotations)]
        elif isinstance(annotations, (list, tuple)):
            annotation_items = [str(item) for item in annotations]
        else:
            raise TypeError("annotations must be a path, glob, or list of paths")

        data_paths = []
        seen = set()
        for annotation in annotation_items:
            matches = cls._expand_annotation_glob(annotation) if glob.has_magic(annotation) else [annotation]
            if glob.has_magic(annotation) and not matches:
                raise FileNotFoundError(f"No annotation paths matched: {annotation}")
            for path in matches:
                if path in seen:
                    continue
                seen.add(path)
                data_paths.append(path)
        return data_paths

    @staticmethod
    def _expand_annotation_glob(annotation: str) -> list[str]:
        if annotation.endswith("/*/") and annotation.count("*") == 1:
            root = annotation[:-3].rstrip("/") + "/"
            return sorted(
                entry.path for entry in megfile.smart_scandir(root) if entry.is_dir() and entry.path != root
            )
        return sorted(megfile.smart_glob(annotation))

    @staticmethod
    def _deterministic_shuffle_data_index(data_index):
        data_index.sort(key=lambda item: item[0])
        rng = random.Random(42)
        rng.shuffle(data_index)
        return data_index

    def _maybe_slice_episode_by_subtask(self, episode_data_list, frame_index):
        if not episode_data_list:
            return episode_data_list, frame_index, ""
        if self.episode_split_prob <= 0 or random.random() >= self.episode_split_prob:
            return episode_data_list, frame_index, ""

        current = episode_data_list[frame_index]
        target_subtask = refine_text(current.get("robot", {}).get("subtask", "") if isinstance(current, dict) else "")
        if not target_subtask:
            return episode_data_list, frame_index, target_subtask

        left = frame_index
        while left - 1 >= 0:
            previous = episode_data_list[left - 1]
            prev_subtask = refine_text(
                previous.get("robot", {}).get("subtask", "") if isinstance(previous, dict) else ""
            )
            if prev_subtask != target_subtask:
                break
            left -= 1

        right = frame_index
        while right + 1 < len(episode_data_list):
            next_frame = episode_data_list[right + 1]
            next_subtask = refine_text(
                next_frame.get("robot", {}).get("subtask", "") if isinstance(next_frame, dict) else ""
            )
            if next_subtask != target_subtask:
                break
            right += 1

        return episode_data_list[left : right + 1], frame_index - left, target_subtask

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        self._maybe_enable_gc_in_dataloader_worker()
        dataset_len = len(self)
        if dataset_len <= 0:
            raise IndexError(f"{self.__class__.__name__} is empty")

        first_idx = idx
        first_error = None
        failures = []
        max_attempts = self.max_sample_retries + 1
        for attempt in range(max_attempts):
            current_idx = idx if attempt == 0 else random.randint(0, dataset_len - 1)
            try:
                return self.unsafe_getitem(current_idx)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                failures.append(f"idx={current_idx} -> {type(exc).__name__}: {str(exc)[:200]}")
                if attempt + 1 < max_attempts:
                    logger.warning(
                        "Failed to load sample {} on attempt {}/{}: {}",
                        current_idx,
                        attempt + 1,
                        max_attempts,
                        exc,
                    )

        raise RuntimeError(
            f"Failed to load a valid sample after {max_attempts} attempts "
            f"(initial_idx={first_idx}). Recent failures: {'; '.join(failures[-3:])}"
        ) from first_error

    def _get_index_cache(self, data_path: str) -> dict:
        filter_cache_file = os.path.join(data_path, "index-filter-cache.json")
        index_cache_file = os.path.join(data_path, "index_cache.json")
        for cache_file in (filter_cache_file, index_cache_file):
            if megfile.smart_exists(cache_file):
                logger.debug("Loading index cache from {}", cache_file)
                with megfile.smart_open(cache_file, "r") as handle:
                    index_cache = json.load(handle)
                if self._check_index_cache(data_path, index_cache):
                    return index_cache
        return self._build_index_cache(data_path)

    def _build_index_cache(self, data_path: str) -> dict:
        logger.debug("Building index cache for {} ...", data_path)
        jsonl_files = megfile.smart_glob(os.path.join(data_path, "**", "*.jsonl"))
        index_cache = {"meta_data": {"total_samples": 0, "total_jsonl_files": len(jsonl_files)}, "data": {}}
        for jsonl_file in jsonl_files:
            samples = _load_jsonl(jsonl_file)
            index_cache["data"][jsonl_file] = len(samples)
            index_cache["meta_data"]["total_samples"] += len(samples)
        index_cache_file = os.path.join(data_path, "index_cache.json")
        with megfile.smart_open(index_cache_file, "w") as handle:
            json.dump(index_cache, handle, indent=2)
        return index_cache

    def _check_index_cache(self, data_path: str, index_cache: dict) -> bool:
        del data_path, index_cache
        return True

    def _validate_datasets_info(self, datasets_info: list[dict]) -> None:
        errors = []
        for idx, dataset_info in enumerate(datasets_info):
            name = dataset_info.get("name") or dataset_info.get("annotations") or f"index={idx}"
            meta_data = dataset_info.get("meta_data")
            if not isinstance(meta_data, dict):
                errors.append(f"{name}: meta_data must be a dict")
                continue

            if not self.norm_stats_mode and not meta_data.get("text_embed_dir"):
                errors.append(f"{name}: meta_data.text_embed_dir is required")

            arrangement = meta_data.get("state_arrangement")
            if arrangement is None:
                continue
            if not isinstance(arrangement, (list, tuple)) or not arrangement:
                errors.append(f"{name}: meta_data.state_arrangement must be a non-empty list")
                action_dim = None
            else:
                action_dim = len(arrangement) + 1

            for key in ("non_delta_mask", "periodic_mask", "periodic_range"):
                if key not in meta_data:
                    errors.append(f"{name}: meta_data.{key} is missing")

            self._validate_dim_mask(
                errors, name, "non_delta_mask", meta_data.get("non_delta_mask"), action_dim, required=True
            )
            self._validate_dim_mask(
                errors, name, "periodic_mask", meta_data.get("periodic_mask"), action_dim, required=False
            )
            if meta_data.get("periodic_mask") is not None and meta_data.get("periodic_range") is None:
                errors.append(f"{name}: meta_data.periodic_range is required when periodic_mask is set")

            if self.norm_stats_mode:
                continue

            norm_stats_path = meta_data.get("norm_stats_path")
            if not norm_stats_path:
                errors.append(f"{name}: meta_data.norm_stats_path is missing")
                continue
            try:
                if not megfile.smart_exists(str(norm_stats_path)):
                    errors.append(f"{name}: norm_stats_path does not exist: {norm_stats_path}")
                    continue
                with megfile.smart_open(str(norm_stats_path), "r") as handle:
                    norm_stats_payload = json.load(handle)
                norm_stats = norm_stats_payload.get("norm_stats")
                if not isinstance(norm_stats, dict):
                    errors.append(f"{name}: norm_stats_path missing top-level norm_stats")
                    continue
                unexpected_keys = [key for key in norm_stats if key not in {"default", "action", "state"}]
                if unexpected_keys:
                    errors.append(f"{name}: norm_stats has unsupported keys {unexpected_keys}")
                action_stats = norm_stats.get("action")
                if not isinstance(action_stats, dict):
                    errors.append(f"{name}: norm_stats.action is missing")
                elif "q01" not in action_stats or "q99" not in action_stats:
                    errors.append(f"{name}: norm_stats.action must contain q01/q99 for quantile normalization")
                elif action_dim is not None:
                    for stat_name in ("q01", "q99"):
                        values = action_stats.get(stat_name)
                        if isinstance(values, (list, tuple)) and len(values) != action_dim:
                            errors.append(
                                f"{name}: norm_stats.action.{stat_name} dim={len(values)} "
                                f"does not match transformed action dim={action_dim}"
                            )
            except Exception as exc:
                errors.append(f"{name}: failed to read norm_stats_path {norm_stats_path}: {exc}")

        if errors:
            raise ValueError("; ".join(errors[:8]))

    @staticmethod
    def _validate_dim_mask(errors, name: str, key: str, value, action_dim: Optional[int], *, required: bool) -> None:
        if value is None:
            if required:
                errors.append(f"{name}: meta_data.{key} is required")
            return
        if not isinstance(value, (list, tuple)):
            errors.append(f"{name}: meta_data.{key} must be a list")
            return
        if action_dim is None:
            return
        invalid_dims = [dim for dim in value if not isinstance(dim, int) or dim < 0 or dim >= action_dim]
        if invalid_dims:
            errors.append(f"{name}: {key} has invalid dims {invalid_dims} for dim={action_dim}")

    def _min_jsonl_frame_window(self) -> int:
        action_window = self.chunk_size + 1
        future_window = 1
        if self.future_frame_count > 0:
            future_window = 1 + (self.future_frame_count - 1) * self.future_frame_stride
        return max(action_window, future_window)

    def _get_jsonl_frame_window(self) -> int:
        value = os.getenv("DW_JSONL_FRAME_WINDOW", "").strip()
        if not value:
            return 0
        window = int(value)
        if window <= 0:
            return 0
        if self.episode_split_prob > 0:
            logger.warning(
                "DW_JSONL_FRAME_WINDOW disabled because episode_split_prob={} "
                "requires full-episode subtask scanning.",
                self.episode_split_prob,
            )
            return 0
        return max(window, self._min_jsonl_frame_window())

    def _log_jsonl_window_state(self) -> None:
        env_window = os.getenv("DW_JSONL_FRAME_WINDOW", "").strip() or "<unset>"
        if self.jsonl_frame_window > 0:
            logger.warning(
                "JSONL frame window ENABLED: env={} effective_window={} min_required={} "
                "chunk_size={} future_count={} future_stride={}",
                env_window,
                self.jsonl_frame_window,
                self._min_jsonl_frame_window(),
                self.chunk_size,
                self.future_frame_count,
                self.future_frame_stride,
            )
        else:
            logger.warning(
                "JSONL frame window DISABLED: env DW_JSONL_FRAME_WINDOW={} episode_split_prob={} "
                "(full jsonl will be loaded per sample).",
                env_window,
                self.episode_split_prob,
            )

    def _load_jsonl_for_frame(
        self,
        jsonl_file: str,
        file_index: int,
        frame_index: int,
        meta_data: dict,
    ) -> tuple[list, int]:
        pid = os.getpid()
        if self.jsonl_frame_window <= 0:
            if pid not in self._jsonl_fallback_log_pids:
                self._jsonl_fallback_log_pids.add(pid)
                logger.warning(
                    "[jsonl-window] pid={} rank={} falling back to full _load_jsonl "
                "(jsonl_frame_window=0); first sample={}",
                    pid,
                    os.getenv("RANK", ""),
                    jsonl_file,
                )
            return _load_jsonl(jsonl_file, parse=True), frame_index

        episode_len = int(self.file_sample_count_map.get(file_index, 0))
        if episode_len <= 0:
            if pid not in self._jsonl_fallback_log_pids:
                self._jsonl_fallback_log_pids.add(pid)
                logger.warning(
                    "[jsonl-window] pid={} rank={} falling back to full _load_jsonl "
                "(episode_len unknown for file_index={}); first sample={}",
                    pid,
                    os.getenv("RANK", ""),
                    file_index,
                    jsonl_file,
                )
            return _load_jsonl(jsonl_file, parse=True), frame_index

        if frame_index >= episode_len:
            frame_index = random.randint(0, episode_len - 1)
        window_len = min(self.jsonl_frame_window, episode_len)
        window_start = min(frame_index, max(0, episode_len - window_len))
        window_end = min(episode_len, window_start + window_len)
        episode_data_list = _load_jsonl_range(jsonl_file, start_line=window_start, end_line=window_end, parse=True)
        local_frame_index = frame_index - window_start
        if not episode_data_list or local_frame_index >= len(episode_data_list):
            if pid not in self._jsonl_fallback_log_pids:
                self._jsonl_fallback_log_pids.add(pid)
                logger.warning(
                    "[jsonl-window] pid={} rank={} windowed read returned empty/short; "
                "falling back to full _load_jsonl. file={} episode_len={} window=[{},{})",
                    pid,
                    os.getenv("RANK", ""),
                    jsonl_file,
                    episode_len,
                    window_start,
                    window_end,
                )
            return _load_jsonl(jsonl_file, parse=True), frame_index

        if pid not in self._jsonl_window_log_pids:
            self._jsonl_window_log_pids.add(pid)
            logger.warning(
                "[jsonl-window] pid={} rank={} windowed read ACTIVE. file={} "
                "episode_len={} window=[{},{}) local_frame_index={}",
                pid,
                os.getenv("RANK", ""),
                jsonl_file,
                episode_len,
                window_start,
                window_end,
                local_frame_index,
            )

        meta_data["jsonl_window_start"] = window_start
        meta_data["jsonl_window_end"] = window_end
        meta_data["jsonl_episode_length"] = episode_len
        return episode_data_list, local_frame_index

    def build_data_transforms(self):
        if self.norm_stats_mode:
            return self.build_norm_stats_transforms()
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                ArrangeState(),
                AddTerminationState(done_tail_length=15),
                AddAction(predict_length=1),
                AddTrajectory(trajectory_length=self.chunk_size, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ActionNormMultiDataset(use_quantiles=True),
                AddProprioTrajectory(trajectory_length=self.chunk_size, padding_mode="last"),
                LoadImagesWithFutureClip(
                    future_frame_count=self.future_frame_count,
                    future_frame_stride=self.future_frame_stride,
                    future_source_key=self.future_source_key,
                    future_image_size=self.future_image_size,
                ),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                ToList(),
            ]
        )

    def build_norm_stats_transforms(self):
        """Build the action-only DW05 transform chain used to compute norm stats."""
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                ArrangeState(),
                AddTerminationState(done_tail_length=15),
                AddAction(predict_length=1),
                AddTrajectory(trajectory_length=self.chunk_size, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(select_frame=True),
            ]
        )

    def unsafe_getitem(self, idx: int) -> dict:
        def alarm_handler(signum, frame):
            del signum, frame
            raise _SampleTimeout(f"sample timeout >{_SAMPLE_TIMEOUT_S}s idx={idx}")

        prev_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(_SAMPLE_TIMEOUT_S)
        try:
            return self._unsafe_getitem_inner(idx)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

    def _unsafe_getitem_inner(self, idx: int) -> dict:
        dataset_len = len(self)
        if dataset_len <= 0:
            raise IndexError(f"{self.__class__.__name__} is empty")
        if idx >= dataset_len:
            idx = random.randint(0, dataset_len - 1)

        dataset_index, file_index, frame_index = self.global_index[idx]
        jsonl_file = self.file_name_map[file_index]
        dataset_info = self.dataset_map[dataset_index]
        meta_data = copy.deepcopy(dataset_info["meta_data"])
        data_path_prefix = dataset_info["data_path_prefix"]
        episode_data_list, frame_index = self._load_jsonl_for_frame(jsonl_file, file_index, frame_index, meta_data)
        if frame_index >= len(episode_data_list):
            frame_index = random.randint(0, len(episode_data_list) - 1)

        sample_type = episode_data_list[frame_index].get("type")
        if not isinstance(sample_type, list):
            raise ValueError(f"Expected frame type list in {jsonl_file} frame {frame_index}, got {sample_type!r}")
        if self.norm_stats_mode:
            if "action" not in sample_type:
                raise ValueError(f"Norm stats require action samples, got {sample_type!r} in {jsonl_file} frame {frame_index}")
            return self._load_norm_stats_sample(
                jsonl_file=jsonl_file,
                frame_index=frame_index,
                meta_data=meta_data,
                data_path_prefix=data_path_prefix,
                episode_data_list=episode_data_list,
            )

        try:
            result = {}
            shared = None
            if "action" in sample_type:
                action_result, shared = self._load_robot_data(
                    jsonl_file,
                    frame_index,
                    meta_data,
                    data_path_prefix,
                    episode_data_list=episode_data_list,
                    shared=shared,
                )
                if shared is not None:
                    result.update(action_result)
                    if "actions" not in result:
                        raise ValueError(f"No action found in {jsonl_file} frame {frame_index}")

            if "wm" in sample_type:
                wm_result, shared = self._load_wm_data(
                    jsonl_file,
                    frame_index,
                    meta_data,
                    data_path_prefix,
                    episode_data_list=episode_data_list,
                    shared=shared,
                )
                result.update(wm_result)

            if not result:
                raise ValueError(f"Unsupported sample type {sample_type!r} in {jsonl_file} frame {frame_index}")
            result["type"] = sample_type
            result["_episode_data_list"] = episode_data_list
            result["_frame_index"] = frame_index
            robot_task_success = episode_data_list[frame_index].get("robot_task_success", 1)
            return self.output_builder.build(result, robot_task_success=robot_task_success)
        except _SampleTimeout:
            logger.warning(
                "[sample-timeout] idx={} jsonl={} frame={} types={}",
                idx,
                jsonl_file,
                frame_index,
                sample_type,
            )
            raise
        except Exception as exc:
            from traceback import format_exc

            logger.error(
                "[ERROR] Failed to load sample idx={} from {} (frame_index={}): {}",
                idx,
                jsonl_file,
                frame_index,
                exc,
            )
            logger.error(format_exc())
            raise

    def _load_norm_stats_sample(
        self,
        *,
        jsonl_file: str,
        frame_index: int,
        meta_data: dict,
        data_path_prefix: str,
        episode_data_list: list,
    ) -> dict:
        """Return one transformed action sample without loading video or text features."""
        episode_data_list, frame_index, _subtask = self._maybe_slice_episode_by_subtask(episode_data_list, frame_index)
        length_decrease = getattr(self.data_transforms, "predict_length", 0)
        if frame_index >= len(episode_data_list) - length_decrease:
            frame_index = random.randint(0, max(0, len(episode_data_list) - length_decrease - 1))

        meta_data.update(
            {
                "fram_indicies": [frame_index],
                "jsonl_file": jsonl_file,
                "dataset": meta_data.get("dataset", ""),
                "images_keys": self.images_keys,
                "depths_keys": None,
                "load_depth": False,
                "data_path_prefix": data_path_prefix,
            }
        )
        episode_data = self.data_transforms(episode_data_list, meta_data=meta_data)
        if not isinstance(episode_data, dict) or "action" not in episode_data:
            raise ValueError(f"No transformed action available in {jsonl_file} frame {frame_index}")

        sample = {"action": self._float_tensor(episode_data["action"])}
        if episode_data.get("state") is not None:
            sample["state"] = self._float_tensor(episode_data["state"])
        for key in ("action_dim_mask", "action_is_pad"):
            value = episode_data.get(key)
            if value is not None:
                sample[key] = self._bool_tensor(value)
        return sample

    def _prepare_shared_episode(
        self,
        jsonl_file: str,
        frame_index: int,
        meta_data: dict,
        data_path_prefix: str,
        episode_data_list: list,
    ) -> dict:
        episode_data_list, frame_index, subtask = self._maybe_slice_episode_by_subtask(episode_data_list, frame_index)
        length_decrease = getattr(self.data_transforms, "predict_length", 0)
        if frame_index >= len(episode_data_list) - length_decrease:
            frame_index = random.randint(0, max(0, len(episode_data_list) - length_decrease - 1))

        meta_data.update(
            {
                "fram_indicies": [frame_index],
                "jsonl_file": jsonl_file,
                "dataset": meta_data.get("dataset", ""),
                "images_keys": self.images_keys,
                "depths_keys": None,
                "load_depth": False,
                "data_path_prefix": data_path_prefix,
            }
        )
        episode_data = self.data_transforms(episode_data_list, meta_data=meta_data)
        if isinstance(episode_data, list):
            episode_data = episode_data[frame_index]
        episode_data["meta_data"] = meta_data

        images = self._extract_rgb_images(episode_data.get("rgb_data", []))
        images = self._apply_augmentations(images)
        images_tensor = concat_views(images, self.future_image_size)
        return {
            "images": images_tensor,
            "subtask": subtask,
            "episode_data": episode_data,
            "frame_index": frame_index,
            "meta_data": meta_data,
        }

    @staticmethod
    def _extract_rgb_images(rgb_data) -> list[Image.Image]:
        images = []
        for rgb in rgb_data:
            if isinstance(rgb, list):
                for image in rgb:
                    if isinstance(image, Image.Image):
                        images.append(image)
            elif isinstance(rgb, Image.Image):
                images.append(rgb)
            elif isinstance(rgb, dict) and isinstance(rgb.get("data"), Image.Image):
                images.append(rgb["data"])
        return images

    def _load_wm_data(
        self,
        jsonl_file: str,
        frame_index: int,
        meta_data: dict,
        data_path_prefix: str,
        episode_data_list: list | None = None,
        shared: dict | None = None,
    ) -> tuple[dict, dict]:
        if shared is None:
            if episode_data_list is None:
                episode_data_list = _load_jsonl(jsonl_file, parse=True)
            shared = self._prepare_shared_episode(
                jsonl_file,
                frame_index,
                meta_data,
                data_path_prefix,
                episode_data_list,
            )
            episode_data = shared["episode_data"]
            prompt = self.prompt_selector.select(
                episode_data,
                episode_data["worldmodel"]["caption"],
                shared["subtask"],
                shared["meta_data"],
                jsonl_file,
                frame_index,
            )
            context, context_mask = self.text_embedding_cache.load(
                prompt,
                meta_data=shared["meta_data"],
                jsonl_file=jsonl_file,
                frame_index=frame_index,
            )
            context[~context_mask] = 0.0
            tokenization_result = {
                "context": context,
                "context_mask": torch.ones_like(context_mask),
                "prompt": prompt,
                "images": shared["images"],
            }
        else:
            tokenization_result = {}

        data = shared["episode_data"]
        result = {**tokenization_result}
        future_images = data.get("future_images")
        if future_images is not None:
            result["future_images"] = prepare_future_images(
                future_images,
                frame_count=self.future_frame_count,
                output_size=self.future_image_size,
            )
            result["has_future_images"] = True
            result["future_valid_frames"] = int(data.get("future_valid_frames", 0))
        else:
            result["has_future_images"] = False
            result["future_valid_frames"] = 0
        return result, shared

    def _load_robot_data(
        self,
        jsonl_file: str,
        frame_index: int,
        meta_data: dict,
        data_path_prefix: str,
        episode_data_list: list | None = None,
        shared: dict | None = None,
    ) -> tuple[dict, dict | None]:
        if shared is None:
            shared = self._prepare_shared_episode(
                jsonl_file,
                frame_index,
                meta_data,
                data_path_prefix,
                episode_data_list,
            )
        if "robot" not in shared["episode_data"]:
            return {}, None

        episode_data = shared["episode_data"]
        frame_index = shared["frame_index"]
        prompt = self.prompt_selector.select(
            episode_data,
            episode_data["robot"]["prompt"],
            shared["subtask"],
            shared["meta_data"],
            jsonl_file,
            frame_index,
        )
        action = episode_data.get("action")
        if action is None:
            raise ValueError(f"No action found in {jsonl_file} frame {frame_index}")

        context, context_mask = self.text_embedding_cache.load(
            prompt,
            meta_data=shared["meta_data"],
            jsonl_file=jsonl_file,
            frame_index=frame_index,
        )
        context[~context_mask] = 0.0

        result = {
            "context": context,
            "context_mask": torch.ones_like(context_mask),
            "actions": self._float_tensor(action),
            "prompt": prompt,
            "images": shared["images"],
        }
        if episode_data.get("state") is not None:
            result["states"] = self._float_tensor(episode_data["state"])
        if episode_data.get("proprio") is not None:
            result["proprio"] = self._float_tensor(episode_data["proprio"])
        for key in ("action_dim_mask", "action_is_pad", "proprio_is_pad"):
            value = episode_data.get(key)
            if value is not None:
                result[key] = self._bool_tensor(value)
        return result, shared

    @staticmethod
    def _float_tensor(value) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy()).float()
        return torch.as_tensor(value).float()

    @staticmethod
    def _bool_tensor(value) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy()).bool()
        return torch.as_tensor(value, dtype=torch.bool)
