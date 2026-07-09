"""Utilities for resolving DW05 data source definitions."""

from __future__ import annotations

import copy
import os
from typing import Any

from loguru import logger

from dexbotic.data.dataset.dw05.data_source import DATASETS, MIX_DATA


DW05_ROBOTWIN_ANNOTATIONS_ENV = "DW05_ROBOTWIN_ANNOTATIONS"
DW05_ROBOTWIN_TEXT_EMBED_DIR_ENV = "DW05_ROBOTWIN_TEXT_EMBED_DIR"
DW05_ROBOTWIN_NORM_STATS_PATH_ENV = "DW05_ROBOTWIN_NORM_STATS_PATH"
DW05_ROBOTWIN_DATA_PATH_PREFIX_ENV = "DW05_ROBOTWIN_DATA_PATH_PREFIX"
DW05_ROBOTWIN_INDEX_PATH_PREFIX_ENV = "DW05_ROBOTWIN_INDEX_PATH_PREFIX"
DW05_ROBOTWIN_MEDIA_PATH_RESOLVER_ENV = "DW05_ROBOTWIN_MEDIA_PATH_RESOLVER"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge dictionaries without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_recipe_entries(entries: Any) -> list[dict[str, Any]]:
    """Normalize recipe entries to dictionaries with ``name`` and ``frequency``."""
    if not isinstance(entries, list):
        raise TypeError(f"recipe entries must be a list, got {type(entries)!r}")

    normalized = []
    for item in entries:
        if isinstance(item, str):
            normalized.append({"name": item, "frequency": 1.0})
            continue
        if isinstance(item, (list, tuple)) and item:
            normalized.append(
                {
                    "name": item[0],
                    "frequency": item[1] if len(item) >= 2 else 1.0,
                }
            )
            continue
        if isinstance(item, dict):
            entry = copy.deepcopy(item)
            if "frequency" not in entry:
                for alias in ("ratio", "weight", "freq"):
                    if alias in entry:
                        entry["frequency"] = entry[alias]
                        break
            entry.setdefault("frequency", 1.0)
            normalized.append(entry)
            continue
        raise TypeError(f"unsupported recipe entry: {item!r}")
    return normalized


def _static_recipe_entries(recipe: str | dict | list) -> list[dict[str, Any]]:
    """Resolve a named recipe or a '+'-separated list of dataset names."""
    if isinstance(recipe, dict):
        return _normalize_recipe_entries([recipe])
    if isinstance(recipe, list):
        return _normalize_recipe_entries(recipe)
    if recipe in MIX_DATA:
        entries = MIX_DATA[recipe]
    else:
        entries = [(name, 1.0) for name in str(recipe).split("+") if name]
    return _normalize_recipe_entries(entries)


def _env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_registered_dataset(
    name: str,
    *,
    require_text_embed_dir: bool = True,
    require_norm_stats_path: bool = True,
) -> dict[str, Any]:
    """Resolve a built-in dataset template from explicit environment variables."""
    info = copy.deepcopy(DATASETS[name])
    annotations = _env_or_none(DW05_ROBOTWIN_ANNOTATIONS_ENV) or info.get("annotations")
    text_embed_dir = _env_or_none(DW05_ROBOTWIN_TEXT_EMBED_DIR_ENV) or info.get("meta_data", {}).get("text_embed_dir")
    norm_stats_path = _env_or_none(DW05_ROBOTWIN_NORM_STATS_PATH_ENV) or info.get("meta_data", {}).get("norm_stats_path")

    missing = []
    if not annotations:
        missing.append(DW05_ROBOTWIN_ANNOTATIONS_ENV)
    if require_text_embed_dir and not text_embed_dir:
        missing.append(DW05_ROBOTWIN_TEXT_EMBED_DIR_ENV)
    if require_norm_stats_path and not norm_stats_path:
        missing.append(DW05_ROBOTWIN_NORM_STATS_PATH_ENV)
    if missing:
        raise ValueError(
            f"Built-in DW05 dataset {name!r} requires explicit path configuration: {missing}. "
            "Set these environment variables or pass local paths through "
            "DW05DataConfig(annotations=..., text_embedding_cache_dir=..., norm_stats_path=...)."
        )

    info["annotations"] = annotations
    info["meta_data"]["text_embed_dir"] = text_embed_dir
    info["meta_data"]["norm_stats_path"] = norm_stats_path
    data_path_prefix = _env_or_none(DW05_ROBOTWIN_DATA_PATH_PREFIX_ENV)
    index_path_prefix = _env_or_none(DW05_ROBOTWIN_INDEX_PATH_PREFIX_ENV)
    media_path_resolver = _env_or_none(DW05_ROBOTWIN_MEDIA_PATH_RESOLVER_ENV)
    if data_path_prefix is not None:
        info["data_path_prefix"] = data_path_prefix
    if index_path_prefix is not None:
        info["index_path_prefix"] = index_path_prefix
    if media_path_resolver is not None:
        info["media_path_resolver"] = media_path_resolver
    return info


def _build_dataset_info(
    entry: dict[str, Any],
    *,
    require_text_embed_dir: bool = True,
    require_norm_stats_path: bool = True,
) -> dict[str, Any] | None:
    name = str(entry.get("name") or entry.get("dataset") or "").strip()
    if not name:
        raise ValueError(f"recipe entry missing dataset name: {entry!r}")

    frequency = float(entry.get("frequency", 1.0))
    if frequency <= 0:
        logger.info("Skipping dataset {} because frequency={} <= 0", name, frequency)
        return None

    if name in DATASETS:
        info = _resolve_registered_dataset(
            name,
            require_text_embed_dir=require_text_embed_dir,
            require_norm_stats_path=require_norm_stats_path,
        )
    elif "annotations" in entry and "meta_data" in entry:
        info = {
            "annotations": entry["annotations"],
            "meta_data": entry["meta_data"],
            "frequency": 1.0,
        }
    else:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; available Robotwin datasets: {known}")

    overrides = {
        key: value
        for key, value in entry.items()
        if key not in {"name", "dataset", "frequency", "ratio", "weight", "freq"}
    }
    if overrides:
        info = _deep_merge(info, overrides)
    info["name"] = name
    info["frequency"] = frequency
    return info


def resolve_datasets_info(recipe: str | dict | list, **kwargs) -> list[dict[str, Any]]:
    """Resolve a Robotwin recipe into dataset info dictionaries.

    Args:
        recipe: A key in :data:`MIX_DATA`, a '+'-separated list of Robotwin
            dataset names, or inline entry dictionaries with ``annotations`` and
            ``meta_data``. Inline entries are the preferred path for local data
            experiments because they avoid built-in remote registry defaults.
        **kwargs: Accepted for backward compatibility with older resolver
            call sites. These values are ignored by the static Robotwin
            registry.

    Returns:
        A non-empty list of dataset info dictionaries consumed by
        :class:`dexbotic.data.dataset.dw05.dw_dataset.DWDataset`.
    """
    require_text_embed_dir = bool(kwargs.pop("require_text_embed_dir", True))
    require_norm_stats_path = bool(kwargs.pop("require_norm_stats_path", True))
    ignored = {key: value for key, value in kwargs.items() if value not in (None, False, 0)}
    if ignored:
        logger.warning("Ignoring unsupported DW05 data source resolver kwargs: {}", sorted(ignored))

    result = []
    for entry in _static_recipe_entries(recipe):
        info = _build_dataset_info(
            entry,
            require_text_embed_dir=require_text_embed_dir,
            require_norm_stats_path=require_norm_stats_path,
        )
        if info is not None:
            result.append(info)
    if not result:
        raise ValueError(f"recipe {recipe!r} resolved to no datasets")
    return result


__all__ = ["resolve_datasets_info"]
