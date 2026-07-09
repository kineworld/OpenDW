"""Robotwin-only data source registry for DW05.

This module intentionally keeps path-like fields empty. Users should provide
local or remote data locations through ``DW05DataConfig`` or the environment
variables resolved in :mod:`dexbotic.data.dataset.dw05.data_source_utils`.
"""

from __future__ import annotations

from typing import Any


ROBOTWIN_META: dict[str, Any] = {
    "state_arrangement": [0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10, 11, 12, -1, 13],
    "non_delta_mask": [7, 15],
    "text_embed_dir": None,
    "norm_stats_path": None,
    "periodic_mask": None,
    "periodic_range": None,
    "action_type": "delta_first_frame",
    "prompt_format": None,
}


DATASETS: dict[str, dict[str, Any]] = {
    "robotwin2_nofilter_clean_random": {
        "annotations": None,
        "meta_data": ROBOTWIN_META,
        "frequency": 1.0,
    }
}


MIX_DATA: dict[str, list[tuple[str, float]]] = {
    "robotwin_baseline": [
        ("robotwin2_nofilter_clean_random", 1.0)
    ],
}


__all__ = [
    "ROBOTWIN_META",
    "DATASETS",
    "MIX_DATA",
]
