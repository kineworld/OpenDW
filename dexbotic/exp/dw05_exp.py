"""DW05 experiment definition.

This mirrors Dexbotic-open's method experiment style: the base experiment owns
the high-level train/inference wiring, while this module binds model, data,
trainer, inference defaults, and local runtime paths for one concrete method.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from dexbotic.exp.base_dw_exp import (
    BaseDWExp,
    Config,
    DWDataConfig,
    DWInferenceConfig,
    DWTrainerConfig,
    apply_dotlist_overrides,
    parse_args,
)
from dexbotic.exp.dw05_trainer import DW05Trainer
from dexbotic.data.dataset.dw05 import DWDataset
from dexbotic.data.dataset.dw05.data_source import ROBOTWIN_META
from dexbotic.data.dataset.dw05.transform import normalize
from dexbotic.model.dw05 import DW05ModelConfig


DW05_MODEL_BASE_PATH_ENV = "DW05_MODEL_BASE_PATH"
DW05_ACTION_DIT_PRETRAINED_PATH_ENV = "DW05_ACTION_DIT_PRETRAINED_PATH"
DIFFSYNTH_MODEL_BASE_PATH_ENV = "DIFFSYNTH_MODEL_BASE_PATH"
DW05_ACTION_DIT_FILENAME = "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"


def default_dw05_model_base_path() -> Optional[str]:
    """Resolve the optional local Wan2.2 cache root for DW05 experiments."""
    return os.environ.get(DW05_MODEL_BASE_PATH_ENV) or os.environ.get(DIFFSYNTH_MODEL_BASE_PATH_ENV)


def default_dw05_action_dit_pretrained_path() -> Optional[str]:
    """Resolve an explicitly configured ActionDiT initialization checkpoint."""
    return os.environ.get(DW05_ACTION_DIT_PRETRAINED_PATH_ENV)


def resolve_dw05_action_dit_pretrained_path(model_base_path: Optional[str]) -> Optional[str]:
    """Derive DW05's ActionDiT checkpoint path from an experiment model root."""
    explicit_path = os.environ.get(DW05_ACTION_DIT_PRETRAINED_PATH_ENV)
    if explicit_path is not None:
        return explicit_path or None
    if not model_base_path:
        return None
    return str(Path(model_base_path) / DW05_ACTION_DIT_FILENAME)


@dataclass
class DW05TrainerConfig(DWTrainerConfig):
    """Training defaults for the DW05 Wan2.2 world-action method."""

    output_dir: str = field(default="./runs/dw05")
    wandb_name: str = field(default="dw05")


@dataclass
class DW05NormStatsConfig(Config):
    """Settings for computing DW05 action normalization statistics."""

    recipe: Optional[str] = field(default=None)
    norm_save_path: str = field(default="./runs/dw05/norm_stats")
    batch_size: int = field(default=128)
    num_workers: int = field(default=8)
    prefetch_factor: int = field(default=2)
    persistent_workers: bool = field(default=False)
    max_batches: Optional[int] = field(default=500)
    shuffle: bool = field(default=True)
    seed: int = field(default=42)
    norm_keys: list[str] = field(default_factory=lambda: ["action"])


@dataclass
class DW05DataConfig(DWDataConfig):
    """Dataset defaults for DW05 robot-action and world-model samples."""

    recipe: Optional[str] = field(default="robotwin_baseline")
    num_frames: int = field(default=33)
    video_size: tuple[int, int] = field(default=(384, 320))
    action_video_freq_ratio: int = field(default=4)
    concat_multi_camera: str = field(default="robotwin")
    images_keys: list[str] = field(default_factory=lambda: ["images_1", "images_2", "images_3"])
    context_len: int = field(default=128)
    enc_id: str = field(default="wan22ti2v5b")
    action_type: str = field(default="state")
    pretrained_norm_stats: Optional[str] = field(default=None)
    annotations: Optional[str | list[str]] = field(default=None)
    data_path_prefix: Optional[str] = field(default=None)
    index_path_prefix: Optional[str] = field(default=None)
    media_path_resolver: Optional[str] = field(default=None)
    text_embedding_cache_dir: Optional[str] = field(default=None)
    norm_stats_path: Optional[str] = field(default=None)
    empty_text_embed_path: Optional[str] = field(default=None)
    dataset_meta_overrides: dict | None = field(default=None)
    recipe_entries: Optional[list[dict] | dict] = field(default=None)
    missing_text_embedding: str = field(default="zero")
    text_embedding_dim: int = field(default=4096)
    prompt_add_prob: float = field(default=0.3)
    check_index_cache: bool = field(default=True)
    max_sample_retries: int = field(default=10)
    episode_split_prob: float = field(default=0.0)
    aug_policy: Optional[str | list[str]] = field(default=None)

    def _build_recipe_entries(self):
        """Return inline recipe entries when the experiment supplies data paths."""
        if self.recipe_entries is not None:
            return self.recipe_entries
        if self.annotations is None:
            return None

        meta_overrides = dict(self.dataset_meta_overrides or {})
        meta = copy.deepcopy(ROBOTWIN_META)
        if self.text_embedding_cache_dir:
            meta["text_embed_dir"] = self.text_embedding_cache_dir
        norm_stats_path = self.norm_stats_path or self.pretrained_norm_stats
        if norm_stats_path:
            meta["norm_stats_path"] = norm_stats_path
        meta.update({key: value for key, value in meta_overrides.items() if value is not None})
        return [
            {
                "name": "dw05_local",
                "annotations": self.annotations,
                "data_path_prefix": self.data_path_prefix or "",
                "index_path_prefix": self.index_path_prefix or "",
                "media_path_resolver": self.media_path_resolver,
                "meta_data": meta,
                "frequency": 1.0,
            }
        ]

    def _build_dataset(self, *, recipe: str, is_val: bool) -> DWDataset:
        """Build a DW05 dataset split for the selected recipe."""
        del is_val
        return self._make_dataset(recipe=recipe, norm_stats_mode=False)

    def _make_dataset(self, *, recipe: str, norm_stats_mode: bool) -> DWDataset:
        """Build a DW05 dataset with shared constructor settings."""
        return DWDataset(
            num_frames=self.num_frames,
            video_size=self.video_size,
            action_video_freq_ratio=self.action_video_freq_ratio,
            concat_multi_camera=self.concat_multi_camera,
            images_keys=self.images_keys,
            context_len=self.context_len,
            enc_id=self.enc_id,
            action_type=self.action_type,
            recipe=recipe,
            pretrained_norm_stats=self.pretrained_norm_stats,
            missing_text_embedding=self.missing_text_embedding,
            text_embedding_dim=self.text_embedding_dim,
            prompt_add_prob=self.prompt_add_prob,
            check_index_cache=self.check_index_cache,
            recipe_entries=self._build_recipe_entries(),
            max_sample_retries=self.max_sample_retries,
            episode_split_prob=self.episode_split_prob,
            aug_policy=self.aug_policy,
            text_embedding_cache_dir=self.text_embedding_cache_dir,
            empty_text_embed_path=self.empty_text_embed_path,
            norm_stats_mode=norm_stats_mode,
        )

    def build_norm_dataset(self, norm_config: DW05NormStatsConfig) -> DWDataset:
        """Build the action-only dataset used by ``compute_norm_stats``."""
        recipe = norm_config.recipe or self.train_recipe or self.recipe
        if recipe is None:
            raise ValueError("DW05DataConfig.recipe or train_recipe must be set before computing norm stats.")
        return self._make_dataset(recipe=recipe, norm_stats_mode=True)

    def compute_norm_stats(self, norm_config: DW05NormStatsConfig | None = None) -> str:
        """Compute and save DW05 normalization statistics for configured data."""
        norm_config = norm_config or DW05NormStatsConfig()
        dataset = self.build_norm_dataset(norm_config)
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=norm_config.batch_size,
            shuffle=norm_config.shuffle,
            num_workers=norm_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator(device="cpu").manual_seed(int(norm_config.seed)),
        )
        if norm_config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = norm_config.prefetch_factor
            loader_kwargs["persistent_workers"] = norm_config.persistent_workers
        dataloader = DataLoader(**loader_kwargs)

        stats = {key: normalize.RunningStats() for key in norm_config.norm_keys}
        max_batches = norm_config.max_batches
        logger.info(
            "Computing DW05 norm stats: samples={} batch_size={} max_batches={} keys={} save_path={}",
            len(dataset),
            norm_config.batch_size,
            max_batches,
            norm_config.norm_keys,
            norm_config.norm_save_path,
        )
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            for key in norm_config.norm_keys:
                if key not in batch:
                    raise KeyError(f"Cannot compute DW05 norm stats: batch missing key {key!r}.")
                values = self._flatten_norm_values(batch[key], key=key, batch=batch)
                if values.size == 0:
                    continue
                stats[key].update(values)
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                logger.info("Processed {} DW05 norm-stat batches", batch_idx + 1)

        norm_stats = {key: running.get_statistics() for key, running in stats.items()}
        normalize.save(norm_config.norm_save_path, norm_stats)
        norm_file_path = str(Path(norm_config.norm_save_path) / "norm_stats.json")
        self.norm_stats_path = norm_file_path
        logger.info("Saved DW05 norm stats to {}", norm_file_path)
        return norm_file_path

    @staticmethod
    def _flatten_norm_values(value: torch.Tensor, *, key: str, batch: dict) -> np.ndarray:
        """Flatten batched DW05 tensors into ``[N, D]`` arrays for RunningStats."""
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.detach().float().cpu()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        elif value.ndim > 2:
            if key == "action" and "action_is_pad" in batch:
                valid = ~batch["action_is_pad"].detach().cpu().bool()
                value = value[valid]
            else:
                value = value.reshape(-1, value.shape[-1])
        return value.numpy()


@dataclass
class DW05InferenceConfig(DWInferenceConfig):
    """Inference defaults for DW05 open-loop video/action rollout."""

    output_mp4: str = field(default="./runs/dw05/inference.mp4")
    image_size: Optional[tuple[int, int]] = field(default=(384, 320))


@dataclass
class DW05Exp(BaseDWExp):
    """Concrete Dexbotic experiment for the DW05 world-action model."""

    trainer_cls = DW05Trainer
    model_config: DW05ModelConfig = field(default_factory=DW05ModelConfig)
    trainer_config: DW05TrainerConfig = field(default_factory=DW05TrainerConfig)
    data_config: DW05DataConfig = field(default_factory=DW05DataConfig)
    inference_config: DW05InferenceConfig = field(default_factory=DW05InferenceConfig)
    norm_stats_config: DW05NormStatsConfig = field(default_factory=DW05NormStatsConfig)
    dw05_model_base_path: Optional[str] = field(default_factory=default_dw05_model_base_path)
    dw05_action_dit_pretrained_path: Optional[str] = field(default_factory=default_dw05_action_dit_pretrained_path)

    def _setup_common_env(self) -> None:
        """Set shared training env plus DW05's optional local model cache root."""
        super()._setup_common_env()
        if self.dw05_model_base_path:
            os.environ.setdefault(DIFFSYNTH_MODEL_BASE_PATH_ENV, self.dw05_model_base_path)

    def _prepare_model_config(self, *, for_inference: bool = False) -> DW05ModelConfig:
        """Inject DW05 experiment runtime settings into a model config copy."""
        model_config = copy.deepcopy(self.model_config)
        if for_inference:
            model_config.load_text_encoder = bool(self.inference_config.load_text_encoder)
            model_config.skip_dit_load_from_pretrain = bool(self.inference_config.skip_dit_load_from_pretrain)
            if model_config.skip_dit_load_from_pretrain:
                model_config.action_dit_pretrained_path = None

        should_fill_action_dit = not (for_inference and model_config.skip_dit_load_from_pretrain)
        if should_fill_action_dit and model_config.action_dit_pretrained_path is None:
            action_dit_path = self.dw05_action_dit_pretrained_path
            if action_dit_path is None:
                action_dit_path = resolve_dw05_action_dit_pretrained_path(self.dw05_model_base_path)
            model_config.action_dit_pretrained_path = action_dit_path
        return model_config


__all__ = [
    "DIFFSYNTH_MODEL_BASE_PATH_ENV",
    "DW05_ACTION_DIT_FILENAME",
    "DW05_ACTION_DIT_PRETRAINED_PATH_ENV",
    "DW05_MODEL_BASE_PATH_ENV",
    "DW05DataConfig",
    "DW05Exp",
    "DW05InferenceConfig",
    "DW05NormStatsConfig",
    "DW05TrainerConfig",
    "default_dw05_action_dit_pretrained_path",
    "default_dw05_model_base_path",
    "resolve_dw05_action_dit_pretrained_path",
]


if __name__ == "__main__":
    args = parse_args()
    exp = DW05Exp()
    apply_dotlist_overrides(exp, args.overrides)
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
    elif args.task == "smoke":
        exp.smoke()
