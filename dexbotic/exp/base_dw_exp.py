"""Base experiment entrypoints for Dexbotic generative world methods.

The file follows the layered configuration style used by Dexbotic-open VLA
experiments: small dataclasses own model/data/trainer/inference settings, and a
base experiment class wires the stages together.  Concrete WM/WAM/VLA+WM methods
should keep method-specific defaults in their own ``*_exp.py`` file and inherit
this module only for shared task orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

from dexbotic.exp.utils import (
    mixed_precision_to_model_dtype,
    normalize_mixed_precision,
    save_mp4,
    setup_logging,
)


@dataclass
class Config:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats", "smoke"],
    )
    parser.add_argument(
        "--compute-norm-stats",
        action="store_true",
        help="Shortcut for `--task compute_norm_stats`, matching Dexbotic-open experiment entrypoints.",
    )
    args, unknown = parser.parse_known_args()
    if args.compute_norm_stats:
        args.task = "compute_norm_stats"
    args.overrides = unknown
    return args


def _parse_override_value(value: str):
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def apply_dotlist_overrides(exp, overrides: list[str]) -> None:
    """Apply simple dataclass overrides such as trainer_config.max_steps=20."""
    for override in overrides:
        if "=" not in override:
            continue
        key, raw_value = override.split("=", 1)
        if not key:
            continue
        target = exp
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], _parse_override_value(raw_value))


@dataclass
class DWTrainerConfig(Config):
    """
    Training configuration shared by raw-Accelerate generative trainers.

    The fields stay close to Dexbotic-open trainer configs, but target the
    lighter ``DexboticGenerativeTrainer`` loop used by WM/WAM/VLA+WM models.
    Method experiments can subclass this config to set project names, output
    directories, and method-specific evaluation defaults.
    """

    output_dir: str = field(default="./runs/dw")
    batch_size: int = field(default=2)
    num_workers: int = field(default=4)
    prefetch_factor: int = field(default=2)
    persistent_workers: bool = field(default=False)
    learning_rate: float = field(default=1.0e-4)
    weight_decay: float = field(default=1.0e-2)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1.0e-8)
    lr_scheduler_type: str = field(default="cosine")
    min_lr_ratio: float = field(default=0.01)
    warmup_steps: int = field(default=0)
    num_epochs: int = field(default=5)
    max_steps: Optional[int] = field(default=None)
    log_every: int = field(default=10)
    save_every: int = field(default=2500)
    save_final: bool = field(default=True)
    eval_every: int = field(default=500)
    eval_num_inference_steps: int = field(default=10)
    eval_fps: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    seed: int = field(default=42)
    mixed_precision: str = field(default="bf16")
    resume: Optional[str] = field(default=None)

    wandb_enabled: bool = field(default=False)
    wandb_workspace: Optional[str] = field(default=None)
    wandb_project: str = field(default="dexbotic-wam")
    wandb_name: str = field(default="dw")
    wandb_group: Optional[str] = field(default=None)
    wandb_mode: str = field(default="online")

    def to_trainer_cfg(self):
        """Convert dataclass fields into the OmegaConf object used by trainers."""
        from omegaconf import OmegaConf

        payload = {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "log_every": self.log_every,
            "save_every": self.save_every,
            "save_final": self.save_final,
            "eval_every": self.eval_every,
            "eval_num_inference_steps": self.eval_num_inference_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "resume": self.resume,
            "mixed_precision": self.mixed_precision,
            "lr_scheduler_type": self.lr_scheduler_type,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "eval_fps": self.eval_fps,
            "wandb": {
                "enabled": self.wandb_enabled,
                "workspace": self.wandb_workspace,
                "project": self.wandb_project,
                "name": self.wandb_name,
                "group": self.wandb_group,
                "mode": self.wandb_mode,
            },
        }
        return OmegaConf.create(payload)


@dataclass
class DWDataConfig(Config):
    """
    Base dataset configuration for Dexbotic world-model style experiments.

    Concrete methods should subclass this dataclass and implement
    :meth:`_build_dataset`.  The base class only owns train/validation recipe
    routing, matching the way Dexbotic-open keeps dataset construction behind a
    config object.
    """

    recipe: Optional[str] = field(default=None)
    train_recipe: Optional[str] = field(default=None)
    val_recipe: Optional[str] = field(default=None)
    val_as_train: bool = field(default=False)

    def _build_dataset(self, *, recipe: str, is_val: bool):
        """Build one dataset split for a concrete method."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `_build_dataset()`."
        )

    def build_data(self) -> tuple[object, Optional[object]]:
        """Build train and validation datasets using Dexbotic experiment style."""
        train_recipe = self.train_recipe or self.recipe
        if train_recipe is None:
            raise ValueError(
                f"{self.__class__.__name__}.recipe or train_recipe must be set before building data."
            )
        val_recipe = self.val_recipe or train_recipe
        train_dataset = self._build_dataset(recipe=train_recipe, is_val=False)
        val_dataset = train_dataset if self.val_as_train else self._build_dataset(recipe=val_recipe, is_val=True)
        return train_dataset, val_dataset


@dataclass
class DWInferenceConfig(Config):
    """
    Base inference configuration for generative world-model rollouts.

    Method configs can add fields or override :meth:`load_image_tensor` when the
    condition image needs custom preprocessing.  The default image path keeps a
    single RGB frame normalized to ``[-1, 1]`` and optionally resizes it when
    ``image_size`` is provided.
    """

    checkpoint_path: Optional[str] = field(default=None)
    output_mp4: str = field(default="./runs/dw/inference.mp4")
    input_image_path: Optional[str] = field(default=None)
    prompt: str = field(default="A video recorded from a robot's point of view.")
    device: str = field(default="cuda:0")
    mixed_precision: str = field(default="bf16")
    num_frames: int = field(default=9)
    action_horizon: int = field(default=32)
    num_inference_steps: int = field(default=10)
    seed: Optional[int] = field(default=42)
    tiled: bool = field(default=False)
    image_size: Optional[tuple[int, int]] = field(default=None)
    fps: int = field(default=8)
    test_action_with_infer_action: bool = field(default=False)
    load_text_encoder: bool = field(default=True)
    skip_dit_load_from_pretrain: bool = field(default=True)

    def load_image_tensor(self, model) -> torch.Tensor:
        """Load and normalize a single RGB condition image for inference."""
        if self.input_image_path is None:
            raise ValueError("`input_image_path` is required for WAM inference.")
        image = Image.open(self.input_image_path).convert("RGB")
        if self.image_size is not None:
            target_h, target_w = self.image_size
            image = image.resize((target_w, target_h), resample=Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=model.device, dtype=model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0


@dataclass
class BaseDWExp(Config):
    """
    Base experiment wrapper for Dexbotic world-model style methods.

    The class owns task dispatch and high-level wiring only: model construction,
    dataset construction, trainer setup, and inference output writing.  Concrete
    experiments provide method-specific config dataclasses in the same way
    Dexbotic-open VLA experiments specialize ``BaseExp``.
    """

    model_config: Config = field(default_factory=Config)
    trainer_config: DWTrainerConfig = field(default_factory=DWTrainerConfig)
    data_config: DWDataConfig = field(default_factory=DWDataConfig)
    inference_config: DWInferenceConfig = field(default_factory=DWInferenceConfig)
    norm_stats_config: Optional[Config] = field(default=None)
    logger_level: str = field(default="INFO")
    nccl_heartbeat_timeout_sec: Optional[int] = field(default=7200)
    trainer_cls: ClassVar[type] = None

    def __post_init__(self):
        logger.remove()
        logger.add(sys.stdout, level=self.logger_level)

    def _setup_common_env(self) -> None:
        """Set environment defaults shared by generative training jobs."""
        if self.nccl_heartbeat_timeout_sec is not None:
            os.environ.setdefault(
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
                str(int(self.nccl_heartbeat_timeout_sec)),
            )

    def _prepare_model_config(self, *, for_inference: bool = False):
        """Return the model config used by ``build_model``.

        Subclasses can deepcopy and inject method-local runtime settings here.
        The base implementation deliberately does not know about checkpoint
        paths, model-cache roots, or architecture-specific load switches.
        """
        del for_inference
        return self.model_config

    def _resolve_train_device(self) -> str:
        """Resolve the local CUDA device for single-process or distributed runs."""
        if not torch.cuda.is_available():
            return "cpu"
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return "cuda:0"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= device_count:
            return "cuda:0"
        return f"cuda:{local_rank}"

    def build_model(self, *, for_inference: bool = False):
        """Build a model with train/inference precision and device settings."""
        model_config = self._prepare_model_config(for_inference=for_inference)
        if not hasattr(model_config, "build_model"):
            raise ValueError(
                f"{self.__class__.__name__}.model_config must provide build_model()."
            )
        mixed_precision = (
            self.inference_config.mixed_precision
            if for_inference
            else self.trainer_config.mixed_precision
        )
        model_dtype = mixed_precision_to_model_dtype(
            normalize_mixed_precision(mixed_precision)
        )
        device = self.inference_config.device if for_inference else self._resolve_train_device()
        return model_config.build_model(model_dtype=model_dtype, device=device)

    def train(self):
        """Run training with the method-specific generative trainer."""
        self._setup_common_env()
        trainer_cfg = self.trainer_config.to_trainer_cfg()
        setup_logging(log_level=self.logger_level)
        Path(self.trainer_config.output_dir).mkdir(parents=True, exist_ok=True)
        if self.trainer_cls is None:
            raise ValueError(
                f"{self.__class__.__name__}.trainer_cls is not set. "
                "Concrete DW experiments must bind a trainer class."
            )

        model = self.build_model(for_inference=False)
        train_dataset, val_dataset = self.data_config.build_data()
        trainer = self.trainer_cls(
            cfg=trainer_cfg,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        trainer.train()

    def compute_norm_stats(self):
        """Compute method-specific normalization statistics from the data config."""
        self._setup_common_env()
        setup_logging(log_level=self.logger_level)
        compute_fn = getattr(self.data_config, "compute_norm_stats", None)
        if not callable(compute_fn):
            raise ValueError(
                f"{self.__class__.__name__}.data_config must provide compute_norm_stats() "
                "for `--task compute_norm_stats`."
            )
        if self.norm_stats_config is None:
            return compute_fn()
        return compute_fn(self.norm_stats_config)

    @torch.no_grad()
    def inference(self):
        """Run generative inference and save the video/action outputs."""
        self._setup_common_env()
        setup_logging(log_level=self.logger_level)
        model = self.build_model(for_inference=True)
        checkpoint_path = self.inference_config.checkpoint_path
        if checkpoint_path:
            logger.info(f"Loading WAM checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)
        model.eval()

        input_image = self.inference_config.load_image_tensor(model)
        result = model.infer(
            prompt=self.inference_config.prompt,
            input_image=input_image,
            num_frames=self.inference_config.num_frames,
            action_horizon=self.inference_config.action_horizon,
            num_inference_steps=self.inference_config.num_inference_steps,
            seed=self.inference_config.seed,
            tiled=self.inference_config.tiled,
            test_action_with_infer_action=self.inference_config.test_action_with_infer_action,
        )
        save_mp4(result["video"], self.inference_config.output_mp4, fps=self.inference_config.fps)
        action_path = Path(self.inference_config.output_mp4).with_suffix(".action.json")
        action = result.get("action")
        if action is not None:
            action_path.write_text(json.dumps(action.tolist(), indent=2))
        logger.info(
            f"WAM inference finished in {self.inference_config.output_mp4}; "
            f"action={action_path if action is not None else 'none'}"
        )
        return result

    def smoke(self):
        """Load one dataset sample and log the key tensor shapes."""
        self._setup_common_env()
        logger.info("Building DW dataset for one-sample smoke test.")
        train_dataset, _ = self.data_config.build_data()
        t0 = time.monotonic()
        sample = train_dataset[0]
        summary = {
            key: tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            for key, value in sample.items()
            if key in {"video", "action", "context", "context_mask", "proprio", "image_is_pad", "action_is_pad"}
        }
        logger.info(f"DW dataset smoke sample loaded in {time.monotonic() - t0:.2f}s: {summary}")
        return sample


if __name__ == "__main__":
    raise SystemExit(
        "base_dw_exp.py defines shared experiment scaffolding. "
        "Run a concrete method experiment from dexbotic/exp or playground/."
    )
