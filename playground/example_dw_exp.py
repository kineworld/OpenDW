import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexbotic.exp.base_dw_exp import apply_dotlist_overrides
from dexbotic.exp.dw05_exp import (
    DW05DataConfig as _DW05DataConfig,
    DW05Exp as _DW05Exp,
    DW05InferenceConfig as _DW05InferenceConfig,
    DW05NormStatsConfig as _DW05NormStatsConfig,
    DW05TrainerConfig as _DW05TrainerConfig,
)
from dexbotic.model.dw05 import DW05ModelConfig as _DW05ModelConfig


def parse_args():
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
        help="Shortcut for `--task compute_norm_stats`.",
    )
    args, unknown = parser.parse_known_args()
    if args.compute_norm_stats:
        args.task = "compute_norm_stats"
    args.overrides = unknown
    return args


@dataclass
class DWTrainerConfig(_DW05TrainerConfig):
    output_dir: str = field(
        default=f"./user_checkpoints/{datetime.now().strftime('%m%d')}-dw05"
    )
    batch_size: int = field(default=2)
    num_workers: int = field(default=4)
    max_steps: int | None = field(default=None)
    wandb_project: str = field(default="dexbotic_wam")


@dataclass
class DWDataConfig(_DW05DataConfig):
    recipe: str = field(default="robotwin_baseline")
    val_as_train: bool = field(default=False)


@dataclass
class DW05ModelConfig(_DW05ModelConfig):
    architecture: str = field(default="dw05")


@dataclass
class DWInferenceConfig(_DW05InferenceConfig):
    checkpoint_path: str | None = field(default=None)
    input_image_path: str | None = field(default=None)
    output_mp4: str = field(default="./runs/dw05/inference.mp4")


@dataclass
class DWNormStatsConfig(_DW05NormStatsConfig):
    norm_save_path: str = field(default="./runs/dw05/norm_stats")


@dataclass
class DWExp(_DW05Exp):
    model_config: DW05ModelConfig = field(default_factory=DW05ModelConfig)
    trainer_config: DWTrainerConfig = field(default_factory=DWTrainerConfig)
    data_config: DWDataConfig = field(default_factory=DWDataConfig)
    inference_config: DWInferenceConfig = field(default_factory=DWInferenceConfig)
    norm_stats_config: DWNormStatsConfig = field(default_factory=DWNormStatsConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = DWExp()
    apply_dotlist_overrides(exp, args.overrides)
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
    elif args.task == "smoke":
        exp.smoke()
