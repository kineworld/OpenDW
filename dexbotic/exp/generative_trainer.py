"""Generic raw-Accelerate trainer for Dexbotic generative model families.

This trainer is the WM/WAM/VLA+WM counterpart to the HuggingFace-based
``DexboticTrainer`` used by VLA/CausalLM models.  It owns only the mechanics that
are common to generative world-style models: dataloader construction, optimizer
and scheduler setup, gradient accumulation, logging, checkpoint state, and
resume.  Method-specific choices such as which modules are trainable and how to
evaluate rollouts live in subclasses.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
import os
from math import ceil
from pathlib import Path
import re
import time
from typing import Optional

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from dexbotic.exp.utils import ResumableEpochSampler, set_global_seed


logger = logging.getLogger(__name__)


def _union_collate(batch: list) -> dict:
    """Collate samples with optional keys by filling missing tensor flags."""
    batch = [dict(sample) for sample in batch]
    all_keys = {key for sample in batch for key in sample}
    for sample in batch:
        for key in all_keys - sample.keys():
            ref = next(other[key] for other in batch if key in other)
            sample[key] = torch.zeros_like(ref, dtype=torch.bool) if isinstance(ref, torch.Tensor) else ref
    return default_collate(batch)


class DexboticGenerativeTrainer:
    """Base trainer for Dexbotic WM, WAM, and VLA+WM generative methods.

    Concrete subclasses should usually override only a few hooks:

    - :meth:`configure_trainable_modules` for method-specific freezing;
    - :meth:`get_trainable_parameters` when trainable parameters are not simply
      all ``requires_grad=True`` parameters;
    - :meth:`evaluate`, :meth:`format_eval_log`, and
      :meth:`build_eval_wandb_payload` for method-specific validation.

    Models are expected to expose ``training_loss(sample) -> (loss, metrics)``
    and ``save_checkpoint/load_checkpoint``.  This is intentionally separate
    from the HuggingFace ``Trainer`` path used by language-model VLA methods.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
        accelerator: Accelerator | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.adam_beta1 = float(cfg.get("adam_beta1", 0.9))
        self.adam_beta2 = float(cfg.get("adam_beta2", 0.95))
        self.adam_epsilon = float(cfg.get("adam_epsilon", 1.0e-8))
        self.min_lr_ratio = float(cfg.get("min_lr_ratio", 0.01))
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.prefetch_factor = int(cfg.prefetch_factor) if cfg.get("prefetch_factor") is not None else 2
        self.persistent_workers = bool(cfg.get("persistent_workers", False))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_final = bool(cfg.get("save_final", True))
        self.eval_every = int(cfg.eval_every)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self._log_accelerator_state()

        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        self.configure_trainable_modules(self.model)
        trainable_params = list(self.get_trainable_parameters(self.model))
        if not trainable_params:
            raise ValueError(
                f"{self.__class__.__name__} found no trainable parameters. "
                "Check the model freeze policy or `get_trainable_parameters()`."
            )
        self.optimizer = self.build_optimizer(trainable_params)

        self.train_loader = self.build_train_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        self._load_weights_before_prepare()

        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(cfg.get("warmup_steps", int(total_train_steps * 0.05)))
        logger.info("warmup_steps=%d (total_train_steps=%d)", warmup_steps, total_train_steps)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _log_accelerator_state(self) -> None:
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = "none"
        if deepspeed_plugin is not None:
            zero_stage = deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d "
            "cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        import torch.distributed as dist

        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "world_size check: accelerator.num_processes=%d dist.get_world_size=%d dist.is_initialized=%s",
            self.accelerator.num_processes,
            dist.get_world_size() if dist.is_initialized() else -1,
            dist.is_initialized(),
        )

    def configure_trainable_modules(self, model: nn.Module) -> None:
        """Apply method-specific freeze/train mode before optimizer creation."""
        return None

    def get_trainable_parameters(self, model: nn.Module) -> Iterable[nn.Parameter]:
        """Return parameters to optimize for a generative model."""
        getter = getattr(model, "get_trainable_parameters", None)
        if callable(getter):
            return getter()
        return (parameter for parameter in model.parameters() if parameter.requires_grad)

    def build_optimizer(self, trainable_params: list[nn.Parameter]) -> torch.optim.Optimizer:
        """Build the optimizer used by the generic raw-Accelerate loop."""
        return torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
        )

    def set_train_mode(self) -> None:
        """Restore train mode after eval or at the beginning of training."""
        self.accelerator.unwrap_model(self.model).train()

    def compute_training_loss(self, sample) -> tuple[torch.Tensor, dict]:
        """Run the model-specific generative loss function."""
        train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
        return train_model.training_loss(sample)

    def build_train_loader(self, dataset, worker_init_fn=None) -> DataLoader:
        """Build the resumable distributed training loader."""
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=_union_collate,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(dataset, **loader_kwargs)

    def maybe_reload_train_dataset(self) -> bool:
        """Return whether the train dataset was refreshed between steps."""
        return False

    def _init_wandb(self) -> None:
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from exc

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str) -> None:
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}")

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        import torch.distributed as dist

        num_processes = max(dist.get_world_size() if dist.is_initialized() else int(self.accelerator.num_processes), 1)
        logger.info(
            "_estimate_total_train_steps: dist.is_initialized=%s dist.get_world_size=%d "
            "accelerator.num_processes=%d -> num_processes=%d",
            dist.is_initialized(),
            dist.get_world_size() if dist.is_initialized() else -1,
            self.accelerator.num_processes,
            num_processes,
        )
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1)
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * self.min_lr_ratio,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self) -> tuple[str, float]:
        elapsed = max(time.perf_counter() - self.run_start_time, 1.0e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1.0e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_weights_before_prepare(self) -> None:
        """Load plain model weights before ``accelerator.prepare`` when needed."""
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            state_file = resume_path / "trainer_state.json"
            if not state_file.exists():
                return
            with open(state_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            orig_world = int(payload.get("original_world_size", self.accelerator.num_processes))
            if orig_world == self.accelerator.num_processes:
                return
            step_tag = resume_path.name
            weights_path = resume_path.parent.parent / "weights" / f"{step_tag}.pt"
            if not weights_path.exists():
                logger.warning("weights_path %s not found; cannot pre-load weights before prepare().", weights_path)
                return
            logger.info(
                "World size changed (%d->%d); pre-loading weights from %s before prepare().",
                orig_world,
                self.accelerator.num_processes,
                weights_path,
            )
            self.model.load_checkpoint(str(weights_path), optimizer=None)
            return
        if not str(resume_path).endswith(".pt"):
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading .pt weights before prepare(): %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)

    def _resume_or_load_checkpoint(self) -> None:
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str) -> None:
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "sample_offset": int(self.global_step * self.batch_size * self.accelerator.num_processes),
            "original_world_size": int(self.accelerator.num_processes),
        }
        with open(state_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        os.makedirs(state_path, exist_ok=True)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str) -> None:
        state_file = Path(state_dir) / "trainer_state.json"
        skip_optimizer = False
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            orig_world = int(payload.get("original_world_size", self.accelerator.num_processes))
            if orig_world != self.accelerator.num_processes:
                skip_optimizer = True
                logger.warning(
                    "World size changed (%d -> %d); skipping optimizer state load.",
                    orig_world,
                    self.accelerator.num_processes,
                )
        if not skip_optimizer:
            self.accelerator.load_state(input_dir=state_dir)
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                if skip_optimizer and "sample_offset" in payload:
                    sample_offset = int(payload["sample_offset"])
                    global_batch = self.batch_size * self.accelerator.num_processes
                    self.global_step = sample_offset // global_batch
                    self.batch_in_epoch = self.global_step
                    self.train_sampler.set_resume_sample_offset(sample_offset)
                    logger.info(
                        "Restored with sample_offset=%d: global_step remapped to %d (world_size=%d)",
                        sample_offset,
                        self.global_step,
                        self.accelerator.num_processes,
                    )
                else:
                    self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                    logger.info(
                        "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                        self.epoch,
                        self.batch_in_epoch,
                        self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                    )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        self.global_step = int(match.group(1)) if match else 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning("State file `%s` is missing; dataloader progress resume is skipped.", state_file)

    @torch.no_grad()
    def evaluate(self):
        """Return eval metrics or ``None``. Subclasses own method-specific eval."""
        return None

    def format_eval_log(self, metrics: dict) -> str:
        """Build a human-readable eval log line for method-specific metrics."""
        numeric = {
            key: value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        details = " ".join(f"{key}={value:.4f}" for key, value in sorted(numeric.items()))
        return f"[eval] step={self.global_step} {details}" if details else ""

    def build_eval_wandb_payload(self, metrics: dict) -> dict[str, float]:
        """Convert eval metrics to wandb scalar payload."""
        return {
            f"eval/{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def train(self) -> None:
        self.set_train_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            if self.maybe_reload_train_dataset():
                data_iter = iter(self.train_loader)
                continue
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    loss, loss_dict = self.compute_training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    global_loss = float(self.accelerator.gather(loss.detach().float().reshape(1)).mean().item())
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(self.accelerator.gather(metric_tensor).mean().item())
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm_tensor = grad_norm.detach().to(device=loss.device, dtype=torch.float32)
                    else:
                        grad_norm_tensor = torch.tensor(float(grad_norm), device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    self._maybe_log_train_step(global_loss, global_loss_metrics, global_grad_norm, current_lr)
                    self._maybe_evaluate()
                    self._maybe_save_checkpoint()

                    if self.global_step >= self.max_steps:
                        self._finish_training_due_to_max_steps()
                        return

        self._finish_training_completed()

    def _maybe_log_train_step(
        self,
        global_loss: float,
        global_loss_metrics: dict[str, float],
        global_grad_norm: float,
        current_lr: float,
    ) -> None:
        if self.log_every <= 0 or self.global_step % self.log_every != 0 or not self.accelerator.is_main_process:
            return

        eta_str, steps_per_sec = self._estimate_eta()
        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
            self.epoch,
            self.global_step,
            self.max_steps,
            global_loss,
        )
        if global_loss_metrics:
            detail_str = " ".join([f"{key}={value:.4f}" for key, value in sorted(global_loss_metrics.items())])
            description += detail_str + " "
        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
            current_lr,
            steps_per_sec,
            steps_per_sec * self.batch_size * self.accelerator.num_processes,
            eta_str,
        )
        logger.info(description)

        wandb_payload = {
            "train/loss": global_loss,
            "train/grad_norm": global_grad_norm,
            "train/lr": current_lr,
            "performance/steps_per_sec": steps_per_sec,
            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
        }
        for key, value in global_loss_metrics.items():
            wandb_payload[f"train/{key}"] = value
        self._wandb_log(wandb_payload)

    def _maybe_evaluate(self) -> None:
        if self.eval_every <= 0 or self.val_dataset is None or self.global_step % self.eval_every != 0:
            return
        metrics = self.evaluate()
        self.accelerator.wait_for_everyone()
        self.set_train_mode()
        if metrics is None or not self.accelerator.is_main_process:
            return

        description = self.format_eval_log(metrics)
        if description:
            logger.info(description)
        self._wandb_log(self.build_eval_wandb_payload(metrics))

    def _maybe_save_checkpoint(self) -> None:
        if self.save_every <= 0 or self.global_step % self.save_every != 0:
            return
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[ckpt] step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )

    def _finish_training_due_to_max_steps(self) -> None:
        try:
            if self.save_final:
                ckpt_info = self.save_checkpoint()
                if self.accelerator.is_main_process:
                    logger.info(
                        "[done] max_steps reached step=%d weights=%s state=%s",
                        self.global_step,
                        ckpt_info["weights_path"],
                        ckpt_info["state_path"],
                    )
            elif self.accelerator.is_main_process:
                logger.info("[done] max_steps reached step=%d final checkpoint disabled", self.global_step)
        finally:
            self._finish_wandb()

    def _finish_training_completed(self) -> None:
        try:
            if self.save_final:
                ckpt_info = self.save_checkpoint()
                if self.accelerator.is_main_process:
                    logger.info(
                        "[done] training finished step=%d weights=%s state=%s",
                        self.global_step,
                        ckpt_info["weights_path"],
                        ckpt_info["state_path"],
                    )
            elif self.accelerator.is_main_process:
                logger.info("[done] training finished step=%d final checkpoint disabled", self.global_step)
        finally:
            self._finish_wandb()


__all__ = ["DexboticGenerativeTrainer"]
