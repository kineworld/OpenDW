"""DW05 sample output conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DW05OutputConfig:
    """Shape and optional-output settings for DW05 model samples."""

    action_horizon: int
    action_dim: int = 32
    future_frame_count: int = 8
    future_frame_stride: int = 4
    future_image_size: tuple[int, int] = (384, 320)


class DW05OutputBuilder:
    """Convert intermediate dataset fields into the model-facing DW05 sample dict."""

    def __init__(self, config: DW05OutputConfig) -> None:
        self.config = config

    def build(self, result: dict, *, robot_task_success=1) -> dict:
        """Return a DW05 training sample from intermediate robot/WM fields."""
        has_action = "actions" in result
        video = torch.cat([result["images"].unsqueeze(0), result["future_images"]], dim=0).permute(1, 0, 2, 3)
        image_is_pad = torch.ones(self.config.future_frame_count + 1, dtype=torch.bool)
        future_valid_frames = int(result.get("future_valid_frames", 0))
        image_is_pad[: future_valid_frames + 1] = False

        if has_action:
            action = self._to_tensor(result["actions"], dtype=torch.float32)
            proprio = self._to_tensor(result.get("proprio", result["states"]), dtype=torch.float32)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            action_is_pad = self._to_tensor(
                result.get("action_is_pad", torch.zeros(action.shape[0], dtype=torch.bool)),
                dtype=torch.bool,
            )
            proprio_is_pad = self._to_tensor(
                result.get("proprio_is_pad", torch.zeros(proprio.shape[0], dtype=torch.bool)),
                dtype=torch.bool,
            )
            action_dim_mask = self._to_tensor(result["action_dim_mask"], dtype=torch.bool)
        else:
            action = torch.zeros(self.config.action_horizon, self.config.action_dim, dtype=torch.float32)
            proprio = torch.zeros(self.config.action_horizon, self.config.action_dim, dtype=torch.float32)
            action_is_pad = torch.ones(self.config.action_horizon, dtype=torch.bool)
            proprio_is_pad = torch.ones(self.config.action_horizon, dtype=torch.bool)
            action_dim_mask = torch.zeros(self.config.action_dim, dtype=torch.bool)

        if not robot_task_success:
            has_action = False

        output = {
            "action": action,
            "action_is_pad": action_is_pad,
            "has_action": torch.tensor(has_action, dtype=torch.bool),
            "context": result["context"],
            "context_mask": result["context_mask"],
            "image_is_pad": image_is_pad,
            "prompt": result["prompt"],
            "proprio": proprio,
            "proprio_is_pad": proprio_is_pad,
            "video": video,
            "action_dim_mask": action_dim_mask,
        }
        return output

    @staticmethod
    def _to_tensor(value, *, dtype=None) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value.copy())
        else:
            tensor = torch.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
