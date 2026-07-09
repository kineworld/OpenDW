import numpy as np
import copy
import json

import megfile


class FatalTransformError(BaseException):
    pass


class ArrangeState:
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict:
            return episode_data_dict
        meta = episode_data_dict.get("meta_data", {})
        arrangement = meta.get("state_arrangement")
        if arrangement is None:
            return episode_data_dict
        state = episode_data_dict["state"]
        if not isinstance(state, np.ndarray) or state.ndim < 1:
            return episode_data_dict
        out_shape = list(state.shape)
        out_shape[-1] = len(arrangement)
        arranged = np.zeros(out_shape, dtype=state.dtype)
        src_dim = state.shape[-1]
        mask_1d = np.zeros(len(arrangement), dtype=bool)
        for dst_idx, src_idx in enumerate(arrangement):
            if src_idx == -1:
                continue
            if src_idx < 0 or src_idx >= src_dim:
                continue
            arranged[..., dst_idx] = state[..., int(src_idx)]
            mask_1d[dst_idx] = True
        episode_data_dict["state"] = arranged
        episode_data_dict["action_dim_mask"] = np.broadcast_to(mask_1d, arranged.shape).copy()
        return episode_data_dict


class AddTerminationState:
    def __init__(self, done_tail_length: int = 15):
        self.done_tail_length = int(done_tail_length)

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict:
            return episode_data_dict
        state = episode_data_dict["state"]
        if not isinstance(state, np.ndarray) or state.ndim < 1:
            return episode_data_dict
        termination = np.zeros(state.shape[:-1] + (1,), dtype=state.dtype)
        meta = episode_data_dict.setdefault("meta_data", {})
        if state.ndim >= 2 and self.done_tail_length > 0:
            episode_length = int(meta.get("jsonl_episode_length", state.shape[0]))
            window_start = int(meta.get("jsonl_window_start", 0))
            done_from = max(0, episode_length - self.done_tail_length)
            absolute_indices = window_start + np.arange(state.shape[0])
            done_mask = absolute_indices >= done_from
            mask_shape = (state.shape[0],) + (1,) * (termination.ndim - 1)
            termination = np.where(done_mask.reshape(mask_shape), 1, termination)
        episode_data_dict["state"] = np.concatenate([state, termination], axis=-1)
        if "action_dim_mask" in episode_data_dict:
            mask = episode_data_dict["action_dim_mask"]
            episode_data_dict["action_dim_mask"] = np.concatenate(
                [mask, np.ones(mask.shape[:-1] + (1,), dtype=bool)], axis=-1
            )
        non_delta_mask = meta.get("non_delta_mask")
        termination_idx = episode_data_dict["state"].shape[-1] - 1
        if non_delta_mask is None:
            meta["non_delta_mask"] = [termination_idx]
        elif isinstance(non_delta_mask, np.ndarray):
            if termination_idx not in non_delta_mask.tolist():
                meta["non_delta_mask"] = np.concatenate(
                    [non_delta_mask, np.array([termination_idx], dtype=non_delta_mask.dtype)]
                )
        else:
            if termination_idx not in non_delta_mask:
                non_delta_mask.append(termination_idx)
        return episode_data_dict


class PadState:
    def __init__(self, ndim: int = 32, axis: int = -1):
        self.ndim = ndim
        self.axis = axis

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        for key in ("state", "proprio"):
            if key not in episode_data_dict:
                continue
            value = episode_data_dict[key]
            if value.shape[self.axis] < self.ndim:
                pad_width = [(0, 0) for _ in range(len(value.shape))]
                pad_width[self.axis] = (0, self.ndim - value.shape[self.axis])
                episode_data_dict[key] = np.pad(value, pad_width, mode="constant", constant_values=0)
        return episode_data_dict


class AddProprioTrajectory:
    def __init__(self, trajectory_length: int = 10, padding_mode: str = "last"):
        self.trajectory_length = int(trajectory_length)
        self.padding_mode = padding_mode
        if self.trajectory_length <= 0:
            raise ValueError(f"trajectory_length must be positive, got {trajectory_length}.")
        if self.padding_mode not in ["last", "zero"]:
            raise ValueError(f"padding_mode must be 'last' or 'zero', got {padding_mode!r}.")

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict:
            return episode_data_dict
        state = episode_data_dict["state"]
        if not isinstance(state, np.ndarray) or state.ndim < 2 or len(state) == 0:
            return episode_data_dict
        valid_len = len(state)
        windows = []
        proprio_is_pad = np.zeros((valid_len, self.trajectory_length), dtype=bool)
        for i in range(valid_len):
            window = state[i: i + self.trajectory_length]
            valid_in_window = len(window)
            if valid_in_window < self.trajectory_length:
                proprio_is_pad[i, valid_in_window:] = True
                pad_state = np.zeros_like(state[-1]) if self.padding_mode == "zero" else state[-1]
                pad = np.array([np.copy(pad_state) for _ in range(self.trajectory_length - valid_in_window)])
                window = np.concatenate([window, pad], axis=0)
            windows.append(window)
        episode_data_dict["proprio"] = np.stack(windows, axis=0)
        episode_data_dict["proprio_is_pad"] = proprio_is_pad
        return episode_data_dict


class PadAction:
    def __init__(self, ndim: int = 32, axis: int = -1):
        self.ndim = ndim
        self.axis = axis

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "action" not in episode_data_dict:
            return episode_data_dict
        action = episode_data_dict["action"]
        if action.shape[self.axis] < self.ndim:
            current_dim = action.shape[self.axis]
            pad_count = self.ndim - current_dim
            pad_width = [(0, 0) for _ in range(len(action.shape))]
            pad_width[self.axis] = (0, pad_count)
            action = np.pad(action, pad_width, mode="constant", constant_values=0)
            episode_data_dict["action"] = action
            if "action_dim_mask" in episode_data_dict:
                mask = episode_data_dict["action_dim_mask"]
                mask_pad = [(0, 0)] * (mask.ndim - 1) + [(0, pad_count)]
                episode_data_dict["action_dim_mask"] = np.pad(mask, mask_pad, mode="constant", constant_values=False)
            else:
                mask_1d = np.concatenate([np.ones(current_dim, dtype=bool), np.zeros(pad_count, dtype=bool)])
                state = episode_data_dict.get("state")
                if state is not None and state.ndim >= 2:
                    episode_data_dict["action_dim_mask"] = np.broadcast_to(
                        mask_1d, state.shape[:-1] + (self.ndim,)
                    ).copy()
                else:
                    episode_data_dict["action_dim_mask"] = mask_1d
        else:
            if "action_dim_mask" not in episode_data_dict:
                state = episode_data_dict.get("state")
                mask_1d = np.ones(action.shape[self.axis], dtype=bool)
                if state is not None and state.ndim >= 2:
                    episode_data_dict["action_dim_mask"] = np.broadcast_to(
                        mask_1d, state.shape[:-1] + (action.shape[self.axis],)
                    ).copy()
                else:
                    episode_data_dict["action_dim_mask"] = mask_1d
        return episode_data_dict


class AddAction:
    def __init__(self, predict_length: int = 1):
        self.predict_length = predict_length

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "state" not in episode_data_dict:
            return episode_data_dict
        state = episode_data_dict["state"]
        action = state[self.predict_length:]
        if len(action) == 0:
            for key in ('state', 'action', 'robot'):
                episode_data_dict.pop(key, None)
            return episode_data_dict
        episode_data_dict["action"] = action
        episode_data_dict["abs_action"] = action
        for key in episode_data_dict.keys():
            if key == "meta_data":
                continue
            episode_data_dict[key] = episode_data_dict[key][: len(action)]
        return episode_data_dict


class DeltaAction:
    def __init__(self, enable=True):
        self.enable = enable

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        meta = episode_data_dict.get('meta_data', {}) or {}
        action_type = meta.get('action_type')
        if action_type is not None:
            if action_type == 'state':
                return episode_data_dict
            if action_type != 'delta_first_frame':
                raise ValueError(f"Unsupported action_type={action_type!r}")
        elif not self.enable:
            return episode_data_dict

        if 'state' not in episode_data_dict or 'action' not in episode_data_dict:
            return episode_data_dict

        non_delta_mask = episode_data_dict['meta_data']['non_delta_mask']
        periodic_mask = episode_data_dict['meta_data']['periodic_mask']
        periodic_range = episode_data_dict['meta_data']['periodic_range']
        state = episode_data_dict['state']
        action = episode_data_dict['action']

        if action.ndim == state.ndim:
            delta_action = action - state
        elif action.ndim == state.ndim + 1:
            delta_action = action - state[..., None, :]
        else:
            raise ValueError(f'action.ndim={action.ndim} incompatible with state.ndim={state.ndim}')

        if periodic_mask is not None:
            for dim in periodic_mask:
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] > periodic_range / 2,
                    delta_action[..., dim] - periodic_range,
                    delta_action[..., dim],
                )
                delta_action[..., dim] = np.where(
                    delta_action[..., dim] < -periodic_range / 2,
                    delta_action[..., dim] + periodic_range,
                    delta_action[..., dim],
                )

        delta_action[..., non_delta_mask] = action[..., non_delta_mask]
        episode_data_dict['delta_action'] = delta_action
        episode_data_dict['action'] = delta_action
        return episode_data_dict


class AddTrajectory:
    def __init__(self, trajectory_length: int = 10, flatten: bool = True,
                 padding_mode: str = 'last', padding_action: bool = True):
        self.trajectory_length = trajectory_length
        self.flatten = flatten
        self.padding_mode = padding_mode
        self.padding_action = padding_action
        if self.padding_mode not in ['last', 'zero']:
            raise ValueError(f"padding_mode must be 'last' or 'zero', got {padding_mode!r}.")

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if 'action' not in episode_data_dict:
            return episode_data_dict
        episode_data_dict['meta_data']['trajectory_length'] = self.trajectory_length
        non_delta_mask = episode_data_dict['meta_data']['non_delta_mask']
        action = episode_data_dict['action']
        valid_trajectory_length = len(action)
        if self.padding_action:
            action = self.pad(action, self.trajectory_length, non_delta_mask)
        else:
            if len(action) < self.trajectory_length:
                raise ValueError(
                    f"action length must be >= trajectory_length when padding_action=False, "
                    f"got action length {len(action)} and trajectory_length {self.trajectory_length}."
                )
        trajectory = [action]
        for i in range(1, self.trajectory_length):
            _next_action = np.copy(action[i:])
            _next_action = self.pad(_next_action, len(action), non_delta_mask)
            trajectory.append(_next_action)
        trajectory = np.stack(trajectory, axis=-1)
        trajectory = np.transpose(trajectory, (0, 2, 1))
        if self.flatten:
            trajectory = trajectory.reshape(trajectory.shape[0], -1)
        trajectory = trajectory[:valid_trajectory_length]
        episode_data_dict['trajectory'] = trajectory
        episode_data_dict['action'] = trajectory
        action_is_pad = np.zeros((valid_trajectory_length, self.trajectory_length), dtype=bool)
        for i in range(valid_trajectory_length):
            valid_in_chunk = min(valid_trajectory_length - i, self.trajectory_length)
            action_is_pad[i, valid_in_chunk:] = True
        episode_data_dict['action_is_pad'] = action_is_pad
        return episode_data_dict

    def pad(self, action, trajectory_length, non_delta_mask):
        if len(action) >= trajectory_length:
            return action
        padding_action = np.zeros_like(action[-1]) if self.padding_mode == 'zero' else action[-1]
        if self.padding_mode == 'zero':
            padding_action[non_delta_mask] = action[-1][non_delta_mask]
        action = np.concatenate(
            [action, np.array([np.copy(padding_action) for _ in range(trajectory_length - len(action))])], axis=0
        )
        return action


class ActionNormMultiDataset:
    def __init__(self, strict: bool = True, use_quantiles: bool = False):
        self.strict = strict
        self.use_quantiles = use_quantiles
        self._norm_stats_cache = {}

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        has_action = ("action" in episode_data_dict) or ("robot" in episode_data_dict) or ("state" in episode_data_dict)
        if not has_action:
            return episode_data_dict
        meta = episode_data_dict.get("meta_data", None)
        if meta is None:
            raise FatalTransformError("action exists but meta_data does not exist")
        norm_stats_path = meta.get("norm_stats_path", None)
        if not norm_stats_path:
            raise FatalTransformError(f"norm_stats_path is empty in meta_data")
        if not megfile.smart_exists(norm_stats_path):
            raise FatalTransformError(f"norm_stats_path does not exist: {norm_stats_path}")
        norm_stats = self._read_norm_stats(norm_stats_path)
        for key in norm_stats.keys():
            if key == "default":
                continue
            if key not in episode_data_dict:
                if self.strict:
                    raise KeyError(f"{key} not in episode_data_dict")
                continue
            episode_data_dict[key] = self._normalize(episode_data_dict[key], norm_stats[key])
        return episode_data_dict

    def _read_norm_stats(self, norm_stats_path: str) -> dict:
        if norm_stats_path in self._norm_stats_cache:
            return self._norm_stats_cache[norm_stats_path]
        with megfile.smart_open(norm_stats_path, "r") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and "norm_stats" in loaded:
            norm_stats = loaded["norm_stats"]
        else:
            norm_stats = {k: v for k, v in loaded.items() if isinstance(v, dict)}
        self._norm_stats_cache[norm_stats_path] = norm_stats
        return norm_stats

    def _normalize(self, data, stats):
        if self.use_quantiles:
            min_v = np.asarray(stats["q01"], dtype=np.float32)
            max_v = np.asarray(stats["q99"], dtype=np.float32)
            data = np.clip(data, min_v, max_v)
            return ((data - min_v) / (max_v - min_v + 1e-6) * 2.0 - 1.0).astype(np.float32)
        mean_v = np.asarray(stats["mean"], dtype=np.float32)
        std_v = np.asarray(stats["std"], dtype=np.float32)
        return ((data - mean_v) / (std_v + 1e-6)).astype(np.float32)
