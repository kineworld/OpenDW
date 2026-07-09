"""Multimodal loading and tensor preparation utilities for DW05 datasets."""

import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional
from loguru import logger
import megfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader
import av
from collections import defaultdict


@dataclass
class _KeySpec:
    is_video: bool
    url: Optional[str] = None
    needed: set = field(default_factory=set)


def _local_mirror_prefixes() -> tuple[str, ...]:
    value = os.getenv("DW_LOCAL_MIRROR_PREFIXES", "")
    return tuple(prefix.strip().rstrip("/") for prefix in value.split(os.pathsep) if prefix.strip())


def _s3_readahead_bytes() -> int:
    try:
        return max(0, int(os.getenv("DW_DVR_S3_READAHEAD_BYTES", str(256 * 1024))))
    except ValueError:
        return 256 * 1024


class _S3RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int, readahead: int = 256 * 1024):
        from megfile.s3_path import get_s3_client
        if not url.startswith("s3://"):
            raise ValueError(f"_S3RangeReader only supports s3:// urls, got {url!r}")
        path = url[len("s3://"):]
        self._bucket, _, self._key = path.partition("/")
        self._client = get_s3_client()
        self._size = int(size)
        self._readahead = max(0, int(readahead))
        self._pos = 0
        self._buf = b""
        self._buf_start = 0

    def readable(self): return True
    def seekable(self): return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self._pos = offset
        elif whence == io.SEEK_CUR: self._pos += offset
        elif whence == io.SEEK_END: self._pos = self._size + offset
        else: raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def tell(self): return self._pos

    def _get(self, start, length):
        end = min(start + length, self._size) - 1
        if end < start: return b""
        resp = self._client.get_object(Bucket=self._bucket, Key=self._key, Range=f"bytes={start}-{end}")
        return resp["Body"].read()

    def read(self, size=-1):
        if size is None or size < 0: size = self._size - self._pos
        if size <= 0 or self._pos >= self._size: return b""
        buf_end = self._buf_start + len(self._buf)
        if self._buf_start <= self._pos < buf_end:
            off = self._pos - self._buf_start
            data = self._buf[off:off + size]
            if data:
                self._pos += len(data)
                return data
        fetch_len = max(size, self._readahead)
        self._buf = self._get(self._pos, fetch_len)
        self._buf_start = self._pos
        data = self._buf[:size]
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

@lru_cache(maxsize=4096)
def _cached_local_url(url: str) -> str:
    if not url.startswith("s3://"):
        return url
    s3_path = url[len("s3://") :]
    for prefix in _local_mirror_prefixes():
        local = f"{prefix}/{s3_path}"
        if megfile.smart_exists(local):
            return local
    return url


def _resolve_media_url(data_path_prefix: str, path: str, media_path_resolver: Optional[str] = None) -> str:
    if media_path_resolver == "panda70m":
        parts = path.split("/")
        if len(parts) >= 3 and parts[1].isdigit() and "videos" not in parts:
            path = "/".join([parts[0], parts[1], "videos", parts[1], *parts[2:]])
    if path.startswith(("s3://", "http://", "https://")) or os.path.isabs(path):
        return path
    return os.path.join(data_path_prefix, path)



def letterbox_resize(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize a CHW or TCHW tensor into a padded canvas without distorting aspect ratio."""
    squeeze = tensor.dim() == 3
    if squeeze:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError(f"Expected [C,H,W] or [T,C,H,W], got shape={tuple(tensor.shape)}")
    _, _, height, width = tensor.shape
    scale = min(float(target_h) / float(height), float(target_w) / float(width))
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    if (resized_h, resized_w) != (height, width):
        tensor = F.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h = target_h - resized_h
    pad_w = target_w - resized_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_h or pad_w:
        tensor = F.pad(tensor, (left, right, top, bottom), mode="constant", value=0.0)
    return tensor.squeeze(0) if squeeze else tensor


def _pil_to_float_chw(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def concat_views(images: list[Image.Image], output_size: tuple[int, int]) -> torch.Tensor:
    """Convert camera views to DW05's normalized multi-view condition image."""
    if not images:
        raise ValueError("At least one RGB image is required to build a DW05 sample.")
    target_h, target_w = int(output_size[0]), int(output_size[1])
    if len(images) < 3:
        canvas = letterbox_resize(_pil_to_float_chw(images[0]), target_h, target_w)
    else:
        top_h = target_h * 2 // 3
        bottom_h = target_h - top_h
        left_w = target_w // 2
        right_w = target_w - left_w
        cam_top = letterbox_resize(_pil_to_float_chw(images[0]), top_h, target_w)
        cam_left = letterbox_resize(_pil_to_float_chw(images[1]), bottom_h, left_w)
        cam_right = letterbox_resize(_pil_to_float_chw(images[2]), bottom_h, right_w)
        canvas = torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)
    return canvas.sub(0.5).div(0.5)


def prepare_future_images(future_images, *, frame_count: int, output_size: tuple[int, int]) -> torch.Tensor:
    """Normalize DW05 future clips to [T, C, H, W] in [-1, 1]."""
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if isinstance(future_images, list):
        views = [
            item.detach().clone().float()
            if isinstance(item, torch.Tensor)
            else torch.from_numpy(np.asarray(item)).float()
            for item in future_images
        ]
        for index, view in enumerate(views):
            if view.ndim != 4:
                raise ValueError(f"future_images[{index}] must be [T,C,H,W], got shape={tuple(view.shape)}")
    else:
        future_tensor = (
            future_images.detach().clone().float()
            if isinstance(future_images, torch.Tensor)
            else torch.from_numpy(np.asarray(future_images)).float()
        )
        if future_tensor.ndim != 5:
            raise ValueError(f"future_images must be 5D, got shape={tuple(future_tensor.shape)}")
        if future_tensor.shape[-1] in (1, 3):
            future_tensor = future_tensor.permute(0, 1, 4, 2, 3).contiguous()
        elif future_tensor.shape[2] in (1, 3):
            future_tensor = future_tensor.contiguous()
        else:
            raise ValueError(
                "Unsupported future_images shape; expected [T,N,H,W,C] or [T,N,C,H,W], "
                f"got {tuple(future_tensor.shape)}."
            )
        views = [future_tensor[:, view_idx] for view_idx in range(future_tensor.shape[1])]
    if not views:
        raise ValueError("future_images contains no views.")

    def fix_time_dim(view: torch.Tensor) -> torch.Tensor:
        if view.shape[0] == 0:
            raise ValueError("future_images contains an empty clip.")
        if view.shape[0] < frame_count:
            pad = view[-1:].expand(frame_count - view.shape[0], *view.shape[1:]).clone()
            view = torch.cat([view, pad], dim=0)
        elif view.shape[0] > frame_count:
            view = view[:frame_count]
        if view.max().item() > 1.0:
            view = view / 255.0
        return view

    views = [fix_time_dim(view) for view in views]
    target_h, target_w = int(output_size[0]), int(output_size[1])
    if len(views) < 3:
        output = letterbox_resize(views[0], target_h, target_w)
    else:
        top_h = target_h * 2 // 3
        bottom_h = target_h - top_h
        left_w = target_w // 2
        right_w = target_w - left_w
        cam_top = letterbox_resize(views[0], top_h, target_w)
        cam_left = letterbox_resize(views[1], bottom_h, left_w)
        cam_right = letterbox_resize(views[2], bottom_h, right_w)
        output = torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)
    return output.sub(0.5).div(0.5)


class LoadImages:
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        meta = episode_data_dict.get("meta_data", {})
        fram_indicies = meta.get("fram_indicies")
        data_path_prefix = meta.get("data_path_prefix", "")
        media_path_resolver = meta.get("media_path_resolver")
        images_keys = meta.get("images_keys")

        if isinstance(fram_indicies, np.ndarray): fram_indicies = fram_indicies.tolist()
        if isinstance(fram_indicies, int): fram_indicies = [fram_indicies]

        if images_keys is not None:
            keys_to_parse = [k for k in images_keys if k in episode_data_dict]
        else:
            keys_to_parse = sorted(k for k in episode_data_dict if k.startswith("images"))

        for key in keys_to_parse:
            episode_data_dict = self._load_rgb(
                episode_data_dict,
                key,
                fram_indicies,
                data_path_prefix,
                media_path_resolver,
            )

        episode_data_dict["rgb_data"] = []
        if keys_to_parse:
            for rgb_data in zip(*[episode_data_dict[key] for key in keys_to_parse]):
                rgb_data = [item.get("data", None) for item in rgb_data]
                episode_data_dict["rgb_data"].append(rgb_data)

        for key in keys_to_parse:
            episode_data_dict.pop(key, None)
        return episode_data_dict

    @staticmethod
    def _load_rgb(episode_data_dict, key, fram_indicies=None, data_path_prefix="", media_path_resolver=None):
        images = episode_data_dict[key]
        image_frames = [
            (idx, frame)
            for idx, frame in enumerate(images)
            if frame["type"] == "image" and (fram_indicies is None or idx in fram_indicies)
        ]
        video_frames = [
            (idx, frame)
            for idx, frame in enumerate(images)
            if frame["type"] == "video" and (fram_indicies is None or idx in fram_indicies)
        ]

        video_cache: dict = {}
        frame_indices_map: dict = defaultdict(list)
        for _, frame in video_frames:
            video_url = _resolve_media_url(data_path_prefix, frame["url"], media_path_resolver)
            frame_indices_map[video_url].append(int(frame["frame_idx"]))
        for video_url, indices in frame_indices_map.items():
            video_cache[video_url] = LoadImages._load_video(video_url, indices)
        for _, frame in video_frames:
            video_url = _resolve_media_url(data_path_prefix, frame["url"], media_path_resolver)
            frame["data"] = video_cache[video_url][int(frame["frame_idx"])]

        for _, frame in image_frames:
            image_url = _resolve_media_url(data_path_prefix, frame["url"], media_path_resolver)
            frame["data"] = LoadImages._load_image(image_url)
        return episode_data_dict

    _video_backend_logged: bool = False

    @staticmethod
    def _load_video(video_url, frame_indices):
        backend = LoadImages._dvr_video_backend()
        if not LoadImages._video_backend_logged:
            logger.info("[LoadImages] video_backend={}", backend)
            LoadImages._video_backend_logged = True
        if backend in ("pyav_seek", "pyav_seek_jpeg_compat"):
            return LoadImages._load_video_pyav_seek(
                video_url,
                frame_indices,
                jpeg_compat=(backend == "pyav_seek_jpeg_compat"),
            )
        video_url = LoadImages._resolve_video_url(video_url)
        with megfile.smart_open(video_url, mode="rb") as f:
            f.seek(0)
            vr = VideoReader(f, num_threads=1)
            frames = vr.get_batch(frame_indices).asnumpy()
            images = {idx: Image.fromarray(frame) for idx, frame in zip(frame_indices, frames)}
            del vr
        return images

    @staticmethod
    def _load_image(image_url):
        image_url = _cached_local_url(image_url)
        with megfile.smart_open(image_url, mode="rb") as f:
            f.seek(0)
            bytes_data = f.read()
            image = Image.open(io.BytesIO(bytes_data), "r").convert("RGB")
        return image

    @staticmethod
    def _dvr_video_backend() -> str:
        value = os.getenv("DW_DVR_VIDEO_BACKEND", "pyav_seek_jpeg_compat").strip().lower()
        allowed = {"decord", "pyav_seek", "pyav_seek_jpeg_compat"}
        if value not in allowed:
            raise ValueError(f"Unsupported DW_DVR_VIDEO_BACKEND={value!r}")
        return value

    @staticmethod
    @lru_cache(maxsize=4096)
    def _resolve_video_url(video_url: str) -> str:
        if not video_url.startswith("s3://"):
            return video_url
        s3_path = video_url[len("s3://") :]
        for prefix in _local_mirror_prefixes():
            local_video_url = f"{prefix}/{s3_path}"
            if megfile.smart_exists(local_video_url):
                return local_video_url
        return video_url

    @staticmethod
    def _pil_default_jpeg_roundtrip_rgb(image) -> Image.Image:
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
            if arr.ndim == 2: arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] == 4: arr = arr[:, :, :3]
            if arr.dtype != np.uint8: arr = np.clip(arr, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="JPEG")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def _decode_pyav_seek_frames_from_container(container, frame_indices, video_url=None):
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if stream.average_rate is None or stream.time_base is None:
            raise RuntimeError("PyAV seek requires video average_rate and time_base")
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        wanted = sorted({int(idx) for idx in frame_indices})
        if not wanted: return {}
        target_ts = int(wanted[0] / fps / time_base)
        container.seek(target_ts, stream=stream)
        remaining = set(wanted)
        last = wanted[-1]
        images = {}
        for frame in container.decode(stream):
            if frame.pts is None: continue
            idx = int(frame.pts * time_base * fps + 0.5)
            if idx in remaining:
                images[idx] = frame.to_ndarray(format="rgb24")
                remaining.discard(idx)
                if not remaining: break
            elif idx > last: break
        if remaining:
            raise RuntimeError(f"Failed to seek/decode frames {sorted(remaining)}")
        return images

    @staticmethod
    def _load_video_pyav_seek(video_url, frame_indices, jpeg_compat=False):
        original_url = video_url
        video_url = LoadImages._resolve_video_url(video_url)

        def decode(container):
            arrays = LoadImages._decode_pyav_seek_frames_from_container(
                container,
                frame_indices,
                video_url=original_url,
            )
            if jpeg_compat:
                return {idx: LoadImages._pil_default_jpeg_roundtrip_rgb(arr) for idx, arr in arrays.items()}
            return {idx: Image.fromarray(arr).convert("RGB") for idx, arr in arrays.items()}

        if "://" not in video_url and os.path.exists(video_url):
            container = av.open(video_url)
            try: return decode(container)
            finally: container.close()

        if video_url.startswith("s3://"):
            reader = container = None
            try:
                size = megfile.smart_stat(video_url).size
                reader = _S3RangeReader(video_url, size, _s3_readahead_bytes())
                container = av.open(reader)
            except Exception as e:
                if container is not None: container.close()
                if reader is not None: reader.close()
                container = None
                logger.warning("[LoadImages] S3 range reader failed for {}: {!r}; falling back", video_url, e)
            if container is not None:
                try: return decode(container)
                finally: container.close(); reader.close()

        with megfile.smart_open(video_url, mode="rb") as f:
            f.seek(0)
            container = av.open(f)
            try: return decode(container)
            finally: container.close()


class LoadImagesWithFutureClip(LoadImages):
    def __init__(self, future_frame_count: int = 5, future_frame_stride: int = 1,
                 future_source_key=None, future_image_size=None):
        self.future_frame_count = int(future_frame_count)
        self.future_frame_stride = int(future_frame_stride)
        self.future_source_key = future_source_key
        if future_image_size is not None:
            h, w = int(future_image_size[0]), int(future_image_size[1])
            self.future_image_size = (h, w)
        else:
            self.future_image_size = None

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        meta = episode_data_dict.get('meta_data', {})
        fram_indicies = self._normalize_fram_indicies(meta.get('fram_indicies'))
        data_path_prefix = meta.get('data_path_prefix', '')
        media_path_resolver = meta.get('media_path_resolver')
        keys_to_parse = self._discover_keys(episode_data_dict, meta.get('images_keys'))

        future_keys = (self._select_future_source_keys(episode_data_dict, keys_to_parse)
                       if (fram_indicies and keys_to_parse) else [])
        episode_len = len(episode_data_dict.get(keys_to_parse[0], [])) if keys_to_parse else 0

        specs = self._build_specs(episode_data_dict, keys_to_parse, data_path_prefix, media_path_resolver)
        self._plan_requests(specs, episode_data_dict, keys_to_parse, future_keys, fram_indicies, episode_len)
        cache = self._materialize_io(specs, episode_data_dict, data_path_prefix, media_path_resolver)

        rgb_data, future_images, future_valid_frames = self._assemble_outputs(
            episode_data_dict, keys_to_parse, future_keys, fram_indicies, episode_len, cache,
        )

        episode_data_dict['rgb_data'] = rgb_data
        episode_data_dict['future_images'] = future_images
        episode_data_dict['future_valid_frames'] = future_valid_frames
        for key in keys_to_parse:
            episode_data_dict.pop(key, None)
        return episode_data_dict

    @staticmethod
    def _normalize_fram_indicies(fram_indicies) -> list:
        if isinstance(fram_indicies, np.ndarray): return fram_indicies.tolist()
        if isinstance(fram_indicies, int): return [fram_indicies]
        return fram_indicies

    @staticmethod
    def _discover_keys(episode_data_dict: dict, images_keys) -> list:
        if images_keys is not None:
            return [k for k in images_keys if k in episode_data_dict]
        return sorted(k for k in episode_data_dict if k.startswith('images'))

    @staticmethod
    def _build_specs(episode_data_dict, keys_to_parse, data_path_prefix, media_path_resolver) -> dict:
        specs = {}
        for key in keys_to_parse:
            frames = episode_data_dict.get(key) or []
            if frames and frames[0].get('type') == 'video':
                url = _resolve_media_url(data_path_prefix, frames[0]['url'], media_path_resolver)
                specs[key] = _KeySpec(is_video=True, url=url)
            else:
                specs[key] = _KeySpec(is_video=False)
        return specs

    def _plan_requests(self, specs, episode_data_dict, keys_to_parse, future_keys, fram_indicies, episode_len):
        for fi in fram_indicies or []:
            if not (0 <= fi < episode_len): continue
            for key in keys_to_parse:
                self._plan_current(specs[key], episode_data_dict.get(key) or [], fi)
            for fk in future_keys:
                self._plan_future(specs[fk], episode_data_dict.get(fk) or [], fi)

    @staticmethod
    def _plan_current(sp: _KeySpec, frames: list, fi: int):
        if not frames: return
        sp.needed.add(min(max(int(fi), 0), len(frames) - 1))

    def _plan_future(self, sp: _KeySpec, frames: list, fi: int):
        n_rows = len(frames)
        if n_rows == 0: return
        clamped_fi = min(max(int(fi), 0), n_rows - 1)
        for offset in range(self.future_frame_count):
            raw_li = clamped_fi + (offset + 1) * self.future_frame_stride
            sp.needed.add(min(raw_li, n_rows - 1))

    def _materialize_io(self, specs, episode_data_dict, data_path_prefix, media_path_resolver) -> dict:
        cache = {}
        for key, sp in specs.items():
            if not sp.needed: continue
            frames = episode_data_dict[key]
            if sp.is_video:
                li_to_fidx = {li: int(frames[li]['frame_idx']) for li in sp.needed}
                decoded = self._load_video(sp.url, sorted(set(li_to_fidx.values())))
                for li, fidx in li_to_fidx.items():
                    cache[(key, li)] = decoded[fidx]
            else:
                for li in sorted(sp.needed):
                    url = _resolve_media_url(data_path_prefix, frames[li]['url'], media_path_resolver)
                    cache[(key, li)] = self._load_image(url)
        return cache

    def _assemble_outputs(self, episode_data_dict, keys_to_parse, future_keys, fram_indicies, episode_len, cache):
        n_views = len(keys_to_parse)
        rgb_data = [[None] * n_views for _ in range(episode_len)]
        future_images = [None] * episode_len
        future_valid_frames = [0] * episode_len

        for fi in fram_indicies or []:
            if not (0 <= fi < episode_len): continue
            rgb_data[fi] = [cache.get((key, fi)) for key in keys_to_parse]
            if future_keys:
                clip, valid = self._assemble_future_for_fi(episode_data_dict, future_keys, fi, cache)
                if clip is not None:
                    future_images[fi] = clip
                    future_valid_frames[fi] = valid
        return rgb_data, future_images, future_valid_frames

    def _assemble_future_for_fi(self, episode_data_dict, future_keys, fi, cache):
        clips, valid_list = [], []
        for fk in future_keys:
            view = self._build_future_clip_one_view(episode_data_dict.get(fk) or [], fk, fi, cache)
            if view is None: continue
            clip_arr, valid = view
            clips.append(self._resize_clip(clip_arr))
            valid_list.append(valid)
        if not clips: return None, 0
        shapes = {c.shape for c in clips}
        if len(shapes) == 1:
            stacked = np.stack(clips, axis=1).transpose(0, 1, 4, 2, 3)
            return stacked, min(valid_list)
        per_view = [c.transpose(0, 3, 1, 2) for c in clips]
        return per_view, min(valid_list)

    def _build_future_clip_one_view(self, frames, key, fi, cache):
        n_rows = len(frames)
        if n_rows == 0: return None
        start_li = min(max(int(fi), 0), n_rows - 1)
        clip_imgs = []
        seen_li = set()
        for offset in range(self.future_frame_count):
            raw_li = start_li + (offset + 1) * self.future_frame_stride
            li = min(raw_li, n_rows - 1)
            seen_li.add(li)
            img = cache.get((key, li))
            if img is None: return None
            clip_imgs.append(np.asarray(img.convert('RGB'), dtype=np.uint8))
        if not clip_imgs: return None
        return np.stack(clip_imgs, axis=0), len(seen_li)

    def _select_future_source_keys(self, episode_data_dict, keys_to_parse):
        if self.future_source_key:
            candidates = (
                self.future_source_key
                if isinstance(self.future_source_key, list)
                else [self.future_source_key]
            )
            found = [k for k in candidates if k in episode_data_dict]
            if found: return found
        return list(keys_to_parse)

    def _resize_clip(self, clip: np.ndarray) -> np.ndarray:
        return clip
