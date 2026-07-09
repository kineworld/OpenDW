"""Prompt selection and cached text embedding utilities for DW05."""

from __future__ import annotations

import hashlib
import os
import random
import warnings
from typing import Optional

from loguru import logger
import megfile
import torch


TEXT_CACHE_SUBDIRS = ("prompt", "rewrite_prompt", "subtask", "subtask_rewrites", "caption")


def _local_mirror_prefixes() -> tuple[str, ...]:
    value = os.getenv("DW_LOCAL_MIRROR_PREFIXES", "")
    return tuple(prefix.strip().rstrip("/") for prefix in value.split(os.pathsep) if prefix.strip())


def _expand_local_mirror_dirs(cache_dir: str) -> list[str]:
    cache_dir = cache_dir.rstrip("/")
    if not cache_dir.startswith("s3://"):
        return [cache_dir]
    s3_path = cache_dir[len("s3://") :]
    return [f"{prefix}/{s3_path}" for prefix in _local_mirror_prefixes()] + [cache_dir]


def refine_text(text: Optional[str]) -> str:
    """Normalize instruction text and add terminal punctuation when needed."""
    if text is None:
        return ""
    refined = str(text).strip()
    if not refined:
        return ""
    if refined[-1] not in ".!?。！？…":
        refined += "."
    return refined


def _string_candidates(value) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return []


def select_rewrite_prompt(episode_data: dict) -> Optional[str]:
    """Select one rewritten prompt from frame metadata, if available."""
    extra = episode_data.get("extra") if isinstance(episode_data, dict) else None
    if not isinstance(extra, dict):
        return None
    candidates = _string_candidates(extra.get("rewrite_prompt"))
    return random.choice(candidates) if candidates else None


def select_subtask_prompt(subtask: str) -> Optional[str]:
    """Select one subtask prompt from a string/list value."""
    candidates = _string_candidates(subtask)
    return random.choice(candidates) if candidates else None


def apply_prompt_format(prompt: str, meta_data: dict) -> str:
    """Apply a dataset-specific prompt format declared in ``meta_data``."""
    fmt = (meta_data or {}).get("prompt_format")
    if fmt is None or not prompt:
        return prompt
    return fmt.format(task=prompt)


class PromptSelector:
    """Build the prompt used by DW05 from raw captions, subtasks, and rewrites."""

    def __init__(self, *, prompt_add_prob: float = 0.3, use_rewrite_prompt_and_subtask: bool = False):
        self.prompt_add_prob = float(prompt_add_prob)
        self.use_rewrite_prompt_and_subtask = bool(use_rewrite_prompt_and_subtask)
        self._fallback_warned: set[tuple[str, str, str]] = set()

    def select(
        self,
        episode_data: dict,
        prompt,
        subtask: str,
        meta_data: dict,
        jsonl_file: str,
        frame_index: Optional[int] = None,
    ) -> str:
        """Choose and format one prompt for a sample."""
        if self.use_rewrite_prompt_and_subtask:
            rewrite_prompt = select_rewrite_prompt(episode_data)
            subtask_prompt = select_subtask_prompt(subtask)
            if rewrite_prompt is not None and subtask_prompt is not None:
                prompt = subtask_prompt if random.random() < 0.5 else rewrite_prompt
                return apply_prompt_format(prompt, meta_data)
            if rewrite_prompt is not None:
                return apply_prompt_format(rewrite_prompt, meta_data)
            if subtask_prompt is not None:
                return apply_prompt_format(subtask_prompt, meta_data)
            self._warn_fallback(episode_data, meta_data, jsonl_file, frame_index)
            if isinstance(prompt, list):
                prompt = random.choice(prompt) if prompt else ""
            return apply_prompt_format(prompt, meta_data)

        if isinstance(prompt, list):
            prompt = random.choice(prompt) if prompt else ""

        if subtask:
            prompt = refine_text(prompt)
            if random.random() < self.prompt_add_prob and prompt:
                prompt = refine_text(f"{prompt} {subtask}")
            else:
                prompt = subtask

        return apply_prompt_format(prompt, meta_data)

    def _warn_fallback(
        self,
        episode_data: dict,
        meta_data: dict,
        jsonl_file: str,
        frame_index: Optional[int],
    ) -> None:
        has_rewrite = bool(select_rewrite_prompt(episode_data))
        dataset = (meta_data or {}).get("dataset") or (meta_data or {}).get("dataset_name") or ""
        warn_key = ("missing_rewrite_and_subtask", dataset, jsonl_file)
        if warn_key in self._fallback_warned:
            return
        self._fallback_warned.add(warn_key)
        logger.warning(
            "USE_REWRITE_PROMPT_AND_SUBTASK=1 but neither rewrite_prompt nor subtask prompt is available; "
            "falling back to the original prompt. has_rewrite={} dataset={!r} jsonl_file={!r} frame_index={!r}",
            has_rewrite,
            dataset,
            jsonl_file,
            frame_index,
        )


class TextEmbeddingCache:
    """Load precomputed T5 text embeddings from per-dataset cache directories."""

    def __init__(
        self,
        *,
        cache_dir: Optional[str] = None,
        context_len: int = 128,
        enc_id: str = "wan22ti2v5b",
        missing: str = "error",
        embedding_dim: int = 4096,
        empty_text_embed_path: Optional[str] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.context_len = int(context_len)
        self.enc_id = str(enc_id)
        self.missing = str(missing).lower()
        if self.missing not in {"error", "zero"}:
            raise ValueError("missing must be one of: error, zero")
        self.embedding_dim = int(embedding_dim)
        self.empty_text_embed_path = empty_text_embed_path
        self._missing_warned: set[str] = set()

    def load(
        self,
        prompt: str,
        *,
        meta_data: Optional[dict] = None,
        jsonl_file: Optional[str] = None,
        frame_index: Optional[int] = None,
    ):
        """Return ``(context, mask)`` tensors for ``prompt``."""
        if prompt:
            prompt = prompt.strip()
        if not prompt:
            dataset_name = (meta_data or {}).get("dataset_name") if meta_data is not None else None
            warnings.warn(
                f"Empty prompt encountered; loading empty text embedding from {self.empty_text_embed_path!r}. "
                f"dataset_name={dataset_name!r}, jsonl_file={jsonl_file!r}, frame_index={frame_index!r}."
            )
            return self._load_empty_text_context()

        cache_dirs = self._candidate_dirs(meta_data)
        if not cache_dirs:
            if self.missing == "zero":
                logger.warning(
                    "text_embedding_cache_dir is not set and meta_data.text_embed_dir is missing; "
                    "using zero context for prompt={!r}.",
                    prompt,
                )
                context = torch.zeros((self.context_len, self.embedding_dim), dtype=torch.bfloat16)
                context_mask = torch.zeros(self.context_len, dtype=torch.bool)
                return context, context_mask
            raise ValueError("text_embedding_cache_dir is not set and meta_data.text_embed_dir is missing.")

        context, mask, tried, _ = self._lookup(prompt, cache_dirs)
        if context is not None:
            return context, mask

        if "\n" in prompt:
            head = prompt.split("\n", 1)[0].strip()
            if head and head != prompt:
                head_ctx, head_mask, head_tried, head_path = self._lookup(head, cache_dirs)
                tried = tried + head_tried
                if head_ctx is not None:
                    warnings.warn(
                        f"Text embedding missing for full prompt; using first-paragraph fallback. "
                        f"len(prompt)={len(prompt)}, len(head)={len(head)}, found={head_path}. "
                        "Regenerate merged_txt and embeddings to drop this fallback."
                    )
                    return head_ctx, head_mask

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self.missing == "zero":
            if prompt_hash not in self._missing_warned:
                self._missing_warned.add(prompt_hash)
                logger.warning(
                    "Missing text embedding cache for prompt hash={}; using zero context. searched={}",
                    prompt_hash,
                    tried,
                )
            context = torch.zeros((self.context_len, self.embedding_dim), dtype=torch.bfloat16)
            context_mask = torch.zeros(self.context_len, dtype=torch.bool)
            return context, context_mask

        raise FileNotFoundError(
            f"Missing text embedding cache in any of {TEXT_CACHE_SUBDIRS} under {cache_dirs}. "
            f"prompt={prompt!r}. tried={tried}. "
            "Run scripts/data/precompute_text_embeds.py first or set missing_text_embedding=zero for smoke tests."
        )

    def _load_empty_text_context(self):
        if not self.empty_text_embed_path:
            if self.missing == "zero":
                context = torch.zeros((self.context_len, self.embedding_dim), dtype=torch.bfloat16)
                context_mask = torch.zeros(self.context_len, dtype=torch.bool)
                return context, context_mask
            raise FileNotFoundError(
                "empty_text_embed_path is not set. Provide a local/megfile path or set missing_text_embedding=zero."
            )
        with megfile.smart_open(self.empty_text_embed_path, "rb") as handle:
            payload = torch.load(handle, map_location="cpu")
        return payload["context"], payload["mask"].bool()

    def _candidate_dirs(self, meta_data: Optional[dict]) -> list[str]:
        candidate_cache_dirs = []
        if meta_data and meta_data.get("text_embed_dir"):
            candidate_cache_dirs.append(str(meta_data["text_embed_dir"]))
        if self.cache_dir:
            candidate_cache_dirs.append(str(self.cache_dir))

        cache_dirs = []
        seen_dirs = set()

        def add_cache_dir(cache_dir: str) -> None:
            cache_dir = cache_dir.rstrip("/")
            if not cache_dir or cache_dir in seen_dirs:
                return
            seen_dirs.add(cache_dir)
            cache_dirs.append(cache_dir)

        for cache_dir in candidate_cache_dirs:
            for variant in _expand_local_mirror_dirs(cache_dir):
                add_cache_dir(variant)
                leaf = os.path.basename(variant.rstrip("/"))
                if leaf in TEXT_CACHE_SUBDIRS:
                    add_cache_dir(os.path.dirname(variant.rstrip("/")))
        return cache_dirs

    def _lookup(self, prompt: str, cache_dirs: list[str]):
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{prompt_hash}.t5_len{self.context_len}.{self.enc_id}.pt"
        tried = []
        for cache_dir in cache_dirs:
            direct_candidate = os.path.join(cache_dir, filename)
            tried.append(direct_candidate)
            loaded = self._load_candidate(direct_candidate)
            if loaded is not None:
                return loaded[0], loaded[1], tried, direct_candidate
            for subdir in TEXT_CACHE_SUBDIRS:
                candidate = os.path.join(cache_dir, subdir, filename)
                tried.append(candidate)
                loaded = self._load_candidate(candidate)
                if loaded is not None:
                    return loaded[0], loaded[1], tried, candidate
        return None, None, tried, None

    @staticmethod
    def _load_candidate(candidate: str):
        if not megfile.smart_exists(candidate):
            return None
        with megfile.smart_open(candidate, "rb") as handle:
            payload = torch.load(handle, map_location="cpu")
        return payload["context"], payload["mask"].bool()
