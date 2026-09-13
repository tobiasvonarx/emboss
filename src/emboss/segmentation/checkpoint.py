"""Pinned public checkpoint retrieval and explicit local overrides."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

MODEL_REPO_ID = "tvonarx/emboss-segmentation"
MODEL_REVISION = "3ebd37afb374a6e13aaa1498b69b056221a74168"
MODEL_FILENAME = "best.pt"
MODEL_SHA256 = "ab44e669fdcd3c92633685abc3ad0004b0cab702fc6e4681957a2d9bde2fc250"
MODEL_SIZE_BYTES = 866306919
CHECKPOINT_ENV_VAR = "EMBOSS_ROOF_SUPERSTRUCTURES_CHECKPOINT"


def checkpoint_override() -> str | None:
    """Read the environment, then .env in the invocation working directory only."""
    override = os.environ.get(CHECKPOINT_ENV_VAR)
    if override:
        return override
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == CHECKPOINT_ENV_VAR:
            return value.strip().strip("\"'") or None
    return None


@lru_cache(maxsize=8)
def _verify_checkpoint(path: str, size: int, mtime_ns: int) -> None:
    del (
        mtime_ns
    )  # Include file identity in the cache key so changed files are rechecked.
    if size != MODEL_SIZE_BYTES:
        raise ValueError(
            f"Checkpoint size mismatch at {path}: expected {MODEL_SIZE_BYTES}, got {size}."
        )
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != MODEL_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch at {path}: expected {MODEL_SHA256}, got {actual}."
        )


def download_checkpoint() -> Path:
    """Get and verify the immutable public checkpoint using the Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    stat = path.stat()
    _verify_checkpoint(str(path), stat.st_size, stat.st_mtime_ns)
    return path


def default_checkpoint_path() -> Path:
    """Resolve an explicit local override or download the pinned public model."""
    override = checkpoint_override()
    if override:
        return Path(override).expanduser().resolve()
    return download_checkpoint()
