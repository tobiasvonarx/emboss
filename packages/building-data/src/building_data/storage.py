"""Atomic manifests and verified source downloads."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from filelock import FileLock


def write_json(path: Path, value: dict) -> None:
    from .serialization import json_ready

    path.parent.mkdir(parents=True, exist_ok=True)
    # Scientific diagnostics may use inf/NaN sentinels (e.g. raw image candidate
    # has no strip distance). Represent these as JSON null, after computation.
    payload = json_ready(json_ready(value), nonfinite_float_as_none=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stac_sha256(value: str | None) -> str | None:
    """Decode plain SHA-256 or STAC hexadecimal multihash (0x12, 0x20)."""
    if not value:
        return None
    normalized = str(value).lower()
    if normalized.startswith("1220") and len(normalized) == 68:
        normalized = normalized[4:]
    elif normalized.startswith("sha256:"):
        normalized = normalized[7:]
    return normalized if re.fullmatch(r"[0-9a-f]{64}", normalized) else None


def download(
    url: str,
    cache: Path,
    *,
    sha256: str | None = None,
    expected_size: int | None = None,
    attempts: int = 4,
    retry_delay_s: float = 1.0,
) -> Path:
    """Cache complete source bytes; retry interrupted/partial downloads atomically."""
    from urllib.parse import urlparse

    if attempts < 1:
        raise ValueError("Download attempts must be positive.")
    if sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("Expected SHA-256 must contain 64 hexadecimal characters.")
    expected_sha = sha256.lower() if sha256 else None
    name = Path(urlparse(url).path).name or "source"
    directory = Path(cache) / hashlib.sha256(url.encode()).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    receipt = target.with_suffix(target.suffix + ".source.json")
    with FileLock(str(target) + ".lock"):
        if target.exists():
            recorded = json.loads(receipt.read_text()) if receipt.is_file() else {}
            actual = checksum(target)
            required_sha = expected_sha or recorded.get("sha256")
            required_size = (
                expected_size if expected_size is not None else recorded.get("size")
            )
            if required_sha and actual != required_sha:
                raise ValueError(f"Cached source checksum mismatch: {target}")
            if required_size is not None and target.stat().st_size != required_size:
                raise ValueError(f"Cached source size mismatch: {target}")
            return target
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(attempts):
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "building-data/0.1",
                        "Accept-Encoding": "identity",
                        "Cache-Control": "no-cache",
                        "Range": "bytes=0-",
                    },
                )
                digest = hashlib.sha256()
                written = 0
                with (
                    urlopen(request, timeout=120) as response,
                    temporary.open("wb") as output,
                ):
                    declared = response.headers.get("Content-Length")
                    declared_size = int(declared) if declared is not None else None
                    status = getattr(response, "status", 200)
                    if status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", content_range)
                        if match is None or int(match[1]) + 1 != int(match[2]):
                            raise OSError(
                                f"Incomplete HTTP 206 source response: {content_range!r}"
                            )
                        if declared_size is not None and declared_size != int(match[2]):
                            raise OSError("HTTP range and content lengths disagree.")
                        declared_size = int(match[2])
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        written += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if written == 0 or (
                    declared_size is not None and written != declared_size
                ):
                    raise OSError(
                        f"Incomplete source download: got {written}, expected {declared_size} bytes."
                    )
                if expected_size is not None and written != expected_size:
                    raise OSError(
                        f"Source size mismatch: got {written}, expected {expected_size} bytes."
                    )
                actual = digest.hexdigest()
                if expected_sha and actual != expected_sha:
                    raise OSError(f"Download checksum mismatch: {url}")
                write_json(receipt, {"url": url, "sha256": actual, "size": written})
                temporary.replace(target)
                return target
            except (OSError, URLError, http.client.HTTPException) as exc:
                retryable = not isinstance(exc, HTTPError) or exc.code in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if not retryable or attempt + 1 == attempts:
                    raise RuntimeError(
                        f"Failed to download complete source after {attempt + 1} attempt(s): {url}: {exc}"
                    ) from exc
                time.sleep(min(float(retry_delay_s) * 2**attempt, 8.0))
            finally:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"No download attempt completed for {url}")


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path is outside the storage directory")
    return path
