"""One-time installation of the canonical national swissBUILDINGS3D source."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.request import Request
from urllib.request import urlopen
import zipfile


RELEASE = "2024-05"
ITEM_ID = f"swissbuildings3d_2_{RELEASE}"
FILE_NAME = f"{ITEM_ID}_2056_5728.gdb.zip"
SOURCE_URL = f"https://data.geo.admin.ch/ch.swisstopo.swissbuildings3d_2/{ITEM_ID}/{FILE_NAME}"
SOURCE_SIZE_BYTES = 3_601_104_519
SOURCE_SHA256 = "93cb0c57b205e4a64631ca4505a7bbe9f90a78d57e98f65626f2a7287c2192a1"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def repository_root(emboss_data_root: str | Path) -> Path:
    return Path(emboss_data_root) / "cache" / "swissbuildings3d_2" / RELEASE


def archive_path(root: str | Path) -> Path:
    return Path(root) / FILE_NAME


def geodatabase_path(root: str | Path) -> Path:
    return Path(root) / f"{ITEM_ID}.gdb"


def _status_path(root: str | Path) -> Path:
    return Path(root) / "status.json"


def _ready_path(root: str | Path) -> Path:
    return Path(root) / "READY"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def status(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    archive = archive_path(root)
    geodatabase = geodatabase_path(root)
    payload = _read_json(_status_path(root))
    ready = _ready_path(root).exists() and archive.exists() and geodatabase.exists()
    state = "ready" if ready else str(payload.get("state", "not_installed"))
    if state == "ready" and not ready:
        state = "not_installed"
    partial = archive.with_suffix(f"{archive.suffix}.part")
    downloaded_bytes = archive.stat().st_size if archive.exists() else (
        partial.stat().st_size if partial.exists() else 0
    )
    return {
        "state": state,
        "ready": ready,
        "release": RELEASE,
        "item_id": ITEM_ID,
        "source_url": SOURCE_URL,
        "archive_path": str(archive),
        "geodatabase_path": str(geodatabase),
        "archive_bytes": archive.stat().st_size if archive.exists() else 0,
        "expected_archive_bytes": SOURCE_SIZE_BYTES,
        "extracted_bytes": int(payload.get("extracted_bytes", 0)) if ready else 0,
        "downloaded_bytes": downloaded_bytes,
        "message": str(payload.get("message", "National swissBUILDINGS3D 2.0 is not installed.")),
        "updated_at_utc": payload.get("updated_at_utc"),
    }


def source_path(root: str | Path) -> Path | None:
    current = status(root)
    return Path(current["geodatabase_path"]) if current["ready"] else None


def source_metadata(root: str | Path) -> dict[str, Any] | None:
    current = status(root)
    if not current["ready"]:
        return None
    return {
        "dataset": "swissBUILDINGS3D 2.0",
        "release": RELEASE,
        "item_id": ITEM_ID,
        "source_url": SOURCE_URL,
        "archive_sha256": SOURCE_SHA256,
        "repository": "national_filegdb",
        "geodatabase_path": current["geodatabase_path"],
    }


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _set_status(root: Path, *, state: str, message: str, **values: Any) -> None:
    previous = _read_json(_status_path(root))
    payload = {
        **previous,
        **values,
        "state": state,
        "message": message,
        "release": RELEASE,
        "item_id": ITEM_ID,
        "source_url": SOURCE_URL,
        "updated_at_utc": _utc_now(),
    }
    _write_json(_status_path(root), payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(root: Path) -> Path:
    destination = archive_path(root)
    if destination.exists() and destination.stat().st_size == SOURCE_SIZE_BYTES:
        return destination

    partial = destination.with_suffix(f"{destination.suffix}.part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > SOURCE_SIZE_BYTES:
        partial.unlink()
        downloaded = 0
    elif downloaded == SOURCE_SIZE_BYTES:
        partial.replace(destination)
        return destination
    headers = {"User-Agent": "emboss-acquisition/1"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = Request(SOURCE_URL, headers=headers)
    with urlopen(request, timeout=120.0) as response:
        appending = downloaded > 0 and getattr(response, "status", None) == 206
        if not appending:
            downloaded = 0
        mode = "ab" if appending else "wb"
        last_report = 0.0
        with partial.open(mode) as handle:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 1.0:
                    percent = 100.0 * downloaded / SOURCE_SIZE_BYTES
                    _set_status(
                        root,
                        state="downloading",
                        message=f"Downloading national FileGDB: {percent:.1f}%.",
                        downloaded_bytes=downloaded,
                    )
                    last_report = now
    if downloaded != SOURCE_SIZE_BYTES:
        raise RuntimeError(
            f"Incomplete national FileGDB download: received {downloaded} of {SOURCE_SIZE_BYTES} bytes."
        )
    partial.replace(destination)
    return destination


def _safe_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members:
        raise RuntimeError("The national FileGDB archive is empty.")
    for member in members:
        path = Path(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("The national FileGDB archive contains an unsafe path.")
    return members


def _extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a ZIP safely while normalizing Windows path separators."""

    for member in _safe_zip_members(archive):
        relative = Path(member.filename.replace("\\", "/"))
        target = destination.joinpath(*relative.parts)
        if member.is_dir() or member.filename.endswith(("/", "\\")):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def _extract_geodatabase(root: Path, archive: Path) -> Path:
    target = geodatabase_path(root)
    if target.exists():
        shutil.rmtree(target)
    staging = root / "extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _set_status(root, state="extracting", message="Extracting the national FileGDB once.")
    try:
        with zipfile.ZipFile(archive) as source:
            _extract_zip(source, staging)
        candidates = sorted(path for path in staging.rglob("*.gdb") if path.is_dir())
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one FileGDB in the national archive, found {len(candidates)}.")
        candidate = candidates[0]
        if not any(candidate.glob("*.gdbtable")):
            raise RuntimeError("The extracted national FileGDB has no table data.")
        os.replace(candidate, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def install(root: str | Path) -> dict[str, Any]:
    """Download, verify, and extract the pinned national FileGDB exactly once."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "install.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if source_path(root) is not None:
            return status(root)
        try:
            _ready_path(root).unlink(missing_ok=True)
            _set_status(root, state="downloading", message="Preparing national FileGDB download.")
            archive = _download_archive(root)
            _set_status(root, state="verifying", message="Verifying the national FileGDB checksum.")
            digest = _sha256(archive)
            if digest != SOURCE_SHA256:
                archive.unlink(missing_ok=True)
                raise RuntimeError(f"National FileGDB checksum mismatch: expected {SOURCE_SHA256}, got {digest}.")
            geodatabase = _extract_geodatabase(root, archive)
            extracted_bytes = _directory_size(geodatabase)
            _ready_path(root).write_text(f"{RELEASE}\n", encoding="utf-8")
            _set_status(
                root,
                state="ready",
                message=f"National swissBUILDINGS3D 2.0 {RELEASE} is ready.",
                downloaded_bytes=SOURCE_SIZE_BYTES,
                extracted_bytes=extracted_bytes,
                archive_sha256=digest,
            )
            return status(root)
        except BaseException as exc:
            _set_status(root, state="failed", message=str(exc) or type(exc).__name__)
            raise
