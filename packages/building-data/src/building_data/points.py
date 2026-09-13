"""Streaming LAS/LAZ readers shared by the applications."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib
import shutil
import zipfile
import laspy
import numpy as np
import pandas as pd

LAS_POINT_COLUMNS = [
    "x",
    "y",
    "z",
    "intensity",
    "return_number",
    "number_of_returns",
    "scan_direction_flag",
    "edge_of_flight_line",
    "classification",
    "scan_angle_rank",
    "user_data",
    "point_source_id",
    "gps_time",
]

def read_las_points(
    las_path: str | Path,
    *,
    bounds_xy: tuple[float, float, float, float] | None = None,
    padding_m: float = 0.0,
    chunk_size: int = 500_000,
    class_ids: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    las_path = Path(las_path)
    chunk_size = max(1, int(chunk_size))
    if bounds_xy is None:
        min_x = min_y = -np.inf
        max_x = max_y = np.inf
    else:
        min_x, min_y, max_x, max_y = bounds_xy
        min_x -= float(padding_m)
        min_y -= float(padding_m)
        max_x += float(padding_m)
        max_y += float(padding_m)
    allowed_classes = None if class_ids is None else np.asarray(class_ids, dtype=np.uint8)
    chunks: list[pd.DataFrame] = []
    with laspy.open(las_path) as reader:
        for records in reader.chunk_iterator(chunk_size):
            xs = np.asarray(records.x, dtype=np.float64)
            ys = np.asarray(records.y, dtype=np.float64)
            zs = np.asarray(records.z, dtype=np.float64)
            classifications = np.asarray(records.classification, dtype=np.uint8)
            mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
            if allowed_classes is not None:
                mask &= np.isin(classifications, allowed_classes)
            if np.any(mask):
                count = int(np.count_nonzero(mask))
                dimension_names = set(records.point_format.dimension_names)

                def values(name: str, dtype: Any, default: int | float = 0) -> np.ndarray:
                    if name not in dimension_names:
                        return np.full(count, default, dtype=dtype)
                    return np.asarray(records[name], dtype=dtype)[mask]

                if "scan_angle_rank" in dimension_names:
                    scan_angles = values("scan_angle_rank", np.int8)
                elif "scan_angle" in dimension_names:
                    raw_angles = np.asarray(records.scan_angle, dtype=np.float64)[mask]
                    scan_angles = np.clip(np.rint(raw_angles), -128, 127).astype(np.int8)
                else:
                    scan_angles = np.zeros(count, dtype=np.int8)
                chunks.append(
                    pd.DataFrame(
                        {
                            "x": xs[mask],
                            "y": ys[mask],
                            "z": zs[mask],
                            "intensity": values("intensity", np.uint16),
                            "return_number": values("return_number", np.uint8),
                            "number_of_returns": values("number_of_returns", np.uint8),
                            "scan_direction_flag": values("scan_direction_flag", np.uint8),
                            "edge_of_flight_line": values("edge_of_flight_line", np.uint8),
                            "classification": classifications[mask],
                            "scan_angle_rank": scan_angles,
                            "user_data": values("user_data", np.uint8),
                            "point_source_id": values("point_source_id", np.uint16),
                            "gps_time": values("gps_time", np.float64),
                        }
                    )
                )
    if not chunks:
        return pd.DataFrame(columns=LAS_POINT_COLUMNS)
    return pd.concat(chunks, ignore_index=True)


def materialize_las_artifact(source: str | Path, cache_root: str | Path) -> Path:
    """Resolve a canonical case LAS or LAS-in-ZIP artifact to a readable LAS file."""

    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing LiDAR artifact {source_path}")
    if source_path.name.lower().endswith((".las", ".laz")):
        return source_path
    if not source_path.name.lower().endswith(".zip"):
        raise RuntimeError(f"Unsupported LiDAR artifact {source_path}; expected LAS, LAZ, or a ZIP containing either")

    stat = source_path.stat()
    fingerprint = hashlib.sha256(
        f"{source_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    with zipfile.ZipFile(source_path) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith((".las", ".laz"))]
        if not members:
            raise RuntimeError(f"No LAS/LAZ file found inside {source_path}")
        member = members[0]
        target = Path(cache_root) / fingerprint / Path(member).name
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        with archive.open(member) as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
        temporary.replace(target)
        return target
