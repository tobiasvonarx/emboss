"""LiDAR loading, roof assignment, and calibrated residuals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import shutil
import zipfile

import laspy
import numpy as np
import pandas as pd
import shapely

from .models import RoofSegment
from .models import VectorHouse
from .workspace import EmbossWorkspace


CLASS_BUILDING = 6
CLASS_BUILDING_EVIDENCE = (CLASS_BUILDING,)
CLASS_ROOF_CANDIDATES = CLASS_BUILDING_EVIDENCE
CLASS_CALIBRATION = CLASS_BUILDING_EVIDENCE
MIN_ROOF_EVIDENCE_SIGNED_RESIDUAL_M = -1.0
DEFAULT_ROOF_RESIDUAL_BAND_M = 0.20
DEFAULT_SUPERSTRUCTURE_RESIDUAL_THRESHOLD_M = 0.40
CALIBRATION_ANCHOR_QUANTILE = 0.25
CALIBRATION_MODE_SEARCH_BAND_M = 0.30
CALIBRATION_MODE_BIN_WIDTH_M = 0.02
GLOBAL_CALIBRATION_MIN_POINTS = 30
GLOBAL_CALIBRATION_MAX_ABS_SHIFT_M = 1.50
SEGMENT_CALIBRATION_MAX_ABS_SHIFT_M = 0.50
SEGMENT_CALIBRATION_SUPPORT_BAND_M = 0.50
SEGMENT_CALIBRATION_MIN_SUPPORT_POINTS = 6
SEGMENT_CALIBRATION_MIN_AFFINE_POINTS = 20
SEGMENT_CALIBRATION_MIN_AFFINE_SPREAD_M = 0.75
SEGMENT_CALIBRATION_MIN_AFFINE_IMPROVEMENT_M = 0.02
SEGMENT_CALIBRATION_TILT_REGULARIZATION = 10.0
SEGMENT_CALIBRATION_INLIER_BAND_M = 0.20
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


@dataclass(frozen=True)
class SegmentCalibration:
    segment_id: str
    alpha: float
    beta: float
    gamma: float
    candidate_count: int
    support_count: int
    support_fraction: float
    support_spread_u_m: float
    support_spread_v_m: float
    method: str
    anchor_quantile: float
    raw_level_m: float
    calibrated_level_m: float

    def delta(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.alpha * np.asarray(u, dtype=np.float64) + self.beta * np.asarray(v, dtype=np.float64) + self.gamma


@dataclass(frozen=True)
class LidarPointStore:
    """Tile-level building-class point cache for batch house processing."""

    points: pd.DataFrame

    def points_for_house(self, house: VectorHouse, *, padding_m: float = 1.0) -> pd.DataFrame:
        if self.points.empty:
            return self.points.copy()
        min_x, min_y, max_x, max_y = house.bounds_xy
        min_x -= float(padding_m)
        min_y -= float(padding_m)
        max_x += float(padding_m)
        max_y += float(padding_m)
        return self.points.loc[
            (self.points["x"] >= min_x)
            & (self.points["x"] <= max_x)
            & (self.points["y"] >= min_y)
            & (self.points["y"] <= max_y)
        ].copy()


def materialize_las(workspace: EmbossWorkspace) -> Path:
    source = workspace.las_path_or_zip
    if source.suffix.lower() in {".las", ".laz"} and source.exists():
        return source
    workspace.tmp_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        candidate = next(workspace.tmp_dir.glob("*.las"), None) or next(workspace.tmp_dir.glob("*.laz"), None)
        if candidate is not None:
            return candidate
        raise FileNotFoundError(f"Missing LAS source {source}")
    with zipfile.ZipFile(source) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith((".las", ".laz"))]
        if not members:
            raise RuntimeError(f"No LAS/LAZ file found inside {source}")
        member = members[0]
        target = workspace.tmp_dir / Path(member).name
        if not target.exists():
            archive.extract(member, path=workspace.tmp_dir)
            extracted = workspace.tmp_dir / member
            if extracted != target:
                extracted.rename(target)
        return target


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


def _read_las_points(
    workspace: EmbossWorkspace,
    *,
    bounds_xy: tuple[float, float, float, float] | None = None,
    padding_m: float = 0.0,
    chunk_size: int = 500_000,
    class_ids: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    return read_las_points(
        materialize_las(workspace),
        bounds_xy=bounds_xy,
        padding_m=padding_m,
        chunk_size=chunk_size,
        class_ids=class_ids,
    )


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


def load_lidar_point_store_artifact(
    source: str | Path,
    *,
    cache_root: str | Path,
) -> LidarPointStore:
    """Load and cache building-class points from a canonical case LiDAR artifact."""

    source_path = Path(source).resolve()
    stat = source_path.stat()
    fingerprint = hashlib.sha256(
        f"{source_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(cache_root) / fingerprint
    points_path = cache_dir / "building_class_points.pkl"
    if points_path.exists():
        return LidarPointStore(pd.read_pickle(points_path))

    las_path = materialize_las_artifact(source_path, Path(cache_root) / "las")
    points = read_las_points(las_path, class_ids=CLASS_BUILDING_EVIDENCE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    points.to_pickle(points_path)
    return LidarPointStore(points)


def read_las_points_for_house(
    workspace: EmbossWorkspace,
    house: VectorHouse,
    *,
    padding_m: float = 1.0,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    return _read_las_points(
        workspace,
        bounds_xy=house.bounds_xy,
        padding_m=padding_m,
        chunk_size=chunk_size,
    )


def _building_point_store_paths(workspace: EmbossWorkspace) -> tuple[Path, Path]:
    root = workspace.tmp_dir / "emboss"
    return root / "building_class_points.pkl", root / "building_class_points.json"


def load_lidar_point_store(workspace: EmbossWorkspace, *, force_refresh: bool = False) -> LidarPointStore:
    """Load all tile building-class returns once for fast per-house cropping."""

    las_path = materialize_las(workspace)
    cache_path, metadata_path = _building_point_store_paths(workspace)
    metadata = {
        "las_path": str(las_path),
        "las_size": int(las_path.stat().st_size),
        "las_mtime_ns": int(las_path.stat().st_mtime_ns),
        "class_ids": list(CLASS_BUILDING_EVIDENCE),
    }
    if not force_refresh and cache_path.exists() and metadata_path.exists():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached_metadata == metadata:
                return LidarPointStore(pd.read_pickle(cache_path))
        except Exception:
            pass

    points = _read_las_points(workspace, class_ids=CLASS_BUILDING_EVIDENCE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_pickle(cache_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return LidarPointStore(points)


def _segment_normal(segment: RoofSegment) -> np.ndarray:
    normal = np.asarray(segment.normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-9:
        return normal / norm
    a, b, _c = segment.plane_coeffs
    fallback = np.asarray([-float(a), -float(b), 1.0], dtype=np.float64)
    return fallback / max(float(np.linalg.norm(fallback)), 1e-9)


def _dominant_xy_axis(segment: RoofSegment) -> np.ndarray:
    geometries = getattr(segment.polygon_xy, "geoms", (segment.polygon_xy,))
    edge_parts = []
    for geometry in geometries:
        exterior = getattr(geometry, "exterior", None)
        if exterior is None:
            continue
        coords = np.asarray(exterior.coords, dtype=np.float64)
        if len(coords) < 2:
            continue
        edges = coords[1:, :2] - coords[:-1, :2]
        lengths = np.linalg.norm(edges, axis=1)
        valid = lengths > 1e-9
        if np.any(valid):
            edge_parts.append((edges[valid], lengths[valid]))
    if not edge_parts:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    edges = np.vstack([part_edges for part_edges, _part_lengths in edge_parts])
    lengths = np.concatenate([part_lengths for _part_edges, part_lengths in edge_parts])
    max_index = int(np.argmax(lengths))
    axis = edges[max_index] / float(lengths[max_index])
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    return axis


def _segment_frame(segment: RoofSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid = segment.polygon_xy.centroid
    origin = np.asarray(
        [float(centroid.x), float(centroid.y), segment.z_at(float(centroid.x), float(centroid.y))],
        dtype=np.float64,
    )
    normal = _segment_normal(segment)
    a, b, _c = segment.plane_coeffs
    axis_xy = _dominant_xy_axis(segment)
    tangent_u = np.asarray(
        [float(axis_xy[0]), float(axis_xy[1]), float(a * axis_xy[0] + b * axis_xy[1])],
        dtype=np.float64,
    )
    tangent_u = tangent_u / max(float(np.linalg.norm(tangent_u)), 1e-9)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v = tangent_v / max(float(np.linalg.norm(tangent_v)), 1e-9)
    return origin, tangent_u, tangent_v, normal


def _local_coordinates(segment: RoofSegment, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(xyz, dtype=np.float64)
    origin, tangent_u, tangent_v, normal = _segment_frame(segment)
    centered = pts - origin[None, :]
    return centered @ tangent_u, centered @ tangent_v, centered @ normal


def _signed_to_vertical_scale(segment: RoofSegment) -> float:
    a, b, _c = segment.plane_coeffs
    return max(float(np.sqrt(a * a + b * b + 1.0)), 1e-9)


def _xyz_with_signed_residual(
    segment: RoofSegment,
    xyz: np.ndarray,
    signed_residual: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).copy()
    signed = np.asarray(signed_residual, dtype=np.float64)
    a, b, c = segment.plane_coeffs
    roof_z = float(a) * pts[:, 0] + float(b) * pts[:, 1] + float(c)
    pts[:, 2] = roof_z + signed * _signed_to_vertical_scale(segment)
    return pts


def _segment_choice_for_points(xy: np.ndarray, z: np.ndarray, segments: list[RoofSegment]) -> np.ndarray:
    """Pick one roof segment per return, resolving overlapping segments by height fit."""

    if len(segments) == 1:
        return np.zeros(len(xy), dtype=np.int64)
    point_geoms = shapely.points(xy[:, 0], xy[:, 1])
    empty = np.full(len(xy), np.inf, dtype=np.float64)
    contains_columns = []
    distance_columns = []
    for segment in segments:
        if segment.polygon_xy.is_empty:
            contains_columns.append(np.zeros(len(xy), dtype=bool))
            distance_columns.append(empty)
            continue
        contains_columns.append(
            np.asarray(
                shapely.intersects_xy(segment.polygon_xy.buffer(0.05), xy[:, 0], xy[:, 1]),
                dtype=bool,
            )
        )
        distance_columns.append(np.asarray(shapely.distance(point_geoms, segment.polygon_xy), dtype=np.float64))
    contains = np.column_stack(contains_columns)
    distances = np.column_stack(distance_columns)
    nearest_all = np.argmin(distances, axis=1)
    nearest_containing = np.argmin(np.where(contains, distances, np.inf), axis=1)
    planes = np.asarray([segment.plane_coeffs for segment in segments], dtype=np.float64)
    a, b, c = planes[:, 0], planes[:, 1], planes[:, 2]
    roof_z = xy[:, 0, None] * a[None, :] + xy[:, 1, None] * b[None, :] + c[None, :]
    signed_scale = np.sqrt(a * a + b * b + 1.0)
    residual = np.abs((np.asarray(z, dtype=np.float64)[:, None] - roof_z) / signed_scale[None, :])
    residual = np.where(np.isfinite(residual), residual, np.inf)
    containing_residual = np.where(contains, residual, np.inf)
    has_height_choice = np.isfinite(containing_residual).any(axis=1)
    nearest_containing_by_height = np.argmin(containing_residual, axis=1)
    containing_choice = np.where(has_height_choice, nearest_containing_by_height, nearest_containing)
    return np.where(contains.any(axis=1), containing_choice, nearest_all)


def assign_points_to_segments(points: pd.DataFrame, house: VectorHouse) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    segments = list(house.roof_segments)
    if not segments:
        return points.iloc[0:0].copy()
    xy = points[["x", "y"]].to_numpy(dtype=np.float64)
    z = points["z"].to_numpy(dtype=np.float64)
    choice = _segment_choice_for_points(xy, z, segments)
    planes = np.asarray([segment.plane_coeffs for segment in segments], dtype=np.float64)
    a, b, c = planes[choice, 0], planes[choice, 1], planes[choice, 2]
    residual = z - (a * xy[:, 0] + b * xy[:, 1] + c)
    signed_residual = residual / np.sqrt(a * a + b * b + 1.0)
    base_ids = {segment.segment_id for segment in house.base_segments}
    segment_ids = np.asarray([segment.segment_id for segment in segments], dtype=object)
    is_base = np.asarray([segment.segment_id in base_ids for segment in segments], dtype=bool)
    output = points.copy()
    output["segment_id"] = segment_ids[choice]
    output["is_base_segment"] = is_base[choice]
    output["residual"] = residual
    output["residual_signed"] = signed_residual
    output["x_cal"] = xy[:, 0]
    output["y_cal"] = xy[:, 1]
    output["z_cal"] = z
    output["residual_calibrated"] = signed_residual
    output["residual_vertical_calibrated"] = residual
    output["calibration_delta"] = 0.0
    output["global_calibration_delta"] = 0.0
    output["segment_calibration_delta"] = 0.0
    output["calibration_applied"] = False
    output["calibration_method"] = "none"
    output["calibration_candidate_count"] = 0
    output["calibration_support_count"] = 0
    output["calibration_support_fraction"] = np.nan
    output["calibration_support_spread_u_m"] = np.nan
    output["calibration_support_spread_v_m"] = np.nan
    output["calibration_anchor_quantile"] = np.nan
    output["calibration_raw_candidate_level_m"] = np.nan
    output["calibration_candidate_calibrated_level_m"] = np.nan
    return output


def filter_building_points(points: pd.DataFrame) -> pd.DataFrame:
    """Keep only LAS building-class returns for Emboss roof fitting."""

    if points.empty or "classification" not in points.columns:
        return points.copy()
    return points.loc[points["classification"].astype(int) == CLASS_BUILDING].copy()


def filter_points_to_roof_envelope(
    points: pd.DataFrame,
    house: VectorHouse,
    *,
    tolerance_m: float = 0.02,
) -> pd.DataFrame:
    """Keep returns whose XY location falls inside the vector roof envelope."""

    if points.empty:
        return points.copy()
    envelope = house.roof_envelope.buffer(float(tolerance_m))
    if envelope.is_empty:
        return points.iloc[0:0].copy()
    min_x, min_y, max_x, max_y = envelope.bounds
    bounded = points.loc[
        (points["x"] >= float(min_x))
        & (points["x"] <= float(max_x))
        & (points["y"] >= float(min_y))
        & (points["y"] <= float(max_y))
    ].copy()
    if bounded.empty:
        return bounded
    xy = bounded[["x", "y"]].to_numpy(dtype=np.float64)
    inside = np.asarray(
        shapely.intersects_xy(envelope, xy[:, 0], xy[:, 1]),
        dtype=bool,
    )
    return bounded.loc[inside].copy()


def filter_assigned_points_to_roof_residual_floor(
    assigned: pd.DataFrame,
    *,
    min_signed_residual_m: float = MIN_ROOF_EVIDENCE_SIGNED_RESIDUAL_M,
) -> pd.DataFrame:
    """Drop gross below-roof returns after segment assignment.

    The filter is applied immediately after assigning each return to a vector roof
    segment, because the signed roof residual is only defined relative to that
    segment. Returns far below the fixed vector roof are not usable support for
    roof-attached superstructure reconstruction.
    """

    if assigned.empty or "residual_signed" not in assigned.columns:
        return assigned.copy()
    residual = assigned["residual_signed"].to_numpy(dtype=np.float64)
    keep = np.isfinite(residual) & (residual >= float(min_signed_residual_m))
    return assigned.loc[keep].copy()


def _anchored_mode_location(values: np.ndarray) -> float:
    """Estimate the roof level as the residual mode anchored at the lower quartile.

    The lower quartile is a robust anchor inside the roof point cluster (the
    roof is the lowest surface, so as long as it holds at least a quarter of the
    returns the quartile lands within it), but as a quantile it sits a fixed
    fraction of the noise width below the roof peak. The histogram mode within a
    band around the anchor removes that bias without the failure mode of an
    unanchored global mode, which can lock onto a dominant elevated plane such
    as a large PV field. The peak bin is refined with the median of the samples
    in and directly around it, so a pure constant offset is recovered exactly.
    """

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    anchor = float(np.quantile(arr, CALIBRATION_ANCHOR_QUANTILE))
    band = float(CALIBRATION_MODE_SEARCH_BAND_M)
    selected = arr[(arr >= anchor - band) & (arr <= anchor + band)]
    if selected.size == 0:
        return anchor
    bin_width = float(CALIBRATION_MODE_BIN_WIDTH_M)
    edges = np.arange(anchor - band, anchor + band + bin_width, bin_width)
    counts, edges = np.histogram(selected, bins=edges)
    peak = int(np.argmax(counts))
    if counts[peak] < 3:
        # Too sparse for a meaningful histogram peak: argmax would tie-break
        # to the lowest occupied bin (~the sample minimum). Use the median.
        return float(np.median(selected))
    refined = selected[
        (selected >= edges[peak] - bin_width) & (selected <= edges[peak + 1] + bin_width)
    ]
    if refined.size == 0:
        return float(0.5 * (edges[peak] + edges[peak + 1]))
    return float(np.median(refined))


def _estimate_segment_calibration(
    segment: RoofSegment,
    segment_points: pd.DataFrame,
    *,
    min_support_points: int = SEGMENT_CALIBRATION_MIN_SUPPORT_POINTS,
    min_affine_points: int = SEGMENT_CALIBRATION_MIN_AFFINE_POINTS,
    min_affine_spread_m: float = SEGMENT_CALIBRATION_MIN_AFFINE_SPREAD_M,
    min_affine_improvement_m: float = SEGMENT_CALIBRATION_MIN_AFFINE_IMPROVEMENT_M,
    tilt_regularization: float = SEGMENT_CALIBRATION_TILT_REGULARIZATION,
    max_abs_shift_m: float = SEGMENT_CALIBRATION_MAX_ABS_SHIFT_M,
    support_band_m: float = SEGMENT_CALIBRATION_SUPPORT_BAND_M,
    inlier_band_m: float = SEGMENT_CALIBRATION_INLIER_BAND_M,
) -> SegmentCalibration | None:
    if len(segment_points) < min_support_points:
        return None
    candidates = segment_points.loc[segment_points["classification"].isin(CLASS_CALIBRATION)].copy()
    candidates = candidates.loc[np.isfinite(candidates["residual_signed"].to_numpy(dtype=np.float64))]
    if len(candidates) < min_support_points:
        return None
    candidate_xyz = candidates[["x", "y", "z"]].to_numpy(dtype=np.float64)
    candidate_u, candidate_v, candidate_residuals = _local_coordinates(segment, candidate_xyz)
    finite = (
        np.isfinite(candidate_u)
        & np.isfinite(candidate_v)
        & np.isfinite(candidate_residuals)
        & (np.abs(candidate_residuals) <= float(max_abs_shift_m))
    )
    if int(np.count_nonzero(finite)) < min_support_points:
        return None
    u = candidate_u[finite]
    v = candidate_v[finite]
    residuals = candidate_residuals[finite]
    raw_level = _anchored_mode_location(residuals)
    support_ceiling = min(
        float(DEFAULT_SUPERSTRUCTURE_RESIDUAL_THRESHOLD_M),
        float(raw_level) + float(support_band_m),
    )
    support = residuals <= float(support_ceiling)
    if int(np.count_nonzero(support)) < min_support_points:
        return None
    support_u = u[support]
    support_v = v[support]
    support_residuals = residuals[support]
    constant_delta = float(
        np.clip(
            _anchored_mode_location(support_residuals),
            -float(max_abs_shift_m),
            float(max_abs_shift_m),
        )
    )
    support_spread_u = _robust_spread(support_u)
    support_spread_v = _robust_spread(support_v)

    constant = SegmentCalibration(
        segment_id=segment.segment_id,
        alpha=0.0,
        beta=0.0,
        gamma=constant_delta,
        candidate_count=int(len(residuals)),
        support_count=int(len(support_residuals)),
        support_fraction=float(len(support_residuals) / max(len(residuals), 1)),
        support_spread_u_m=float(support_spread_u),
        support_spread_v_m=float(support_spread_v),
        method="segment_mode_constant",
        anchor_quantile=float(CALIBRATION_ANCHOR_QUANTILE),
        raw_level_m=float(raw_level),
        calibrated_level_m=_anchored_mode_location(support_residuals - constant_delta),
    )
    if (
        len(support_residuals) < int(min_affine_points)
        or support_spread_u < float(min_affine_spread_m)
        or support_spread_v < float(min_affine_spread_m)
    ):
        return constant

    alpha, beta, gamma = _fit_affine_delta(
        support_u,
        support_v,
        support_residuals,
        regularization=float(tilt_regularization),
    )
    delta = np.clip(
        alpha * support_u + beta * support_v + gamma,
        -float(max_abs_shift_m),
        float(max_abs_shift_m),
    )
    errors = support_residuals - delta
    centered_errors = errors - float(np.median(errors))
    inliers = np.abs(centered_errors) <= float(inlier_band_m)
    if int(np.count_nonzero(inliers)) >= int(min_affine_points) and not np.all(inliers):
        alpha, beta, gamma = _fit_affine_delta(
            support_u[inliers],
            support_v[inliers],
            support_residuals[inliers],
            regularization=float(tilt_regularization),
        )
        delta = np.clip(
            alpha * support_u + beta * support_v + gamma,
            -float(max_abs_shift_m),
            float(max_abs_shift_m),
        )

    constant_rmse = float(np.sqrt(np.mean((support_residuals - constant_delta) ** 2)))
    affine_rmse = float(np.sqrt(np.mean((support_residuals - delta) ** 2)))
    if constant_rmse - affine_rmse < float(min_affine_improvement_m):
        return constant

    return SegmentCalibration(
        segment_id=segment.segment_id,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        candidate_count=int(len(residuals)),
        support_count=int(len(support_residuals)),
        support_fraction=float(len(support_residuals) / max(len(residuals), 1)),
        support_spread_u_m=float(support_spread_u),
        support_spread_v_m=float(support_spread_v),
        method="segment_mode_affine",
        anchor_quantile=float(CALIBRATION_ANCHOR_QUANTILE),
        raw_level_m=float(raw_level),
        calibrated_level_m=_anchored_mode_location(support_residuals - delta),
    )


def _robust_spread(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    low, high = np.quantile(arr, [0.05, 0.95])
    return float(high - low)


def _fit_affine_delta(
    u: np.ndarray,
    v: np.ndarray,
    residuals: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, float, float]:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    u0 = float(np.mean(u))
    v0 = float(np.mean(v))
    design = np.column_stack([u - u0, v - v0, np.ones(len(residuals))])
    ridge = np.diag([float(regularization), float(regularization), 0.0])
    lhs = design.T @ design + ridge
    rhs = design.T @ residuals
    try:
        alpha, beta, gamma_centered = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        alpha, beta, gamma_centered = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    gamma = float(gamma_centered) - float(alpha) * u0 - float(beta) * v0
    return float(alpha), float(beta), gamma


def _estimate_global_calibration_delta(
    assigned: pd.DataFrame,
    *,
    min_points: int = GLOBAL_CALIBRATION_MIN_POINTS,
    max_abs_shift_m: float = GLOBAL_CALIBRATION_MAX_ABS_SHIFT_M,
) -> float:
    if assigned.empty or "residual_signed" not in assigned.columns:
        return 0.0
    candidates = assigned.loc[
        assigned["classification"].isin(CLASS_CALIBRATION)
        & assigned["is_base_segment"].astype(bool)
    ]
    residuals = candidates["residual_signed"].to_numpy(dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    residuals = residuals[np.abs(residuals) <= float(max_abs_shift_m)]
    if len(residuals) < int(min_points):
        return 0.0
    delta = _anchored_mode_location(residuals)
    return float(np.clip(delta, -float(max_abs_shift_m), float(max_abs_shift_m)))


def calibrate_residuals(assigned: pd.DataFrame, house: VectorHouse) -> pd.DataFrame:
    output = assigned.copy()
    if output.empty:
        return output
    segment_by_id = {segment.segment_id: segment for segment in house.roof_segments}
    output["global_calibration_delta"] = 0.0
    output["segment_calibration_delta"] = 0.0
    if "calibration_support_count" not in output.columns:
        output["calibration_support_count"] = 0
    if "calibration_support_fraction" not in output.columns:
        output["calibration_support_fraction"] = np.nan
    if "calibration_support_spread_u_m" not in output.columns:
        output["calibration_support_spread_u_m"] = np.nan
    if "calibration_support_spread_v_m" not in output.columns:
        output["calibration_support_spread_v_m"] = np.nan
    if "calibration_anchor_quantile" not in output.columns:
        output["calibration_anchor_quantile"] = np.nan
    if "calibration_raw_candidate_level_m" not in output.columns:
        output["calibration_raw_candidate_level_m"] = np.nan
    if "calibration_candidate_calibrated_level_m" not in output.columns:
        output["calibration_candidate_calibrated_level_m"] = np.nan
    global_delta = _estimate_global_calibration_delta(output)
    working = output.copy()
    if abs(global_delta) > 1e-9:
        for segment_id, group in working.groupby("segment_id"):
            segment = segment_by_id.get(str(segment_id))
            if segment is None:
                continue
            mask = working["segment_id"] == segment_id
            raw_xyz = working.loc[mask, ["x", "y", "z"]].to_numpy(dtype=np.float64)
            signed = working.loc[mask, "residual_signed"].to_numpy(dtype=np.float64) - float(global_delta)
            calibrated_xyz = _xyz_with_signed_residual(segment, raw_xyz, signed)
            working.loc[mask, "z"] = calibrated_xyz[:, 2]
            working.loc[mask, "residual_signed"] = signed
            working.loc[mask, "residual"] = segment.vertical_residual(calibrated_xyz)
            output.loc[mask, ["x_cal", "y_cal", "z_cal"]] = calibrated_xyz
            output.loc[mask, "residual_calibrated"] = signed
            output.loc[mask, "residual_vertical_calibrated"] = working.loc[mask, "residual"]
            output.loc[mask, "global_calibration_delta"] = float(global_delta)
            output.loc[mask, "calibration_delta"] = float(global_delta)
            output.loc[mask, "calibration_applied"] = True
            output.loc[mask, "calibration_method"] = "global_mode"
            output.loc[mask, "calibration_anchor_quantile"] = float(CALIBRATION_ANCHOR_QUANTILE)
            output.loc[mask, "calibration_raw_candidate_level_m"] = float(global_delta)
            output.loc[mask, "calibration_candidate_calibrated_level_m"] = 0.0

    for segment_id, group in working.groupby("segment_id"):
        segment = segment_by_id.get(str(segment_id))
        if segment is None:
            continue
        calibration = _estimate_segment_calibration(segment, group)
        if calibration is None:
            continue
        mask = output["segment_id"] == segment_id
        raw_xyz = working.loc[mask, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        u, v, _w = _local_coordinates(segment, raw_xyz)
        delta = np.clip(
            calibration.delta(u, v),
            -float(SEGMENT_CALIBRATION_MAX_ABS_SHIFT_M),
            float(SEGMENT_CALIBRATION_MAX_ABS_SHIFT_M),
        )
        signed = working.loc[mask, "residual_signed"].to_numpy(dtype=np.float64) - delta
        calibrated_xyz = _xyz_with_signed_residual(segment, raw_xyz, signed)
        output.loc[mask, ["x_cal", "y_cal", "z_cal"]] = calibrated_xyz
        total_delta = output.loc[mask, "global_calibration_delta"].to_numpy(dtype=np.float64) + delta
        output.loc[mask, "calibration_delta"] = total_delta
        output.loc[mask, "segment_calibration_delta"] = delta
        output.loc[mask, "calibration_applied"] = True
        if abs(global_delta) > 1e-9:
            output.loc[mask, "calibration_method"] = "global_mode+" + calibration.method
        else:
            output.loc[mask, "calibration_method"] = calibration.method
        output.loc[mask, "calibration_candidate_count"] = calibration.candidate_count
        output.loc[mask, "calibration_support_count"] = calibration.support_count
        output.loc[mask, "calibration_support_fraction"] = calibration.support_fraction
        output.loc[mask, "calibration_support_spread_u_m"] = calibration.support_spread_u_m
        output.loc[mask, "calibration_support_spread_v_m"] = calibration.support_spread_v_m
        output.loc[mask, "calibration_anchor_quantile"] = calibration.anchor_quantile
        output.loc[mask, "calibration_raw_candidate_level_m"] = calibration.raw_level_m
        output.loc[mask, "calibration_candidate_calibrated_level_m"] = calibration.calibrated_level_m
        output.loc[mask, "residual_calibrated"] = signed
        output.loc[mask, "residual_vertical_calibrated"] = segment.vertical_residual(calibrated_xyz)
    return output


def prepare_lidar_observations(
    workspace: EmbossWorkspace,
    house: VectorHouse,
    *,
    point_store: LidarPointStore | None = None,
) -> pd.DataFrame:
    points = (
        point_store.points_for_house(house)
        if point_store is not None
        else read_las_points_for_house(workspace, house)
    )
    return prepare_lidar_observations_from_points(points, house)


def prepare_lidar_observations_from_points(
    points: pd.DataFrame,
    house: VectorHouse,
) -> pd.DataFrame:
    """Apply the canonical Emboss filtering, assignment, and calibration stages."""

    points = filter_building_points(points)
    points = filter_points_to_roof_envelope(points, house)
    assigned = assign_points_to_segments(points, house)
    assigned = filter_assigned_points_to_roof_residual_floor(assigned)
    return calibrate_residuals(assigned, house)


def residual_thresholds(_observations: pd.DataFrame) -> tuple[float, float]:
    return DEFAULT_ROOF_RESIDUAL_BAND_M, DEFAULT_SUPERSTRUCTURE_RESIDUAL_THRESHOLD_M
