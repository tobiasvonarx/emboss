"""Fit roof superstructures from image regions and roof-relative LiDAR returns."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import shapely
from shapely.errors import GEOSException
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split
from shapely.ops import unary_union

from building_data.geometry import clean_polygonal_geometry
from building_data.geometry import roof_drop_wall_faces
from building_data.geometry import surface_xy_line
from ..image_superstructures import ImageClassComponent
from ..image_superstructures import AREA_COMPARISON_EPSILON_M2
from ..image_superstructures import MIN_IMAGE_FOOTPRINT_AREA_M2
from ..image_superstructures import image_class_components
from ..lidar import CLASS_ROOF_CANDIDATES
from ..models import RectangularTopFace
from ..models import ReturnSupportModel
from ..models import RoofSegment
from ..models import SuperstructureSolid
from ..models import VectorHouse
from ..return_support import return_support_partition


MIN_LIDAR_ONLY_FOOTPRINT_AREA_M2 = 0.10
MAX_LIDAR_TOP_SLOPE_DEGREES = 60.0
SCAFFOLD_DETAIL_MATCH_FRACTION = 0.50


@dataclass(frozen=True)
class _FaceFit:
    polygon: Polygon
    plane: tuple[float, float, float]
    height_m: float
    height_model: str


@dataclass(frozen=True)
class _TopModel:
    plane: tuple[float, float, float]
    height_m: float
    height_model: str
    rmse_m: float
    support: pd.DataFrame


@dataclass(frozen=True)
class _Candidate:
    solid: SuperstructureSolid
    overlap_geometry: BaseGeometry


@dataclass(frozen=True)
class _ImageFootprint:
    footprint: Polygon
    fit_geometry: Polygon


@dataclass(frozen=True)
class _ImageProposal:
    footprint: Polygon
    fit_geometry: Polygon
    component: ImageClassComponent


@dataclass(frozen=True)
class _LidarProposal:
    footprint: Polygon
    points: pd.DataFrame


@dataclass(frozen=True)
class _ScaffoldDetail:
    footprint: Polygon
    segments: tuple[RoofSegment, ...]


@dataclass(frozen=True)
class _FitContext:
    house: VectorHouse
    support_model: ReturnSupportModel
    components: tuple[ImageClassComponent, ...]
    scaffold_details: tuple[_ScaffoldDetail, ...]
    drop_lines: tuple[LineString, ...]
    footprint_simplify_tolerance_m: float


def fit_superstructures(
    observations: pd.DataFrame,
    house: VectorHouse,
    support_model: ReturnSupportModel,
    *,
    image_class_map: np.ndarray | None = None,
    image_extent_lv95: tuple[float, float, float, float] | None = None,
) -> tuple[SuperstructureSolid, ...]:
    """Fit roof-attached superstructures from image masks and elevated LiDAR."""

    points = _measurement_points(observations, house, support_model)
    components = _image_components_from_map(
        image_class_map,
        image_extent_lv95,
    )
    drop_lines = tuple(
        line
        for line in (surface_xy_line(drop) for drop in roof_drop_wall_faces(house))
        if line is not None
    )
    context = _FitContext(
        house=house,
        support_model=support_model,
        components=components,
        scaffold_details=_scaffold_details(house),
        drop_lines=drop_lines,
        footprint_simplify_tolerance_m=_footprint_simplify_tolerance(image_class_map, image_extent_lv95),
    )
    selected = _select_candidates(_proposal_candidates(points, context))

    ordered = sorted(
        (candidate.solid for candidate in selected),
        key=lambda solid: (
            0 if solid.source.startswith("image") else 1,
            -float(solid.height_offset_m),
            tuple(round(value, 3) for value in Polygon(solid.footprint_xy).bounds)
            if len(solid.footprint_xy) >= 3
            else (),
            solid.solid_id,
        ),
    )
    return tuple(_renumber(solid, index) for index, solid in enumerate(ordered))


def _image_components_from_map(
    image_class_map: np.ndarray | None,
    image_extent_lv95: tuple[float, float, float, float] | None,
) -> tuple[ImageClassComponent, ...]:
    if image_class_map is None or image_extent_lv95 is None:
        return tuple()
    return image_class_components(
        class_map=image_class_map,
        extent_lv95=image_extent_lv95,
    )


def _proposal_candidates(points: pd.DataFrame, context: _FitContext) -> list[_Candidate]:
    image_candidates: list[_Candidate] = []
    for proposal in _image_proposals(points, context):
        image_candidates.extend(
            _solid_candidates_from_footprint(
                proposal.footprint,
                points,
                context,
                solid_id=f"candidate_{len(image_candidates):03d}",
                component=proposal.component,
                source="image",
                fit_geometry=proposal.fit_geometry,
            )
        )
    image_geometry = _clean_geometry(unary_union([item.overlap_geometry for item in image_candidates]))
    lidar_points = points
    if not image_geometry.is_empty:
        claimed = np.asarray(
            [_interiors_overlap(cell, image_geometry) for cell in points["_emboss_support_cell"]],
            dtype=bool,
        )
        lidar_points = points.loc[~claimed].copy()

    candidates = list(image_candidates)
    for proposal in _lidar_proposals(lidar_points, context):
        candidates.extend(
            _solid_candidates_from_footprint(
                proposal.footprint,
                proposal.points,
                context,
                solid_id=f"candidate_{len(candidates):03d}",
                component=None,
                source="lidar",
            )
        )
    return candidates


def _footprint_simplify_tolerance(
    image_class_map: np.ndarray | None,
    image_extent_lv95: tuple[float, float, float, float] | None,
) -> float:
    if image_class_map is None or image_extent_lv95 is None:
        return 0.0
    labels = np.asarray(image_class_map)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        return 0.0
    min_x, max_x, min_y, max_y = (float(value) for value in image_extent_lv95)
    pixel_x = abs(max_x - min_x) / float(labels.shape[1])
    pixel_y = abs(max_y - min_y) / float(labels.shape[0])
    return max(pixel_x, pixel_y)


def _measurement_points(
    observations: pd.DataFrame,
    house: VectorHouse,
    support_model: ReturnSupportModel,
) -> pd.DataFrame:
    count = len(support_model.return_xy)
    if count == 0:
        return pd.DataFrame(
            columns=[
                "x_cal",
                "y_cal",
                "segment_id",
                "residual_calibrated",
                "_emboss_support_weight_m2",
                "_emboss_support_cell",
                "_emboss_source_index",
                "_emboss_strong_lidar",
            ]
        )

    return_xy = np.asarray(support_model.return_xy, dtype=np.float64)
    residuals = np.asarray(support_model.residuals, dtype=np.float64)
    weights = np.asarray(support_model.return_area_weights_m2, dtype=np.float64)
    cells, partition_weights = return_support_partition(return_xy, house.roof_envelope)
    if len(weights) != count:
        weights = partition_weights
    source_indices = (
        np.asarray(support_model.source_indices, dtype=np.int64)
        if len(support_model.source_indices) == count
        else np.arange(count, dtype=np.int64)
    )

    out = pd.DataFrame(
        {
            "x_cal": return_xy[:, 0],
            "y_cal": return_xy[:, 1],
            "residual_calibrated": residuals,
            "_emboss_support_weight_m2": weights,
            "_emboss_support_cell": cells,
            "_emboss_source_index": source_indices,
            "segment_id": _segment_ids_for_points(return_xy, house),
        }
    )
    _copy_observation_columns(out, observations, source_indices)

    finite = np.isfinite(out[["x_cal", "y_cal", "residual_calibrated"]].to_numpy(dtype=np.float64)).all(axis=1)
    inside_roof = _contains_xy(house.roof_envelope, out[["x_cal", "y_cal"]].to_numpy(dtype=np.float64), buffer_m=1e-6)
    keep = finite & inside_roof
    if "classification" in out:
        classification = out["classification"]
        keep &= classification.isna().to_numpy(dtype=bool) | classification.isin(CLASS_ROOF_CANDIDATES).to_numpy(
            dtype=bool
        )
    if "is_base_segment" in out:
        base = out["is_base_segment"]
        keep &= base.isna().to_numpy(dtype=bool) | base.astype(bool).to_numpy(dtype=bool)

    out = out.loc[keep].copy()
    out["_emboss_strong_lidar"] = out["residual_calibrated"].to_numpy(dtype=np.float64) >= float(
        support_model.super_threshold_m
    )
    return out.reset_index(drop=True)


def _copy_observation_columns(
    out: pd.DataFrame,
    observations: pd.DataFrame,
    source_indices: np.ndarray,
) -> None:
    if observations.empty:
        return
    try:
        matched = observations.reindex(source_indices)
    except (KeyError, ValueError):
        return
    if len(matched) != len(out):
        return
    for column in ("classification", "is_base_segment", "segment_id", "z_cal", "z"):
        if column not in observations.columns:
            continue
        values = matched[column].to_numpy()
        if column in out:
            mask = pd.notna(values)
            out.loc[mask, column] = values[mask]
        else:
            out[column] = values


def _segment_ids_for_points(xy: np.ndarray, house: VectorHouse) -> list[str]:
    segments = house.base_segments or house.roof_segments
    if not segments:
        return [""] * len(xy)
    ids: list[str] = []
    prepared = [(segment, segment.polygon_xy.buffer(1e-8)) for segment in segments]
    for x, y in xy:
        point = Point(float(x), float(y))
        containing = [segment for segment, polygon in prepared if polygon.covers(point)]
        if containing:
            ids.append(str(containing[0].segment_id))
            continue
        nearest = min(segments, key=lambda segment: float(segment.polygon_xy.distance(point)))
        ids.append(str(nearest.segment_id))
    return ids


def _image_proposals(points: pd.DataFrame, context: _FitContext) -> tuple[_ImageProposal, ...]:
    proposals: list[_ImageProposal] = []
    for component in context.components:
        geometry = _clean_geometry(component.geometry.intersection(context.house.roof_envelope))
        for proposal in _image_component_footprints(geometry, points, context):
            proposals.append(
                _ImageProposal(
                    footprint=proposal.footprint,
                    fit_geometry=proposal.fit_geometry,
                    component=component,
                )
            )
    return tuple(proposals)


def _scaffold_details(house: VectorHouse) -> tuple[_ScaffoldDetail, ...]:
    segments = tuple(segment for segment in house.roof_segments if not segment.is_base)
    footprints = _polygon_parts(_clean_geometry(unary_union([segment.polygon_xy for segment in segments])))
    return tuple(
        _ScaffoldDetail(
            footprint=footprint,
            segments=tuple(segment for segment in segments if segment.polygon_xy.intersects(footprint)),
        )
        for footprint in footprints
    )


def _image_component_footprints(
    geometry: BaseGeometry,
    points: pd.DataFrame,
    context: _FitContext,
) -> tuple[_ImageFootprint, ...]:
    footprints: list[_ImageFootprint] = []
    for part in _polygon_parts(geometry):
        if _is_already_modeled(part, points, context):
            continue
        fit_geometry = _owned_drop_side(part, points, context)
        if fit_geometry.is_empty or float(fit_geometry.area) <= 0.0:
            continue
        footprints.append(_ImageFootprint(footprint=fit_geometry, fit_geometry=fit_geometry))
    return tuple(footprints)


def _is_already_modeled(part: Polygon, points: pd.DataFrame, context: _FitContext) -> bool:
    """Reject a scaffold match unless LiDAR shows additional height above it."""

    matching_details = tuple(
        detail
        for detail in context.scaffold_details
        if _overlap_fraction(part, detail.footprint) >= SCAFFOLD_DETAIL_MATCH_FRACTION
    )
    if not matching_details:
        return False
    return not any(
        _has_lidar_above_detail(
            part,
            points,
            detail,
            context.house,
            context.support_model.super_threshold_m,
        )
        for detail in matching_details
    )


def _overlap_fraction(left: Polygon, right: Polygon) -> float:
    intersection_area = float(left.intersection(right).area)
    return min(intersection_area / float(left.area), intersection_area / float(right.area))


def _has_lidar_above_detail(
    footprint: Polygon,
    points: pd.DataFrame,
    detail: _ScaffoldDetail,
    house: VectorHouse,
    threshold_m: float,
) -> bool:
    local = _points_inside(points, footprint)
    if local.empty:
        return False
    xy = local[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    z = _point_z(local, xy, house)
    for (x, y), point_z in zip(xy, z, strict=True):
        segment = _nearest_upper_segment(detail.segments, float(x), float(y))
        a, b, _c = segment.plane_coeffs
        signed_height = (
            float(point_z) - segment.z_at(float(x), float(y))
        ) / math.sqrt(a * a + b * b + 1.0)
        if signed_height >= float(threshold_m):
            return True
    return False


def _nearest_upper_segment(segments: tuple[RoofSegment, ...], x: float, y: float) -> RoofSegment:
    point = Point(float(x), float(y))
    distance = min(float(segment.polygon_xy.distance(point)) for segment in segments)
    nearest = [segment for segment in segments if float(segment.polygon_xy.distance(point)) <= distance + 1e-8]
    return max(nearest, key=lambda segment: segment.z_at(float(x), float(y)))


def _owned_drop_side(part: Polygon, points: pd.DataFrame, context: _FitContext) -> Polygon:
    owned = part
    for line in context.drop_lines:
        pieces = _drop_split_parts(owned, line)
        if len(pieces) < 2:
            continue
        owned = _choose_drop_side_by_support(pieces, points)
    return owned


def _drop_split_parts(geometry: BaseGeometry, line: LineString) -> tuple[Polygon, ...]:
    if geometry.is_empty or not geometry.intersects(line):
        return tuple()
    splitter = _extended_line_across_geometry(line, geometry)
    try:
        exact = _polygon_parts(split(geometry, splitter))
    except (GEOSException, ValueError):
        exact = tuple()
    return tuple(part for part in exact if float(part.area) > 0.0)


def _extended_line_across_geometry(line: LineString, geometry: BaseGeometry) -> LineString:
    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        return line
    direction = coords[-1, :2] - coords[0, :2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return line
    direction = direction / norm
    min_x, min_y, max_x, max_y = geometry.bounds
    reach = 2.0 * max(float(math.hypot(max_x - min_x, max_y - min_y)), float(line.length), 1.0)
    center = 0.5 * (coords[0, :2] + coords[-1, :2])
    return LineString(
        [
            tuple(center - reach * direction),
            tuple(center + reach * direction),
        ]
    )


def _choose_drop_side_by_support(parts: tuple[Polygon, ...], points: pd.DataFrame) -> Polygon:
    return max(
        parts,
        key=lambda part: (
            _strong_support_weight(points, part),
            _strong_support_count(points, part),
            float(part.area),
            tuple(round(value, 9) for value in part.bounds),
        ),
    )


def _strong_support_weight(points: pd.DataFrame, geometry: BaseGeometry) -> float:
    local = _points_inside(points, geometry)
    if local.empty or "_emboss_support_weight_m2" not in local:
        return 0.0
    strong = local["_emboss_strong_lidar"].astype(bool)
    return float(local.loc[strong, "_emboss_support_weight_m2"].sum())


def _strong_support_count(points: pd.DataFrame, geometry: BaseGeometry) -> int:
    local = _points_inside(points, geometry)
    if local.empty or "_emboss_strong_lidar" not in local:
        return 0
    return int(local["_emboss_strong_lidar"].sum())


def _solid_candidate_from_footprint(
    footprint: Polygon,
    points: pd.DataFrame,
    context: _FitContext,
    *,
    solid_id: str,
    component: ImageClassComponent | None,
    source: str,
    top_model: _TopModel | None = None,
    fit_geometry: BaseGeometry | None = None,
) -> _Candidate | None:
    footprint = _simplified_footprint(footprint, context, source=source)
    if not _polygon_parts(footprint) or float(footprint.area) <= 0.0:
        return None
    if top_model is None:
        local_points = _points_inside(points, fit_geometry if fit_geometry is not None else footprint)
        top_model = _fit_top_model(
            local_points,
            footprint,
            context.house,
            context.support_model,
            allow_image_prior=component is not None,
        )
    if top_model is None:
        return None

    faces = _clip_face_to_roof_parts(
        _FaceFit(
            polygon=footprint,
            plane=top_model.plane,
            height_m=top_model.height_m,
            height_model=top_model.height_model,
        ),
        context.house,
    )
    if not faces:
        return None
    visible = _clean_geometry(unary_union([face.polygon for face in faces]))
    visible_parts = _simplified_polygon_parts(visible, context, source=source)
    if not visible_parts:
        return None
    faces = tuple(
        _FaceFit(
            polygon=part,
            plane=top_model.plane,
            height_m=top_model.height_m,
            height_model=top_model.height_model,
        )
        for part in visible_parts
    )
    visible = _clean_geometry(unary_union([face.polygon for face in faces]))
    if source == "image" and float(visible.area) <= (
        MIN_IMAGE_FOOTPRINT_AREA_M2 + AREA_COMPARISON_EPSILON_M2
    ):
        return None
    if source == "lidar" and float(visible.area) < MIN_LIDAR_ONLY_FOOTPRINT_AREA_M2:
        return None

    overlap = _image_overlap(visible, (component,) if component is not None else context.components)
    support_summary = _support_summary(context.support_model, visible)
    strong_support_weight = float(
        top_model.support.loc[top_model.support["_emboss_strong_lidar"].astype(bool), "_emboss_support_weight_m2"].sum()
    )
    roof_contradiction_weight = _roof_support_weight(visible, context.support_model, top_model.support)
    top_faces = tuple(
        RectangularTopFace(
            face_id=f"top_{index:02d}",
            footprint_xy=_polygon_vertices(face.polygon),
            plane=face.plane,
        )
        for index, face in enumerate(faces)
    )
    if not top_faces:
        return None
    export_polygon = _footprint_export_polygon(visible)
    height = float(np.average([face.height_m for face in faces], weights=[max(float(face.polygon.area), 1e-9) for face in faces]))
    fit_terms = {
        "fit_rmse_m": top_model.rmse_m,
        "height_m": height,
        "footprint_area_m2": float(visible.area),
        "top_face_count": float(len(top_faces)),
        "sloped_top_face_count": 1.0 if top_model.height_model == "lidar_world_plane" else 0.0,
        "support_return_count": float(len(top_model.support)),
        "support_weight_m2": float(top_model.support["_emboss_support_weight_m2"].sum())
        if "_emboss_support_weight_m2" in top_model.support
        else 0.0,
        "strong_lidar_return_count": float(int(top_model.support["_emboss_strong_lidar"].sum()))
        if "_emboss_strong_lidar" in top_model.support
        else 0.0,
        "strong_lidar_support_weight_m2": strong_support_weight,
        "roof_contradiction_weight_m2": roof_contradiction_weight,
        "elevated_support_weight_m2": float(support_summary["elevated_support_weight_m2"]),
        "observed_support_weight_m2": float(support_summary["observed_support_weight_m2"]),
        "elevated_support_fraction": _fraction(
            support_summary["elevated_support_weight_m2"],
            support_summary["observed_support_weight_m2"],
        ),
        "image_overlap_area_m2": float(overlap["area_m2"]),
        "image_overlap_footprint_fraction": float(overlap["footprint_fraction"]),
        "image_overlap_component_fraction": float(overlap["component_fraction"]),
    }
    solid_source = _solid_source(source, top_model)
    solid = SuperstructureSolid(
        solid_id=solid_id,
        footprint_xy=_polygon_vertices(export_polygon),
        top_plane=top_faces[0].plane,
        host_segment_ids=_segment_ids_for_footprint_and_points(visible, top_model.support, context.house),
        point_count=int(len(top_model.support)),
        height_offset_m=height,
        fit_terms=fit_terms,
        top_faces=top_faces,
        source=solid_source,
        class_id=int(component.class_id) if component is not None else None,
        class_label=str(component.class_label) if component is not None else None,
        height_model=top_model.height_model,
    )
    return _Candidate(
        solid=solid,
        overlap_geometry=visible,
    )


def _solid_candidates_from_footprint(
    footprint: Polygon,
    points: pd.DataFrame,
    context: _FitContext,
    *,
    solid_id: str,
    component: ImageClassComponent | None,
    source: str,
    fit_geometry: BaseGeometry | None = None,
) -> list[_Candidate]:
    raw_footprint = _clean_geometry(footprint)
    footprint = _simplified_footprint(raw_footprint, context, source=source)
    if not _polygon_parts(footprint) or float(footprint.area) <= 0.0:
        return []
    support_geometry = fit_geometry if fit_geometry is not None else raw_footprint
    local_points = _points_inside(points, support_geometry)
    top_model = _fit_top_model(
        local_points,
        footprint,
        context.house,
        context.support_model,
        allow_image_prior=component is not None,
    )
    if top_model is None:
        return []

    candidate = _solid_candidate_from_footprint(
        footprint,
        points,
        context,
        solid_id=solid_id,
        component=component,
        source=source,
        top_model=top_model,
    )
    return [candidate] if candidate is not None else []


def _fit_top_model(
    points: pd.DataFrame,
    footprint: Polygon,
    house: VectorHouse,
    support_model: ReturnSupportModel,
    *,
    allow_image_prior: bool,
) -> _TopModel | None:
    strong = points.loc[points["_emboss_strong_lidar"].astype(bool)].copy() if not points.empty else points.copy()
    if strong.empty:
        if not allow_image_prior:
            return None
        segment = _dominant_segment_for_polygon(footprint, house)
        offset = float(support_model.roof_band_m)
        plane = _offset_plane(segment.plane_coeffs, offset)
        return _TopModel(
            plane=plane,
            height_m=offset,
            height_model="image_prior_roof_parallel",
            rmse_m=0.0,
            support=points.copy(),
        )

    return _fit_lidar_top_model(strong, house)


def _fit_lidar_top_model(
    strong: pd.DataFrame,
    house: VectorHouse,
) -> _TopModel | None:
    xy = strong[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    z = _point_z(strong, xy, house)
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    xy = xy[finite]
    z = z[finite]
    strong = strong.loc[finite].copy()
    if len(strong) == 0:
        return None

    plane = _constant_z_plane(z)
    height_model = "lidar_constant_z"
    use_plane = len(strong) >= 4 and (
        _leave_one_out_error(xy, z, model="plane")
        < _leave_one_out_error(xy, z, model="constant")
    )
    if use_plane:
        candidate_plane = _fit_plane(xy, z)
        max_grade = math.tan(math.radians(MAX_LIDAR_TOP_SLOPE_DEGREES))
        if math.hypot(candidate_plane[0], candidate_plane[1]) <= max_grade:
            plane = candidate_plane
            height_model = "lidar_world_plane"
    rmse = _fit_rmse(plane, xy, z)
    heights = _height_above_reference_segments(plane, strong, xy, house)
    height = max(float(np.mean(heights)), 0.0)
    return _TopModel(
        plane=plane,
        height_m=height,
        height_model=height_model,
        rmse_m=rmse,
        support=strong,
    )


def _lidar_proposals(points: pd.DataFrame, context: _FitContext) -> tuple[_LidarProposal, ...]:
    if points.empty:
        return tuple()
    strong = points.loc[points["_emboss_strong_lidar"].astype(bool)].copy()
    proposals: list[_LidarProposal] = []
    for cluster in _point_clusters(strong, context):
        footprint = _lidar_cluster_footprint(cluster, context)
        if footprint is None:
            continue
        proposals.append(_LidarProposal(footprint=footprint, points=cluster))
    return tuple(proposals)


def _point_clusters(points: pd.DataFrame, context: _FitContext) -> tuple[pd.DataFrame, ...]:
    if points.empty:
        return tuple()
    xy = points[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    cells = points["_emboss_support_cell"].tolist()
    parent = list(range(len(points)))
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            shared = cells[left].intersection(cells[right])
            adjacent = not shared.is_empty
            if adjacent and not _points_cross_drop_lines(xy[left], xy[right], context.drop_lines):
                _join(parent, left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(_root(parent, index), []).append(index)
    return tuple(points.iloc[indices].copy() for indices in groups.values())


def _lidar_cluster_footprint(cluster: pd.DataFrame, context: _FitContext) -> Polygon | None:
    if cluster.empty:
        return None
    xy = cluster[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    geometry = MultiPoint([(float(x), float(y)) for x, y in xy]).convex_hull
    segment_geometry = _segments_for_cluster(cluster, context.house)
    clipped = _clean_geometry(geometry.intersection(segment_geometry).intersection(context.house.roof_envelope))
    parts = [part for part in _polygon_parts(clipped) if float(part.area) > 0.0]
    if not parts:
        return None
    footprint = _clean_geometry(unary_union(parts))
    owned = _owned_drop_side(_footprint_export_polygon(footprint), cluster, context)
    if owned.is_empty or float(owned.area) <= 0.0 or _is_already_modeled(owned, cluster, context):
        return None
    return owned


def _segments_for_cluster(cluster: pd.DataFrame, house: VectorHouse) -> BaseGeometry:
    segment_by_id = {str(segment.segment_id): segment for segment in house.roof_segments}
    ids = {str(value) for value in cluster["segment_id"].tolist() if str(value) in segment_by_id}
    if not ids:
        return house.roof_envelope
    return _clean_geometry(unary_union([segment_by_id[segment_id].polygon_xy for segment_id in ids]))


def _select_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    image = _drop_contained_image_candidates(
        [candidate for candidate in candidates if candidate.solid.source.startswith("image")]
    )
    lidar = [candidate for candidate in candidates if not candidate.solid.source.startswith("image")]
    return [
        *image,
        *(
            candidate
            for candidate in lidar
            if not any(_interiors_overlap(candidate.overlap_geometry, item.overlap_geometry) for item in image)
        ),
    ]


def _drop_contained_image_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: -float(item.overlap_geometry.area)):
        if any(item.overlap_geometry.covers(candidate.overlap_geometry) for item in kept):
            continue
        kept.append(candidate)
    return kept


def _interiors_overlap(left: BaseGeometry, right: BaseGeometry) -> bool:
    if left.is_empty or right.is_empty:
        return False
    return bool(left.relate_pattern(right, "T********"))


def _solid_source(source: str, top_model: _TopModel) -> str:
    if source == "lidar":
        return "lidar"
    if top_model.height_model == "image_prior_roof_parallel":
        return "image_segmentation"
    return "image_lidar"


def _support_summary(support_model: ReturnSupportModel, geometry: BaseGeometry) -> dict[str, float]:
    if len(support_model.return_xy) == 0:
        return {
            "observed_support_weight_m2": 0.0,
            "elevated_support_weight_m2": 0.0,
        }
    xy = np.asarray(support_model.return_xy, dtype=np.float64)
    residuals = np.asarray(support_model.residuals, dtype=np.float64)
    weights = np.asarray(support_model.return_area_weights_m2, dtype=np.float64)
    if len(weights) != len(xy):
        _cells, weights = return_support_partition(xy, geometry)
    inside = _contains_xy(geometry, xy, buffer_m=1e-8)
    elevated = inside & (residuals >= float(support_model.super_threshold_m))
    return {
        "observed_support_weight_m2": float(np.sum(weights[inside])),
        "elevated_support_weight_m2": float(np.sum(weights[elevated])),
    }


def _roof_support_weight(
    geometry: BaseGeometry,
    support_model: ReturnSupportModel,
    support: pd.DataFrame,
) -> float:
    if len(support_model.return_xy) == 0:
        return 0.0
    xy = np.asarray(support_model.return_xy, dtype=np.float64)
    residuals = np.asarray(support_model.residuals, dtype=np.float64)
    weights = np.asarray(support_model.return_area_weights_m2, dtype=np.float64)
    if len(weights) != len(xy):
        _cells, weights = return_support_partition(xy, geometry)
    sources = (
        np.asarray(support_model.source_indices, dtype=np.int64)
        if len(support_model.source_indices) == len(xy)
        else np.arange(len(xy), dtype=np.int64)
    )
    support_sources = set(int(value) for value in support["_emboss_source_index"].tolist()) if not support.empty else set()
    inside = _contains_xy(geometry, xy, buffer_m=1e-8)
    roof_like = np.abs(residuals) <= float(support_model.roof_band_m)
    excluded = np.asarray([int(source) in support_sources for source in sources], dtype=bool)
    return float(np.sum(weights[inside & roof_like & ~excluded]))


def _points_inside(points: pd.DataFrame, geometry: BaseGeometry) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    inside = _contains_xy(geometry, points[["x_cal", "y_cal"]].to_numpy(dtype=np.float64), buffer_m=1e-8)
    return points.loc[inside].copy()


def _contains_xy(geometry: BaseGeometry, xy: np.ndarray, *, buffer_m: float = 1e-8) -> np.ndarray:
    if geometry is None or geometry.is_empty or len(xy) == 0:
        return np.zeros(len(xy), dtype=bool)
    query = geometry.buffer(float(buffer_m)) if buffer_m else geometry
    return np.asarray(shapely.intersects_xy(query, xy[:, 0], xy[:, 1]), dtype=bool)


def _points_cross_drop_lines(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    drop_lines: tuple[LineString, ...],
) -> bool:
    if not drop_lines:
        return False
    connector = LineString(
        [
            (float(left_xy[0]), float(left_xy[1])),
            (float(right_xy[0]), float(right_xy[1])),
        ]
    )
    if connector.length <= 1e-9:
        return False
    return any(connector.crosses(line) for line in drop_lines)


def _root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _join(parent: list[int], left: int, right: int) -> None:
    left_root = _root(parent, left)
    right_root = _root(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _fit_plane(xy: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    xy = np.asarray(xy, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    center_xy = np.mean(xy, axis=0)
    center_z = float(np.mean(z))
    design = np.column_stack(
        [
            xy[:, 0] - float(center_xy[0]),
            xy[:, 1] - float(center_xy[1]),
            np.ones(len(xy), dtype=np.float64),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, z - center_z, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])
    c = float(center_z + coeffs[2] - a * float(center_xy[0]) - b * float(center_xy[1]))
    return (a, b, c)


def _constant_z_plane(z: np.ndarray) -> tuple[float, float, float]:
    z = np.asarray(z, dtype=np.float64)
    return (0.0, 0.0, float(np.mean(z)))


def _leave_one_out_error(
    xy: np.ndarray,
    z: np.ndarray,
    *,
    model: str,
) -> float:
    errors = np.empty(len(z), dtype=np.float64)
    for held_out in range(len(z)):
        keep = np.arange(len(z)) != held_out
        train_xy = xy[keep]
        if model == "plane":
            design = np.column_stack(
                [
                    train_xy[:, 0] - float(np.mean(train_xy[:, 0])),
                    train_xy[:, 1] - float(np.mean(train_xy[:, 1])),
                    np.ones(len(train_xy), dtype=np.float64),
                ]
            )
            if np.linalg.matrix_rank(design) < 3:
                return math.inf
            fitted = _fit_plane(train_xy, z[keep])
        else:
            fitted = _constant_z_plane(z[keep])
        predicted = _eval_plane(fitted, xy[held_out : held_out + 1])[0]
        errors[held_out] = float(z[held_out] - predicted)
    return float(np.mean(np.square(errors)))


def _fit_rmse(
    plane: tuple[float, float, float],
    xy: np.ndarray,
    z: np.ndarray,
) -> float:
    if len(xy) == 0:
        return 0.0
    errors = z - _eval_plane(plane, xy)
    return float(np.sqrt(np.mean(np.square(errors))))


def _eval_plane(plane: tuple[float, float, float], xy: np.ndarray) -> np.ndarray:
    return float(plane[0]) * xy[:, 0] + float(plane[1]) * xy[:, 1] + float(plane[2])


def _point_z(points: pd.DataFrame, xy: np.ndarray, house: VectorHouse) -> np.ndarray:
    if "z_cal" in points.columns and points["z_cal"].notna().any():
        values = points["z_cal"].to_numpy(dtype=np.float64)
        missing = ~np.isfinite(values)
    elif "z" in points.columns and points["z"].notna().any():
        values = points["z"].to_numpy(dtype=np.float64)
        missing = ~np.isfinite(values)
    else:
        values = np.full(len(points), np.nan, dtype=np.float64)
        missing = np.ones(len(points), dtype=bool)
    if np.any(missing):
        residual = points["residual_calibrated"].to_numpy(dtype=np.float64)
        roof_z = _roof_z_for_points(points, xy, house)
        values[missing] = roof_z[missing] + residual[missing]
    return values


def _roof_z_for_points(points: pd.DataFrame, xy: np.ndarray, house: VectorHouse) -> np.ndarray:
    segment_by_id = {str(segment.segment_id): segment for segment in house.roof_segments}
    z = np.zeros(len(points), dtype=np.float64)
    for index, row in enumerate(points.itertuples(index=False)):
        segment = segment_by_id.get(str(getattr(row, "segment_id", "")))
        if segment is None:
            segment = _segment_for_xy(float(xy[index, 0]), float(xy[index, 1]), house)
        z[index] = segment.z_at(float(xy[index, 0]), float(xy[index, 1]))
    return z


def _height_above_reference_segments(
    plane: tuple[float, float, float],
    points: pd.DataFrame,
    xy: np.ndarray,
    house: VectorHouse,
) -> np.ndarray:
    return _eval_plane(plane, xy) - _roof_z_for_points(points, xy, house)


def _segment_for_xy(x: float, y: float, house: VectorHouse) -> RoofSegment:
    segments = house.base_segments or house.roof_segments
    point = Point(float(x), float(y))
    containing = [segment for segment in segments if segment.polygon_xy.buffer(1e-8).covers(point)]
    if containing:
        return containing[0]
    return min(segments, key=lambda segment: float(segment.polygon_xy.distance(point)))


def _dominant_segment_for_polygon(polygon: Polygon, house: VectorHouse) -> RoofSegment:
    segments = house.base_segments or house.roof_segments
    overlaps = [(float(polygon.intersection(segment.polygon_xy).area), segment) for segment in segments]
    area, segment = max(overlaps, key=lambda item: item[0])
    if area > 0.0:
        return segment
    point = polygon.representative_point()
    return _segment_for_xy(float(point.x), float(point.y), house)


def _offset_plane(plane: tuple[float, float, float], offset_m: float) -> tuple[float, float, float]:
    return (float(plane[0]), float(plane[1]), float(plane[2]) + float(offset_m))


def _clip_face_to_roof_parts(face: _FaceFit, house: VectorHouse) -> tuple[_FaceFit, ...]:
    pieces: list[Polygon] = []
    for segment in house.roof_segments:
        overlap = _clean_geometry(face.polygon.intersection(segment.polygon_xy))
        for overlap_part in _polygon_parts(overlap):
            pieces.extend(_clip_polygon_above_roof(overlap_part, face.plane, segment.plane_coeffs))
    if not pieces:
        return tuple()
    clipped = _clean_geometry(unary_union(pieces))
    clipped_parts = [part for part in _polygon_parts(clipped) if float(part.area) > 0.0]
    return tuple(
        _FaceFit(
            polygon=polygon,
            plane=face.plane,
            height_m=face.height_m,
            height_model=face.height_model,
        )
        for polygon in sorted(clipped_parts, key=lambda part: (-float(part.area), tuple(part.bounds)))
    )


def _clip_polygon_above_roof(
    polygon: Polygon,
    top_plane: tuple[float, float, float],
    roof_plane: tuple[float, float, float],
) -> tuple[Polygon, ...]:
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
        return tuple()
    clipped = _clip_coords_by_plane_difference(coords, top_plane, roof_plane)
    if len(clipped) < 3:
        return tuple()
    result = _clean_geometry(Polygon([(float(x), float(y)) for x, y in clipped]))
    return tuple(part for part in _polygon_parts(result) if float(part.area) > 1e-8)


def _clip_coords_by_plane_difference(
    coords: np.ndarray,
    top_plane: tuple[float, float, float],
    roof_plane: tuple[float, float, float],
) -> np.ndarray:
    output = [np.asarray(point, dtype=np.float64) for point in coords]
    delta = (
        float(top_plane[0] - roof_plane[0]),
        float(top_plane[1] - roof_plane[1]),
        float(top_plane[2] - roof_plane[2]),
    )

    def clearance(point: np.ndarray) -> float:
        return float(delta[0] * float(point[0]) + delta[1] * float(point[1]) + delta[2])

    clipped: list[np.ndarray] = []
    previous = output[-1]
    previous_value = clearance(previous)
    previous_inside = previous_value >= -1e-9
    for current in output:
        current_value = clearance(current)
        current_inside = current_value >= -1e-9
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-12:
                t = previous_value / denominator
                clipped.append(previous + t * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(clipped, dtype=np.float64)


def _segment_ids_for_footprint_and_points(
    footprint: BaseGeometry,
    points: pd.DataFrame | None,
    house: VectorHouse,
) -> tuple[str, ...]:
    ids = {
        str(segment.segment_id)
        for segment in house.roof_segments
        if float(footprint.intersection(segment.polygon_xy).area) > 1e-8
    }
    if points is not None and not points.empty and "segment_id" in points:
        ids.update(str(value) for value in points["segment_id"].tolist())
    return tuple(sorted(ids))


def _image_overlap(footprint: BaseGeometry, components: tuple[ImageClassComponent, ...]) -> dict[str, float | int | str | None]:
    if footprint.is_empty or not components:
        return {
            "score": 0.0,
            "area_m2": 0.0,
            "footprint_fraction": 0.0,
            "component_fraction": 0.0,
            "class_id": None,
            "class_label": None,
        }
    footprint_area = max(float(footprint.area), 1e-9)
    best = {
        "score": 0.0,
        "area_m2": 0.0,
        "footprint_fraction": 0.0,
        "component_fraction": 0.0,
        "class_id": None,
        "class_label": None,
    }
    for component in components:
        overlap_area = float(footprint.intersection(component.geometry).area)
        if overlap_area <= 0.0:
            continue
        component_area = max(float(component.geometry.area), 1e-9)
        footprint_fraction = float(np.clip(overlap_area / footprint_area, 0.0, 1.0))
        component_fraction = float(np.clip(overlap_area / component_area, 0.0, 1.0))
        score = min(footprint_fraction, component_fraction)
        if score > float(best["score"]):
            best = {
                "score": score,
                "area_m2": overlap_area,
                "footprint_fraction": footprint_fraction,
                "component_fraction": component_fraction,
                "class_id": int(component.class_id),
                "class_label": str(component.class_label),
            }
    return best


def _renumber(solid: SuperstructureSolid, index: int) -> SuperstructureSolid:
    return SuperstructureSolid(
        solid_id=f"solid_{index:03d}",
        footprint_xy=solid.footprint_xy,
        top_plane=solid.top_plane,
        host_segment_ids=solid.host_segment_ids,
        point_count=solid.point_count,
        height_offset_m=solid.height_offset_m,
        fit_terms=solid.fit_terms,
        top_faces=solid.top_faces,
        source=solid.source,
        class_id=solid.class_id,
        class_label=solid.class_label,
        height_model=solid.height_model,
    )


def _clean_geometry(geometry: BaseGeometry | None) -> BaseGeometry:
    try:
        return clean_polygonal_geometry(geometry)
    except (GEOSException, ValueError):
        return Polygon()


def _polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if geometry.is_empty:
        return tuple()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type == "MultiPolygon":
        return tuple(part for part in geometry.geoms if part.geom_type == "Polygon" and not part.is_empty)
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return tuple()
    return tuple(part for item in geoms for part in _polygon_parts(item))


def _footprint_export_polygon(geometry: BaseGeometry) -> Polygon:
    parts = _polygon_parts(geometry)
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    hull = geometry.convex_hull.buffer(0)
    return hull if hull.geom_type == "Polygon" and not hull.is_empty else max(parts, key=lambda part: float(part.area))


def _simplified_footprint(geometry: BaseGeometry, context: _FitContext, *, source: str) -> Polygon:
    footprint = _footprint_export_polygon(_clean_geometry(geometry))
    if footprint.is_empty:
        return footprint
    tolerance = max(float(context.footprint_simplify_tolerance_m), 0.0) if source == "image" else 0.0
    if tolerance <= 0.0:
        return footprint
    simplified = _footprint_export_polygon(
        _clean_geometry(footprint.simplify(tolerance, preserve_topology=True))
    )
    if simplified.is_empty or float(simplified.area) <= 0.0:
        return footprint
    return simplified


def _simplified_polygon_parts(
    geometry: BaseGeometry,
    context: _FitContext,
    *,
    source: str,
) -> tuple[Polygon, ...]:
    clean = _clean_geometry(geometry)
    tolerance = max(float(context.footprint_simplify_tolerance_m), 0.0) if source == "image" else 0.0
    if tolerance > 0.0:
        clean = _clean_geometry(clean.simplify(tolerance, preserve_topology=True))
    return tuple(
        part
        for part in _polygon_parts(clean)
        if float(part.area) > 0.0
    )


def _polygon_vertices(polygon: Polygon) -> tuple[tuple[float, float], ...]:
    if polygon.is_empty:
        return tuple()
    coords = tuple((float(x), float(y)) for x, y in polygon.exterior.coords[:-1])
    if _signed_area(coords) < 0.0:
        coords = tuple(reversed(coords))
    return coords


def _signed_area(coords: tuple[tuple[float, float], ...]) -> float:
    if len(coords) < 3:
        return 0.0
    total = 0.0
    for left, right in zip(coords, coords[1:] + coords[:1], strict=False):
        total += left[0] * right[1] - right[0] * left[1]
    return 0.5 * total


def _fraction(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(np.clip(float(numerator) / float(denominator), 0.0, 1.0))
