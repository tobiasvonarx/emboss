"""Vector house loading and roof geometry helpers."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import make_valid
from shapely import set_precision
from shapely.errors import GEOSException
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .models import HouseIndexRow
from .models import RoofFace
from .models import RoofSegment
from .models import SurfaceFace
from .models import VectorHouse


MIN_ROOF_FACE_PROJECTED_AREA_M2 = 0.02
FLAT_ROOF_MAX_SLOPE_DEG = 5.0
HOUSE_TOUCH_TOLERANCE_M = 0.05
DROP_WALL_MIN_HEIGHT_M = 0.50
DROP_WALL_MIN_LENGTH_M = 0.25
TOPOLOGY_KEY_DECIMALS = 6
GEOMETRY_PRECISION_GRID_M = 1e-6
DEGENERATE_INTERIOR_RING_AREA_M2 = 1e-8
EdgeKey = tuple[tuple[float, float], tuple[float, float]]


def _parse_touching_building_fids(value: Any) -> tuple[int, ...]:
    try:
        if pd.isna(value):
            return ()
    except (TypeError, ValueError):
        pass
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip() for item in value]
    else:
        tokens = str(value).replace(",", ";").split(";")
    parsed: list[int] = []
    for token in tokens:
        text = str(token).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            continue
        parsed.append(int(float(text)))
    return tuple(parsed)


def fit_plane(points_xyz: list[tuple[float, float, float]] | np.ndarray) -> tuple[tuple[float, float, float], float]:
    pts = np.unique(np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3), axis=0)
    if len(pts) < 3:
        raise ValueError("Need at least three unique points to fit a plane")
    design = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float64)])
    coeffs, *_ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
    fitted = design @ coeffs
    rmse = float(np.sqrt(np.mean((pts[:, 2] - fitted) ** 2)))
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2])), rmse


def plane_normal(coeffs: tuple[float, float, float]) -> np.ndarray:
    a, b, _ = coeffs
    normal = np.asarray([-a, -b, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def _extract_polygon_paths(geometry: Any) -> list[list[tuple[float, float, float]]]:
    if geometry is None or geometry.IsEmpty():
        return []
    name = geometry.GetGeometryName().upper()
    if name == "POLYGON":
        ring = geometry.GetGeometryRef(0)
        if ring is None:
            return []
        return [
            [
                (float(ring.GetX(i)), float(ring.GetY(i)), float(ring.GetZ(i)))
                for i in range(ring.GetPointCount())
            ]
        ]
    if name in {"MULTIPOLYGON", "GEOMETRYCOLLECTION"} or geometry.GetGeometryCount() > 0:
        paths: list[list[tuple[float, float, float]]] = []
        for index in range(geometry.GetGeometryCount()):
            paths.extend(_extract_polygon_paths(geometry.GetGeometryRef(index)))
        return paths
    return []


def _coord_xyz(coord: Any) -> tuple[float, float, float]:
    values = tuple(coord)
    z = values[2] if len(values) >= 3 else 0.0
    return (float(values[0]), float(values[1]), float(z))


def _extract_shapely_polygon_paths(geometry: Any) -> list[list[tuple[float, float, float]]]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [[_coord_xyz(coord) for coord in geometry.exterior.coords]]
    geoms = getattr(geometry, "geoms", None)
    if geoms is not None:
        paths: list[list[tuple[float, float, float]]] = []
        for part in geoms:
            paths.extend(_extract_shapely_polygon_paths(part))
        return paths
    return []


def _iter_surface_features_with_ogr(
    vector_path: Path,
    *,
    layer_name: str,
    building_fid: int | None = None,
    required: bool = True,
) -> list[dict[str, Any]]:
    try:
        from osgeo import ogr
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GDAL/OGR is unavailable") from exc

    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise RuntimeError(f"Could not open vector source {vector_path}")
    layer = ds.GetLayerByName(str(layer_name))
    if layer is None and required and str(layer_name) == "Roof":
        layer = ds.GetLayer(0)
    if layer is None:
        if required:
            raise RuntimeError(f"No {layer_name} layer found in {vector_path}")
        return []

    rows: list[dict[str, Any]] = []
    layer.ResetReading()
    for feature in layer:
        fid = int(feature.GetFID())
        if building_fid is not None and fid != int(building_fid):
            continue
        geometry = feature.GetGeometryRef()
        paths = _extract_polygon_paths(geometry)
        if not paths:
            continue
        object_type = feature.GetField("OBJEKTART") if feature.GetFieldIndex("OBJEKTART") >= 0 else ""
        rows.append(
            {
                "building_fid": fid,
                "object_type": str(object_type or ""),
                "surface_kind": str(layer_name).lower(),
                "paths": paths,
            }
        )
    return rows


def _iter_surface_features_with_pyogrio(
    vector_path: Path,
    *,
    layer_name: str,
    building_fid: int | None = None,
    required: bool = True,
) -> list[dict[str, Any]]:
    try:
        import pyogrio
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pyogrio is unavailable") from exc

    try:
        layers = pyogrio.list_layers(vector_path)
        layer_names = [str(row[0]) for row in layers]
        if not layer_names:
            raise RuntimeError(f"No layers found in {vector_path}")
        selected_layer = str(layer_name)
        if selected_layer not in layer_names:
            if required and selected_layer == "Roof":
                selected_layer = layer_names[0]
            elif required:
                raise RuntimeError(f"No {layer_name} layer found in {vector_path}")
            else:
                return []
        info = pyogrio.read_info(vector_path, layer=selected_layer)
    except Exception as exc:
        raise RuntimeError(f"Could not open vector source {vector_path}") from exc
    columns = ["OBJEKTART"] if "OBJEKTART" in set(info.get("fields", ())) else []
    read_kwargs: dict[str, Any] = {
        "layer": selected_layer,
        "columns": columns,
        "fid_as_index": True,
    }
    if building_fid is not None:
        read_kwargs["fids"] = [int(building_fid)]
    try:
        frame = pyogrio.read_dataframe(vector_path, **read_kwargs)
    except Exception as exc:
        if building_fid is not None and "Could not read feature with fid" in str(exc):
            return []
        raise RuntimeError(f"Could not open vector source {vector_path}") from exc

    rows: list[dict[str, Any]] = []
    for fid, row in frame.iterrows():
        paths = _extract_shapely_polygon_paths(row.geometry)
        if not paths:
            continue
        object_type = row["OBJEKTART"] if "OBJEKTART" in frame.columns else ""
        rows.append(
            {
                "building_fid": int(fid),
                "object_type": str(object_type or ""),
                "surface_kind": str(selected_layer).lower(),
                "paths": paths,
            }
        )
    return rows


def _iter_surface_features(
    vector_path: Path,
    *,
    layer_name: str,
    building_fid: int | None = None,
    required: bool = True,
) -> list[dict[str, Any]]:
    try:
        return _iter_surface_features_with_pyogrio(
            vector_path,
            layer_name=layer_name,
            building_fid=building_fid,
            required=required,
        )
    except RuntimeError as pyogrio_exc:
        try:
            return _iter_surface_features_with_ogr(
                vector_path,
                layer_name=layer_name,
                building_fid=building_fid,
                required=required,
            )
        except RuntimeError as ogr_exc:
            raise RuntimeError(
                f"Could not read vector house geometry from {vector_path} "
                f"(OGR: {ogr_exc}; pyogrio: {pyogrio_exc})"
            ) from pyogrio_exc


def _iter_roof_features(vector_path: Path, *, building_fid: int | None = None) -> list[dict[str, Any]]:
    return _iter_surface_features(vector_path, layer_name="Roof", building_fid=building_fid, required=True)


def _polygon_from_path(path: list[tuple[float, float, float]]) -> Polygon:
    return _largest_polygon(clean_polygonal_geometry(Polygon([(float(x), float(y)) for x, y, _z in path])))


def clean_polygonal_geometry(
    geometry: BaseGeometry | None,
    *,
    precision_grid_m: float = GEOMETRY_PRECISION_GRID_M,
    min_interior_ring_area_m2: float = DEGENERATE_INTERIOR_RING_AREA_M2,
) -> BaseGeometry:
    """Normalize polygonal geometry before overlay operations."""

    if geometry is None or geometry.is_empty:
        return Polygon()
    repaired = _make_valid_polygonal_geometry(geometry)
    cleaned = _drop_degenerate_interiors(repaired, min_area_m2=float(min_interior_ring_area_m2))
    if cleaned.is_empty:
        return Polygon()
    if precision_grid_m > 0.0:
        try:
            cleaned = set_precision(cleaned, float(precision_grid_m))
        except (GEOSException, ValueError):
            pass
        cleaned = _drop_degenerate_interiors(
            _make_valid_polygonal_geometry(cleaned),
            min_area_m2=float(min_interior_ring_area_m2),
        )
    return _drop_degenerate_interiors(
        _make_valid_polygonal_geometry(cleaned),
        min_area_m2=float(min_interior_ring_area_m2),
    )


def _make_valid_polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return Polygon()
    try:
        if geometry.is_valid:
            return geometry
    except GEOSException:
        pass
    try:
        return make_valid(geometry)
    except (GEOSException, ValueError):
        try:
            return geometry.buffer(0)
        except (GEOSException, ValueError):
            return Polygon()


def _drop_degenerate_interiors(geometry: BaseGeometry, *, min_area_m2: float) -> BaseGeometry:
    parts = _polygonal_parts_without_degenerate_interiors(geometry, min_area_m2=float(min_area_m2))
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    try:
        return unary_union(parts)
    except (GEOSException, ValueError):
        return max(parts, key=lambda part: float(part.area))


def _polygonal_parts_without_degenerate_interiors(
    geometry: BaseGeometry,
    *,
    min_area_m2: float,
) -> tuple[Polygon, ...]:
    if geometry.is_empty:
        return tuple()
    if geometry.geom_type == "Polygon":
        polygon = _polygon_without_degenerate_interiors(geometry, min_area_m2=float(min_area_m2))
        return (polygon,) if not polygon.is_empty and float(polygon.area) > 0.0 else tuple()
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return tuple()
    return tuple(
        part
        for item in geoms
        for part in _polygonal_parts_without_degenerate_interiors(item, min_area_m2=float(min_area_m2))
        if not part.is_empty and float(part.area) > 0.0
    )


def _polygon_without_degenerate_interiors(polygon: Polygon, *, min_area_m2: float) -> Polygon:
    interiors = []
    for ring in polygon.interiors:
        try:
            area = abs(float(Polygon(ring).area))
        except (GEOSException, ValueError):
            area = 0.0
        if area >= float(min_area_m2):
            interiors.append(tuple(ring.coords))
    try:
        return Polygon(tuple(polygon.exterior.coords), interiors)
    except (GEOSException, ValueError):
        return Polygon()


def _largest_polygon(geometry: Any) -> Polygon:
    if geometry is None or geometry.is_empty:
        return Polygon()
    if geometry.geom_type == "MultiPolygon":
        polygons = [geom for geom in geometry.geoms if geom.geom_type == "Polygon" and not geom.is_empty]
        return max(polygons, key=lambda geom: float(geom.area)) if polygons else Polygon()
    return geometry if geometry.geom_type == "Polygon" else Polygon()


def _polygon_union(polygons: list[Polygon] | tuple[Polygon, ...]) -> Polygon | MultiPolygon:
    if not polygons:
        return Polygon()
    geometry = clean_polygonal_geometry(unary_union(polygons))
    # Disconnected roofs can belong to one source building. Keep each component
    # in selection bounds, correction masks and LiDAR evidence envelopes.
    return geometry if geometry.geom_type == "MultiPolygon" else _largest_polygon(geometry)


def _bounds_xyz(points_xyz: list[tuple[float, float, float]] | np.ndarray) -> tuple[float, float, float, float, float, float]:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.min(points[:, 2])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
        float(np.max(points[:, 2])),
    )


def _close_surface_path(path: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], ...]:
    points = tuple((float(x), float(y), float(z)) for x, y, z in path)
    if len(points) >= 3 and points[0] != points[-1]:
        points = (*points, points[0])
    return points


def _surface_area_3d(points_xyz: tuple[tuple[float, float, float], ...]) -> float:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        return 0.0
    area_vector = np.zeros(3, dtype=np.float64)
    for index, point in enumerate(points):
        area_vector += np.cross(point, points[(index + 1) % len(points)])
    return 0.5 * float(np.linalg.norm(area_vector))


def surface_xy_line(face: SurfaceFace) -> LineString | None:
    """Return the dominant top-down line occupied by a vertical surface face."""

    coords: list[tuple[float, float]] = []
    for x, y, _z in face.points_xyz:
        xy = (round(float(x), 6), round(float(y), 6))
        if not coords or coords[-1] != xy:
            coords.append(xy)
    unique: list[tuple[float, float]] = []
    for xy in coords:
        if xy not in unique:
            unique.append(xy)
    if len(unique) < 2:
        return None
    best = (unique[0], unique[1])
    best_len = -1.0
    for left_index, left in enumerate(unique):
        for right in unique[left_index + 1 :]:
            length = float(np.hypot(left[0] - right[0], left[1] - right[1]))
            if length > best_len:
                best = (left, right)
                best_len = length
    return LineString(best) if best_len > 1e-9 else None


def _surface_faces_from_features(
    features: list[dict[str, Any]],
    *,
    building_fid: int,
    surface_kind: str,
) -> tuple[SurfaceFace, ...]:
    faces: list[SurfaceFace] = []
    for feature in features:
        for path_index, path in enumerate(feature["paths"]):
            points = _close_surface_path(path)
            if len(points) < 4:
                continue
            z_values = [point[2] for point in points]
            faces.append(
                SurfaceFace(
                    face_id=f"{surface_kind}:{int(building_fid)}:{len(faces) + 1}:{path_index}",
                    building_fid=int(building_fid),
                    surface_kind=str(surface_kind),
                    points_xyz=points,
                    area_m2=_surface_area_3d(points),
                    z_min=float(min(z_values)),
                    z_max=float(max(z_values)),
                )
            )
    return tuple(faces)


def build_house_index(vector_path: Path) -> pd.DataFrame:
    """Build the house index; roofs are flat when at least half their projected
    face area has slope <= 5 degrees. Roofs without usable faces are unlabelled.
    """

    rows: dict[int, dict[str, Any]] = {}
    footprint_parts: dict[int, list[Polygon]] = defaultdict(list)
    xyz_by_fid: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    roof_area: dict[int, float] = defaultdict(float)
    flat_roof_area: dict[int, float] = defaultdict(float)
    for feature in _iter_roof_features(vector_path):
        fid = int(feature["building_fid"])
        row = rows.setdefault(
            fid,
            {
                "building_fid": fid,
                "object_type": str(feature["object_type"]),
                "surface_count": 0,
                "roof_face_count": 0,
            },
        )
        row["surface_count"] = int(row["surface_count"]) + 1
        for path in feature["paths"]:
            polygon = _polygon_from_path(path)
            if polygon.is_empty or float(polygon.area) < MIN_ROOF_FACE_PROJECTED_AREA_M2:
                continue
            footprint_parts[fid].append(polygon)
            xyz_by_fid[fid].extend(path)
            row["roof_face_count"] = int(row["roof_face_count"]) + 1
            # Centre LV95 coordinates before fitting to keep the slope stable.
            points = np.asarray(path, dtype=np.float64)
            (a, b, _), _rmse = fit_plane(points - points[0])
            slope_deg = float(np.rad2deg(np.arctan(np.hypot(a, b))))
            roof_area[fid] += float(polygon.area)
            if slope_deg <= FLAT_ROOF_MAX_SLOPE_DEG:
                flat_roof_area[fid] += float(polygon.area)

    footprints: dict[int, Polygon] = {}
    for fid, row in rows.items():
        footprint = _polygon_union(tuple(footprint_parts.get(fid, ())))
        footprints[fid] = footprint
        bounds_xyz = _bounds_xyz(xyz_by_fid.get(fid, []))
        row["footprint_area_m2"] = float(footprint.area) if not footprint.is_empty else 0.0
        row["roof_type"] = ""
        if roof_area[fid] > 0.0:
            row["roof_type"] = "flat" if flat_roof_area[fid] >= 0.5 * roof_area[fid] else "pitched"
        row["is_independent"] = True
        row["touching_building_fids"] = ""
        row["touching_house_count"] = 0
        row["min_x"], row["min_y"], row["max_x"], row["max_y"] = (
            tuple(float(value) for value in footprint.bounds)
            if not footprint.is_empty
            else (0.0, 0.0, 0.0, 0.0)
        )
        row["min_z"] = bounds_xyz[2]
        row["max_z"] = bounds_xyz[5]

    fids = sorted(rows)
    touching: dict[int, set[int]] = {fid: set() for fid in fids}
    for left_index, fid in enumerate(fids):
        footprint = footprints[fid]
        if footprint.is_empty:
            continue
        for other_fid in fids[left_index + 1 :]:
            other = footprints[other_fid]
            if other.is_empty:
                continue
            if float(footprint.distance(other)) <= HOUSE_TOUCH_TOLERANCE_M:
                touching[fid].add(other_fid)
                touching[other_fid].add(fid)

    for fid, neighbors in touching.items():
        rows[fid]["is_independent"] = not bool(neighbors)
        rows[fid]["touching_house_count"] = len(neighbors)
        rows[fid]["touching_building_fids"] = ";".join(str(value) for value in sorted(neighbors))

    columns = [
        "building_fid",
        "object_type",
        "surface_count",
        "roof_face_count",
        "footprint_area_m2",
        "roof_type",
        "is_independent",
        "touching_house_count",
        "touching_building_fids",
        "min_x",
        "min_y",
        "max_x",
        "max_y",
        "min_z",
        "max_z",
    ]
    return pd.DataFrame(rows.values(), columns=columns).sort_values(["object_type", "building_fid"]).reset_index(drop=True)


def _union_find_parent(size: int) -> list[int]:
    return list(range(size))


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def build_roof_segments(faces: tuple[RoofFace, ...]) -> tuple[RoofSegment, ...]:
    if not faces:
        return tuple()
    parent = _union_find_parent(len(faces))
    normals = [plane_normal(face.plane_coeffs) for face in faces]
    for left in range(len(faces)):
        for right in range(left + 1, len(faces)):
            if float(faces[left].polygon_xy.distance(faces[right].polygon_xy)) > 0.08:
                continue
            angle_ok = abs(float(np.dot(normals[left], normals[right]))) >= float(np.cos(np.deg2rad(8.0)))
            if not angle_ok:
                continue
            center = np.asarray(faces[left].polygon_xy.centroid.coords[0], dtype=np.float64)
            z_gap = abs(faces[left].z_at(center[0], center[1]) - faces[right].z_at(center[0], center[1]))
            if z_gap <= 0.35:
                _union(parent, left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(faces)):
        grouped[_find(parent, index)].append(index)

    raw_segments: list[RoofSegment] = []
    for seg_index, indices in enumerate(grouped.values()):
        segment_faces = [faces[index] for index in indices]
        points = [point for face in segment_faces for point in face.points_xyz]
        coeffs, _rmse = fit_plane(points)
        polygon = clean_polygonal_geometry(unary_union([face.polygon_xy for face in segment_faces]))
        raw_segments.append(
            RoofSegment(
                segment_id=f"segment_{seg_index:02d}",
                face_ids=tuple(face.face_id for face in segment_faces),
                polygon_xy=polygon,
                plane_coeffs=coeffs,
                normal=tuple(float(value) for value in plane_normal(coeffs)),
                area_m2=float(polygon.area),
                is_base=False,
            )
        )

    if not raw_segments:
        return tuple()
    max_area = max(segment.area_m2 for segment in raw_segments)
    ordered = sorted(raw_segments, key=lambda item: item.area_m2, reverse=True)
    cumulative = 0.0
    total = sum(segment.area_m2 for segment in raw_segments)
    base_ids: set[str] = set()
    for segment in ordered:
        if segment.area_m2 >= 0.15 * max_area or cumulative / max(total, 1e-9) < 0.80:
            base_ids.add(segment.segment_id)
            cumulative += segment.area_m2
    return tuple(
        RoofSegment(
            segment_id=segment.segment_id,
            face_ids=segment.face_ids,
            polygon_xy=segment.polygon_xy,
            plane_coeffs=segment.plane_coeffs,
            normal=segment.normal,
            area_m2=segment.area_m2,
            is_base=segment.segment_id in base_ids,
        )
        for segment in raw_segments
    )


def synthesized_roof_drop_wall_faces(
    roof_segments: tuple[RoofSegment, ...],
    *,
    building_fid: int,
) -> tuple[SurfaceFace, ...]:
    """Build vertical wall faces where roof segments share a real height drop."""

    walls: list[SurfaceFace] = []
    for left_index, left in enumerate(roof_segments):
        for right in roof_segments[left_index + 1 :]:
            for line in _shared_boundary_lines(left.polygon_xy, right.polygon_xy):
                wall = _drop_wall_from_boundary(
                    left,
                    right,
                    line,
                    building_fid=int(building_fid),
                    wall_index=len(walls),
                )
                if wall is not None:
                    walls.append(wall)
    return tuple(walls)


def roof_drop_wall_faces(house: VectorHouse) -> tuple[SurfaceFace, ...]:
    """Return roof drops derived from segment topology and validated source wall patches."""

    exact = list(
        synthesized_roof_drop_wall_faces(
            tuple(house.roof_segments),
            building_fid=int(house.building_fid),
        )
    )
    exact_lines = [line for line in (surface_xy_line(face) for face in exact) if line is not None]
    source = [
        face
        for face in _source_wall_roof_drop_faces(house)
        if not _line_matches_any(surface_xy_line(face), exact_lines)
    ]
    source = [
        _copy_surface_face(face, face_id=f"source_drop_wall:{int(house.building_fid)}:{index:03d}")
        for index, face in enumerate(source)
    ]
    return tuple((*exact, *source))


def _source_wall_roof_drop_faces(house: VectorHouse) -> tuple[SurfaceFace, ...]:
    source_walls = tuple(face for face in house.wall_faces if not str(face.face_id).startswith("drop_wall:"))
    if not source_walls or not house.roof_segments:
        return tuple()

    segment_edges = _segment_boundary_edge_index(tuple(house.roof_segments))
    wall_edges = [_surface_topology_edges(face) for face in source_walls]
    components = _wall_face_components(wall_edges)

    drops: list[SurfaceFace] = []
    for component in components:
        contacts: dict[str, list[LineString]] = defaultdict(list)
        for face_index in component:
            for line in wall_edges[face_index]:
                for segment_id in segment_edges.get(_line_key(line), ()):
                    contacts[str(segment_id)].append(line)
        if len(contacts) < 2:
            continue
        ranked = sorted(
            contacts.items(),
            key=lambda item: sum(float(line.length) for line in item[1]),
            reverse=True,
        )
        left_id, left_lines = ranked[0]
        right_id, right_lines = ranked[1]
        left = _segment_by_id(house.roof_segments, left_id)
        right = _segment_by_id(house.roof_segments, right_id)
        if left is None or right is None:
            continue

        left_z = _segment_contact_z(left, left_lines)
        right_z = _segment_contact_z(right, right_lines)
        if abs(left_z - right_z) < DROP_WALL_MIN_HEIGHT_M:
            continue
        line = max((*left_lines, *right_lines), key=lambda item: float(item.length))
        if float(line.length) < DROP_WALL_MIN_LENGTH_M:
            continue
        z_min = min(left_z, right_z)
        z_max = max(left_z, right_z)
        start = line.coords[0]
        end = line.coords[-1]
        points = (
            (float(start[0]), float(start[1]), float(z_max)),
            (float(end[0]), float(end[1]), float(z_max)),
            (float(end[0]), float(end[1]), float(z_min)),
            (float(start[0]), float(start[1]), float(z_min)),
            (float(start[0]), float(start[1]), float(z_max)),
        )
        drops.append(
            SurfaceFace(
                face_id=f"source_drop_wall:{int(house.building_fid)}:{len(drops):03d}",
                building_fid=int(house.building_fid),
                surface_kind="wall",
                points_xyz=points,
                area_m2=_surface_area_3d(points),
                z_min=float(z_min),
                z_max=float(z_max),
            )
        )
    return tuple(drops)


def _segment_by_id(segments: tuple[RoofSegment, ...], segment_id: str) -> RoofSegment | None:
    for segment in segments:
        if str(segment.segment_id) == str(segment_id):
            return segment
    return None


def _segment_contact_z(segment: RoofSegment, lines: list[LineString]) -> float:
    values = []
    for line in lines:
        for x, y in line.coords:
            values.append(segment.z_at(float(x), float(y)))
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _copy_surface_face(face: SurfaceFace, *, face_id: str) -> SurfaceFace:
    return SurfaceFace(
        face_id=str(face_id),
        building_fid=face.building_fid,
        surface_kind=face.surface_kind,
        points_xyz=face.points_xyz,
        area_m2=face.area_m2,
        z_min=face.z_min,
        z_max=face.z_max,
    )


def _segment_boundary_edge_index(segments: tuple[RoofSegment, ...]) -> dict[EdgeKey, list[str]]:
    by_edge: dict[EdgeKey, list[str]] = defaultdict(list)
    for segment in segments:
        for line in _polygon_boundary_edges(segment.polygon_xy):
            by_edge[_line_key(line)].append(str(segment.segment_id))
    return by_edge


def _polygon_boundary_edges(polygon: BaseGeometry) -> tuple[LineString, ...]:
    edges = []
    for boundary in _boundary_line_parts(polygon.boundary):
        coords = list(boundary.coords)
        edges.extend(
            line
            for start, end in zip(coords, coords[1:], strict=False)
            if float((line := LineString([start, end])).length) >= DROP_WALL_MIN_LENGTH_M
        )
    return tuple(edges)


def _surface_topology_edges(face: SurfaceFace) -> tuple[LineString, ...]:
    coords: list[tuple[float, float]] = []
    for x, y, _z in face.points_xyz:
        xy = _xy_key((float(x), float(y)))
        if not coords or coords[-1] != xy:
            coords.append(xy)
    return tuple(
        line
        for start, end in zip(coords, coords[1:], strict=False)
        if start != end and float((line := LineString([start, end])).length) >= DROP_WALL_MIN_LENGTH_M
    )


def _wall_face_components(wall_edges: list[tuple[LineString, ...]]) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(wall_edges)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_edge: dict[EdgeKey, list[int]] = defaultdict(list)
    for face_index, edges in enumerate(wall_edges):
        for line in edges:
            by_edge[_line_key(line)].append(face_index)
    for face_indices in by_edge.values():
        first = face_indices[0]
        for face_index in face_indices[1:]:
            union(first, face_index)

    components: dict[int, list[int]] = defaultdict(list)
    for face_index in range(len(wall_edges)):
        components[find(face_index)].append(face_index)
    return tuple(tuple(indices) for indices in components.values())


def _line_matches_any(line: LineString | None, others: list[LineString]) -> bool:
    if line is None:
        return False
    line_key = _line_key(line)
    return any(_line_key(other) == line_key for other in others)


def _line_key(line: LineString) -> EdgeKey:
    coords = list(line.coords)
    return tuple(sorted((_xy_key(coords[0]), _xy_key(coords[-1]))))


def _xy_key(coord: tuple[float, float]) -> tuple[float, float]:
    return (round(float(coord[0]), TOPOLOGY_KEY_DECIMALS), round(float(coord[1]), TOPOLOGY_KEY_DECIMALS))


def _drop_wall_from_boundary(
    left: RoofSegment,
    right: RoofSegment,
    line: BaseGeometry,
    *,
    building_fid: int,
    wall_index: int,
) -> SurfaceFace | None:
    if float(line.length) < DROP_WALL_MIN_LENGTH_M:
        return None
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    start = coords[0]
    end = coords[-1]
    z_left = np.asarray([left.z_at(start[0], start[1]), left.z_at(end[0], end[1])], dtype=np.float64)
    z_right = np.asarray([right.z_at(start[0], start[1]), right.z_at(end[0], end[1])], dtype=np.float64)
    gap = np.abs(z_left - z_right)
    if float(np.median(gap)) < DROP_WALL_MIN_HEIGHT_M:
        return None
    upper = np.maximum(z_left, z_right)
    lower = np.minimum(z_left, z_right)
    points = (
        (float(start[0]), float(start[1]), float(upper[0])),
        (float(end[0]), float(end[1]), float(upper[1])),
        (float(end[0]), float(end[1]), float(lower[1])),
        (float(start[0]), float(start[1]), float(lower[0])),
        (float(start[0]), float(start[1]), float(upper[0])),
    )
    return SurfaceFace(
        face_id=f"drop_wall:{int(building_fid)}:{wall_index:03d}",
        building_fid=int(building_fid),
        surface_kind="wall",
        points_xyz=points,
        area_m2=_surface_area_3d(points),
        z_min=float(np.min(lower)),
        z_max=float(np.max(upper)),
    )


def _shared_boundary_lines(left: Polygon, right: Polygon) -> tuple[BaseGeometry, ...]:
    return _long_boundary_lines(left.boundary.intersection(right.boundary))


def _long_boundary_lines(geometry: BaseGeometry) -> tuple[BaseGeometry, ...]:
    return tuple(line for line in _boundary_line_parts(geometry) if float(line.length) >= DROP_WALL_MIN_LENGTH_M)


def _boundary_line_parts(geometry: BaseGeometry) -> tuple[BaseGeometry, ...]:
    if geometry.is_empty:
        return tuple()
    if geometry.geom_type == "LineString":
        return (geometry,)
    if geometry.geom_type == "LinearRing":
        return (LineString(geometry),)
    if geometry.geom_type == "MultiLineString":
        return tuple(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return tuple(part for item in geometry.geoms for part in _boundary_line_parts(item))
    return tuple()


def load_vector_house(vector_path: Path, building_fid: int, *, index: pd.DataFrame | None = None) -> VectorHouse:
    features = _iter_roof_features(vector_path, building_fid=int(building_fid))
    wall_features = _iter_surface_features(
        vector_path,
        layer_name="Wall",
        building_fid=int(building_fid),
        required=False,
    )
    floor_features = _iter_surface_features(
        vector_path,
        layer_name="Floor",
        building_fid=int(building_fid),
        required=False,
    )
    faces: list[RoofFace] = []
    object_type = ""
    for feature in features:
        object_type = str(feature["object_type"])
        for path_index, path in enumerate(feature["paths"]):
            polygon = _polygon_from_path(path)
            if polygon.is_empty or float(polygon.area) < MIN_ROOF_FACE_PROJECTED_AREA_M2:
                continue
            try:
                coeffs, rmse = fit_plane(path)
            except ValueError:
                continue
            faces.append(
                RoofFace(
                    face_id=f"Roof:{int(building_fid)}:{len(faces) + 1}:{path_index}",
                    building_fid=int(building_fid),
                    polygon_xy=polygon,
                    points_xyz=tuple((float(x), float(y), float(z)) for x, y, z in path),
                    plane_coeffs=coeffs,
                    vertex_rmse_m=rmse,
                    area_m2=float(polygon.area),
                )
            )
    if not faces:
        raise RuntimeError(f"No roof faces found for building_fid={building_fid}")

    segments = build_roof_segments(tuple(faces))
    envelope = _polygon_union(tuple(face.polygon_xy for face in faces))
    bounds_xy = tuple(float(value) for value in envelope.bounds)
    wall_faces = _surface_faces_from_features(wall_features, building_fid=int(building_fid), surface_kind="wall")
    floor_faces = _surface_faces_from_features(floor_features, building_fid=int(building_fid), surface_kind="floor")
    bounds_xyz = _bounds_xyz(
        [
            point
            for face in (*faces, *wall_faces, *floor_faces)
            for point in face.points_xyz
        ]
    )

    is_independent = True
    touching: tuple[int, ...] = ()
    if index is not None and len(index):
        rows = index.loc[index["building_fid"].astype(int) == int(building_fid)]
        if not rows.empty:
            row = rows.iloc[0]
            is_independent = bool(row.get("is_independent", True))
            touching = _parse_touching_building_fids(row.get("touching_building_fids", ""))
    return VectorHouse(
        building_fid=int(building_fid),
        object_type=object_type,
        roof_faces=tuple(faces),
        roof_segments=segments,
        roof_envelope=envelope,
        bounds_xy=bounds_xy,
        bounds_xyz=bounds_xyz,
        is_independent=is_independent,
        touching_building_fids=touching,
        wall_faces=wall_faces,
        floor_faces=floor_faces,
    )


def load_vector_house_scaffold(scaffold_path: str | Path, building_fid: int | None = None) -> VectorHouse:
    """Load a canonical roof-scaffold GeoJSON as a Emboss vector house."""

    path = Path(scaffold_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = list(payload.get("features") or [])
    envelope_feature = _first_feature(features, "roof_envelope")
    building = _coerce_building_fid(
        building_fid
        if building_fid is not None
        else _feature_properties(envelope_feature).get("building_fid")
    )

    faces = _scaffold_roof_faces(features, building_fid=building)
    if not faces:
        raise RuntimeError(f"No roof faces found in scaffold {path}")

    segments = _scaffold_roof_segments(features, faces=tuple(faces))
    wall_faces = _scaffold_surface_faces(features, kind="wall_face", building_fid=building, surface_kind="wall")
    floor_faces = _scaffold_surface_faces(features, kind="floor_face", building_fid=building, surface_kind="floor")
    envelope = _scaffold_envelope(envelope_feature, faces)
    points = [
        point
        for face in (*faces, *wall_faces, *floor_faces)
        for point in face.points_xyz
    ]
    return VectorHouse(
        building_fid=building,
        object_type=str(_feature_properties(envelope_feature).get("provider") or "roof_scaffold"),
        roof_faces=tuple(faces),
        roof_segments=segments,
        roof_envelope=envelope,
        bounds_xy=tuple(float(value) for value in envelope.bounds),
        bounds_xyz=_bounds_xyz(points),
        is_independent=bool(_feature_properties(envelope_feature).get("is_independent", True)),
        touching_building_fids=_parse_touching_building_fids(
            _feature_properties(envelope_feature).get("touching_building_fids", "")
        ),
        wall_faces=wall_faces,
        floor_faces=floor_faces,
    )


def _first_feature(features: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for item in features:
        if _feature_properties(item).get("kind") == kind:
            return item
    return None


def _feature_properties(feature: dict[str, Any] | None) -> dict[str, Any]:
    return dict((feature or {}).get("properties") or {})


def _coerce_building_fid(value: Any) -> int:
    if value is None or str(value).strip() == "":
        return -1
    return int(float(value))


def _feature_polygon(feature: dict[str, Any]) -> Polygon:
    geometry = clean_polygonal_geometry(shape(feature.get("geometry")))
    return _largest_polygon(geometry)


def _feature_polygonal(feature: dict[str, Any]) -> Polygon | MultiPolygon:
    geometry = clean_polygonal_geometry(shape(feature.get("geometry")))
    return geometry if geometry.geom_type == "MultiPolygon" else _largest_polygon(geometry)


def _geojson_surface_paths(geometry: dict[str, Any] | None) -> list[list[tuple[float, float, float]]]:
    if not geometry:
        return []
    geom_type = str(geometry.get("type") or "")
    coordinates = geometry.get("coordinates")
    if geom_type == "Polygon":
        rings = coordinates or []
        return [[_coord_xyz(coord) for coord in rings[0]]] if rings else []
    if geom_type == "MultiPolygon":
        paths = []
        for polygon in coordinates or []:
            if polygon:
                paths.append([_coord_xyz(coord) for coord in polygon[0]])
        return paths
    return []


def _plane_from_properties(properties: dict[str, Any], polygon: Polygon | MultiPolygon) -> tuple[float, float, float]:
    raw = properties.get("plane_coeffs")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return tuple(float(value) for value in raw)
    parts = tuple(polygon.geoms) if polygon.geom_type == "MultiPolygon" else (polygon,)
    points = [(float(x), float(y), 0.0) for part in parts for x, y in part.exterior.coords]
    coeffs, _rmse = fit_plane(points)
    return coeffs


def _scaffold_points_xyz(polygon: Polygon, plane: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    a, b, c = plane
    return tuple(
        (float(x), float(y), float(a * float(x) + b * float(y) + c))
        for x, y in polygon.exterior.coords
    )


def _scaffold_surface_faces(
    features: list[dict[str, Any]],
    *,
    kind: str,
    building_fid: int,
    surface_kind: str,
) -> tuple[SurfaceFace, ...]:
    faces: list[SurfaceFace] = []
    for index, item in enumerate(features):
        properties = _feature_properties(item)
        if properties.get("kind") != kind:
            continue
        for path_index, path in enumerate(_geojson_surface_paths(item.get("geometry"))):
            points = _close_surface_path(path)
            if len(points) < 4:
                continue
            z_values = [point[2] for point in points]
            faces.append(
                SurfaceFace(
                    face_id=str(properties.get("face_id") or f"{surface_kind}_{index:03d}_{path_index:02d}"),
                    building_fid=building_fid,
                    surface_kind=surface_kind,
                    points_xyz=points,
                    area_m2=float(properties.get("area_m2") or _surface_area_3d(points)),
                    z_min=float(properties.get("z_min") if properties.get("z_min") is not None else min(z_values)),
                    z_max=float(properties.get("z_max") if properties.get("z_max") is not None else max(z_values)),
                )
            )
    return tuple(faces)


def _scaffold_roof_faces(features: list[dict[str, Any]], *, building_fid: int) -> list[RoofFace]:
    faces = []
    for index, item in enumerate(features):
        properties = _feature_properties(item)
        if properties.get("kind") != "roof_face":
            continue
        polygon = _feature_polygon(item)
        if polygon.is_empty or float(polygon.area) < MIN_ROOF_FACE_PROJECTED_AREA_M2:
            continue
        plane = _plane_from_properties(properties, polygon)
        face_id = str(properties.get("face_id") or f"scaffold_face_{index:03d}")
        faces.append(
            RoofFace(
                face_id=face_id,
                building_fid=building_fid,
                polygon_xy=polygon,
                points_xyz=_scaffold_points_xyz(polygon, plane),
                plane_coeffs=plane,
                vertex_rmse_m=float(properties.get("vertex_rmse_m") or 0.0),
                area_m2=float(polygon.area),
            )
        )
    return faces


def _scaffold_roof_segments(features: list[dict[str, Any]], *, faces: tuple[RoofFace, ...]) -> tuple[RoofSegment, ...]:
    segments = []
    for index, item in enumerate(features):
        properties = _feature_properties(item)
        if properties.get("kind") != "roof_segment":
            continue
        polygon = _feature_polygonal(item)
        if polygon.is_empty:
            continue
        plane = _plane_from_properties(properties, polygon)
        normal = properties.get("normal")
        if not isinstance(normal, (list, tuple)) or len(normal) != 3:
            normal = tuple(float(value) for value in plane_normal(plane))
        segments.append(
            RoofSegment(
                segment_id=str(properties.get("segment_id") or f"segment_{index:03d}"),
                face_ids=tuple(str(value) for value in properties.get("face_ids") or ()),
                polygon_xy=polygon,
                plane_coeffs=plane,
                normal=tuple(float(value) for value in normal),
                area_m2=float(polygon.area),
                is_base=bool(properties.get("is_base", True)),
            )
        )
    return tuple(segments) if segments else build_roof_segments(faces)


def _scaffold_envelope(envelope_feature: dict[str, Any] | None, faces: list[RoofFace]) -> Polygon | MultiPolygon:
    if envelope_feature is not None:
        envelope = _feature_polygonal(envelope_feature)
        if not envelope.is_empty:
            return envelope
    return _polygon_union(tuple(face.polygon_xy for face in faces))


def house_index_rows(frame: pd.DataFrame) -> tuple[HouseIndexRow, ...]:
    rows: list[HouseIndexRow] = []
    for row in frame.itertuples(index=False):
        touching = _parse_touching_building_fids(getattr(row, "touching_building_fids", ""))
        rows.append(
            HouseIndexRow(
                building_fid=int(row.building_fid),
                object_type=str(row.object_type or ""),
                surface_count=int(row.surface_count),
                roof_face_count=int(row.roof_face_count),
                footprint_area_m2=float(row.footprint_area_m2),
                is_independent=bool(row.is_independent),
                touching_building_fids=touching,
                bounds_xy=(float(row.min_x), float(row.min_y), float(row.max_x), float(row.max_y)),
                bounds_xyz=(
                    float(row.min_x),
                    float(row.min_y),
                    float(row.min_z),
                    float(row.max_x),
                    float(row.max_y),
                    float(row.max_z),
                ),
            )
        )
    return tuple(rows)
