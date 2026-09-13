"""Canonical CAD mesh helpers for eval artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
import shapely

from .scaffold_geometry import iter_scaffold_geometries
from .scaffold_geometry import load_scaffold_geometry
from .scaffold_geometry import safe_intersection
from .scaffold_geometry import shape_eval_geometry
from building_data.geometry import _iter_surface_features

VERTICAL_DROP_TOLERANCE_M = 0.02
ROOF_CONTACT_TOLERANCE_M = 0.02
TOP_EDGE_MATCH_TOLERANCE_M = 1e-4
GEOMETRY_PRECISION_M = 1e-6
MIN_POLYGON_AREA_M2 = 1e-8

Point3 = tuple[float, float, float]
Plane = tuple[float, float, float]


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, ...]]


@dataclass(frozen=True)
class RoofSurface:
    geometry: BaseGeometry
    plane_coeffs: Plane


@dataclass(frozen=True)
class TopPolygon:
    points_xyz: tuple[Point3, ...]
    support_plane: Plane | None = None


@dataclass(frozen=True)
class TopEdge:
    start: Point3
    end: Point3
    support_plane: Plane | None


def write_swissbuildings3d_mesh(
    *,
    surfaces_vector_path: str | Path,
    building_fid: int | str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a canonical PLY mesh from swissBUILDINGS3D roof, wall, and floor layers."""

    source_path = Path(surfaces_vector_path)
    target = Path(output_path)
    mesh = Mesh(vertices=[], faces=[])
    vertex_index: dict[tuple[float, float, float], int] = {}
    layer_face_counts: dict[str, int] = {}
    polygon_count = 0
    for layer_name in ("Roof", "Wall", "Floor"):
        before = len(mesh.faces)
        features = _iter_surface_features(
            source_path,
            layer_name=layer_name,
            building_fid=int(building_fid),
            required=False,
        )
        for feature in features:
            for path in feature["paths"]:
                ring = _clean_ring(path)
                if len(ring) < 3:
                    continue
                _append_polygon_faces(mesh, vertex_index, ring)
                polygon_count += 1
        layer_face_counts[layer_name] = len(mesh.faces) - before

    if not mesh.vertices or not mesh.faces:
        raise RuntimeError(f"No mesh surfaces found for building_fid={building_fid} in {source_path}")

    mesh = clean_mesh(mesh)
    write_ascii_ply_mesh(
        target,
        mesh,
        comments=(
            "crs EPSG:2056",
            "units m",
            "source swissBUILDINGS3D Roof Wall Floor layers",
            f"building_fid {int(building_fid)}",
        ),
    )
    return {
        "source": "swissbuildings3d",
        "building_fid": int(building_fid),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "polygons": polygon_count,
        "layer_faces": layer_face_counts,
    }




def write_prediction_mesh(
    *,
    roof_scaffold: Path,
    superstructures_path: str | Path,
    output_path: str | Path,
    base_mesh_path: str | Path | None = None,
    crs: str | None = None,
) -> dict[str, Any] | None:
    """Write a method mesh by combining the method scaffold/base mesh with predictions."""

    base_mesh = Path(base_mesh_path) if base_mesh_path is not None else None
    if base_mesh is not None and base_mesh.exists():
        mesh = read_ascii_ply_mesh(base_mesh)
        base_mesh_label = str(base_mesh)
        base_mesh_source = "mesh"
    else:
        mesh = _mesh_from_roof_scaffold(roof_scaffold)
        if mesh is None:
            return None
        base_mesh_label = str(roof_scaffold)
        base_mesh_source = "roof_scaffold"
    output = Path(output_path)
    superstructures = Path(superstructures_path)
    if not superstructures.exists() or _geojson_feature_count(superstructures) == 0:
        write_ascii_ply_mesh(
            output,
            mesh,
            comments=(
                f"crs {crs or 'EPSG:2056'}",
                "units m",
                f"base_{base_mesh_source} {base_mesh_label}",
                f"superstructures {superstructures}",
            ),
        )
        return {
            f"source_base_{base_mesh_source}": base_mesh_label,
            "superstructure_count": 0,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        }

    roof_surfaces = _load_roof_surfaces(roof_scaffold)
    min_z = min((vertex[2] for vertex in mesh.vertices), default=0.0)
    vertex_index = {_vertex_key(vertex): index for index, vertex in enumerate(mesh.vertices)}
    added_solids = 0
    skipped_solids = 0
    payload = json.loads(superstructures.read_text(encoding="utf-8"))
    for item in payload.get("features", []):
        geometry = item.get("geometry")
        if not geometry:
            continue
        properties = dict(item.get("properties") or {})
        added, skipped = _append_superstructure_mesh(
            mesh=mesh,
            vertex_index=vertex_index,
            geometry=shape_eval_geometry(geometry),
            properties=properties,
            roof_surfaces=roof_surfaces,
            fallback_z=min_z,
        )
        added_solids += added
        skipped_solids += skipped

    mesh = clean_mesh(mesh)
    write_ascii_ply_mesh(
        output,
        mesh,
        comments=(
            f"crs {crs or 'EPSG:2056'}",
            "units m",
            f"base_{base_mesh_source} {base_mesh_label}",
            f"superstructures {superstructures}",
        ),
    )
    return {
        f"source_base_{base_mesh_source}": base_mesh_label,
        "superstructure_count": added_solids,
        "skipped_superstructure_count": skipped_solids,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }


def read_ascii_ply_mesh(path: str | Path) -> Mesh:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Not an ASCII PLY mesh: {path}")
    vertex_count = 0
    face_count = 0
    header_end = None
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])
        elif line.strip() == "end_header":
            header_end = index + 1
            break
    if header_end is None:
        raise ValueError(f"PLY header is missing end_header: {path}")
    vertices = []
    for line in lines[header_end : header_end + vertex_count]:
        x, y, z = line.split()[:3]
        vertices.append((float(x), float(y), float(z)))
    faces = []
    face_start = header_end + vertex_count
    for line in lines[face_start : face_start + face_count]:
        parts = line.split()
        if not parts:
            continue
        count = int(parts[0])
        indices = tuple(int(value) for value in parts[1 : 1 + count])
        if len(indices) >= 3:
            faces.append(indices)
    return clean_mesh(Mesh(vertices=vertices, faces=faces))


def _mesh_from_roof_scaffold(path: Path) -> Mesh | None:
    mesh = Mesh(vertices=[], faces=[])
    vertex_index: dict[Point3, int] = {}
    for surface in _load_roof_surfaces(path):
        for polygon in _as_polygons(surface.geometry):
            ring = [
                (
                    float(x),
                    float(y),
                    _plane_z(surface.plane_coeffs, float(x), float(y)),
                )
                for x, y in polygon.exterior.coords
            ]
            _append_mesh_face(mesh, vertex_index, ring)
    mesh = clean_mesh(mesh)
    return mesh if mesh.vertices and mesh.faces else None


def clean_mesh(mesh: Mesh) -> Mesh:
    """Normalize mesh topology without dropping valid small geometry."""

    vertices: list[Point3] = []
    vertex_index: dict[Point3, int] = {}
    remap: dict[int, int] = {}
    for index, vertex in enumerate(mesh.vertices):
        key = _vertex_key(vertex)
        target = vertex_index.get(key)
        if target is None:
            target = len(vertices)
            vertex_index[key] = target
            vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
        remap[index] = target

    faces: list[tuple[int, ...]] = []
    seen_faces: set[tuple[int, ...]] = set()
    for face in mesh.faces:
        if any(index not in remap for index in face):
            continue
        remapped = [remap[index] for index in face]
        for cleaned in _simple_face_loops(remapped):
            face_key = _canonical_face_key(cleaned)
            if face_key in seen_faces:
                continue
            seen_faces.add(face_key)
            faces.append(cleaned)

    compact_vertices: list[Point3] = []
    compact_index: dict[int, int] = {}
    compact_faces: list[tuple[int, ...]] = []
    for face in faces:
        compact_face = []
        for index in face:
            target = compact_index.get(index)
            if target is None:
                target = len(compact_vertices)
                compact_index[index] = target
                compact_vertices.append(vertices[index])
            compact_face.append(target)
        compact_faces.append(tuple(compact_face))

    return Mesh(vertices=compact_vertices, faces=compact_faces)


def write_ascii_ply_mesh(path: str | Path, mesh: Mesh, *, comments: tuple[str, ...] = ()) -> None:
    """Normalize the mesh, then preserve its float64 coordinates losslessly."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = clean_mesh(mesh)
    lines = [
        "ply",
        "format ascii 1.0",
        *(f"comment {comment}" for comment in comments),
        f"element vertex {len(mesh.vertices)}",
        "property double x",
        "property double y",
        "property double z",
        f"element face {len(mesh.faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.17g} {y:.17g} {z:.17g}" for x, y, z in mesh.vertices)
    lines.extend(f"{len(face)} {' '.join(str(index) for index in face)}" for face in mesh.faces)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_ogr_polygon_surfaces(geometry: Any, ogr: Any) -> list[Any]:
    if geometry is None:
        return []
    candidate = geometry.Clone()
    try:
        candidate = candidate.GetLinearGeometry()
    except Exception:
        pass
    geom_type = ogr.GT_Flatten(candidate.GetGeometryType())
    geom_name = candidate.GetGeometryName().upper()
    if geom_type == ogr.wkbPolygon:
        return [candidate.Clone()]
    if geom_type == ogr.wkbMultiPolygon or geom_name in {"TRIANGLE", "TIN", "POLYHEDRALSURFACE", "TRIANGULATEDSURFACE"}:
        polygons: list[Any] = []
        for index in range(candidate.GetGeometryCount()):
            polygons.extend(_extract_ogr_polygon_surfaces(candidate.GetGeometryRef(index), ogr))
        return polygons
    if candidate.GetGeometryCount() > 0:
        polygons = []
        for index in range(candidate.GetGeometryCount()):
            polygons.extend(_extract_ogr_polygon_surfaces(candidate.GetGeometryRef(index), ogr))
        return polygons
    return []


def _polygon_outer_ring_points(polygon: Any) -> list[tuple[float, float, float]]:
    if polygon is None or polygon.GetGeometryCount() <= 0:
        return []
    ring = polygon.GetGeometryRef(0)
    if ring is None:
        return []
    return [
        (
            float(ring.GetX(index)),
            float(ring.GetY(index)),
            float(ring.GetZ(index)),
        )
        for index in range(ring.GetPointCount())
    ]


def _shapely_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]  # type: ignore[list-item]
    if geometry.geom_type == "MultiPolygon":
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]  # type: ignore[union-attr]
    if hasattr(geometry, "geoms"):
        polygons: list[Polygon] = []
        for part in geometry.geoms:  # type: ignore[union-attr]
            polygons.extend(_shapely_polygons(part))
        return polygons
    return []


def _shapely_outer_ring_points(polygon: Polygon) -> list[tuple[float, float, float]]:
    points = []
    for coord in polygon.exterior.coords:
        if len(coord) < 3:
            return []
        points.append((float(coord[0]), float(coord[1]), float(coord[2])))
    return points


def _xy_intersects(geometry: BaseGeometry, selector: BaseGeometry) -> bool:
    try:
        projected = shapely.force_2d(geometry)
        return bool(shapely.intersects(projected, selector))
    except Exception:
        return False


def _geojson_roof_envelope(path: Path) -> BaseGeometry | None:
    if not path.exists():
        return None
    try:
        return load_scaffold_geometry(
            path,
            preferred_kinds=("roof_envelope",),
            fallback_kinds=("roof_segment", "roof_face"),
        )
    except ValueError:
        return None


def _clean_ring(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    cleaned: list[tuple[float, float, float]] = []
    for point in points:
        if cleaned and _vertex_key(cleaned[-1]) == _vertex_key(point):
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and _vertex_key(cleaned[0]) == _vertex_key(cleaned[-1]):
        cleaned.pop()
    return cleaned


def _clean_adjacent_duplicate_indices(indices: list[int]) -> list[int]:
    adjacent_cleaned: list[int] = []
    for index in indices:
        if adjacent_cleaned and adjacent_cleaned[-1] == index:
            continue
        adjacent_cleaned.append(index)
    if len(adjacent_cleaned) > 1 and adjacent_cleaned[0] == adjacent_cleaned[-1]:
        adjacent_cleaned.pop()
    return adjacent_cleaned


def _simple_face_loops(indices: list[int]) -> list[tuple[int, ...]]:
    loops: list[tuple[int, ...]] = []
    stack: list[int] = []
    positions: dict[int, int] = {}
    for index in _clean_adjacent_duplicate_indices(indices):
        repeat_at = positions.get(index)
        if repeat_at is None:
            positions[index] = len(stack)
            stack.append(index)
            continue

        loop = tuple(stack[repeat_at:])
        if len(loop) >= 3:
            loops.append(loop)
        stack = stack[: repeat_at + 1]
        positions = {value: offset for offset, value in enumerate(stack)}

    if len(stack) >= 3:
        loops.append(tuple(stack))
    return loops


def _simple_point_loops(points: list[Point3]) -> list[list[Point3]]:
    loops: list[list[Point3]] = []
    stack: list[Point3] = []
    positions: dict[Point3, int] = {}
    for point in _clean_ring(points):
        key = _vertex_key(point)
        repeat_at = positions.get(key)
        if repeat_at is None:
            positions[key] = len(stack)
            stack.append(point)
            continue

        loop = stack[repeat_at:]
        if len(loop) >= 3:
            loops.append(loop)
        stack = stack[: repeat_at + 1]
        positions = {_vertex_key(value): offset for offset, value in enumerate(stack)}

    if len(stack) >= 3:
        loops.append(stack)
    return loops


def _canonical_face_key(face: tuple[int, ...]) -> tuple[int, ...]:
    return min(_minimum_rotation(face), _minimum_rotation(tuple(reversed(face))))


def _minimum_rotation(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return tuple()
    return min(values[index:] + values[:index] for index in range(len(values)))


def _append_polygon_faces(
    mesh: Mesh,
    vertex_index: dict[Point3, int],
    ring: list[Point3],
) -> bool:
    return _append_mesh_face(mesh, vertex_index, ring)


def _append_mesh_face(
    mesh: Mesh,
    vertex_index: dict[Point3, int],
    ring: list[Point3] | tuple[Point3, ...],
) -> bool:
    added = False
    for points in _simple_point_loops(list(ring)):
        mesh.faces.append(tuple(_add_vertex(mesh, vertex_index, point) for point in points))
        added = True
    return added


def _append_superstructure_mesh(
    *,
    mesh: Mesh,
    vertex_index: dict[tuple[float, float, float], int],
    geometry: BaseGeometry,
    properties: dict[str, Any],
    roof_surfaces: tuple[RoofSurface, ...],
    fallback_z: float,
) -> tuple[int, int]:
    top_polygons = _superstructure_top_polygons(
        geometry=geometry,
        properties=properties,
        roof_surfaces=roof_surfaces,
        fallback_z=fallback_z,
    )
    if not top_polygons:
        return 0, max(1, len(_as_polygons(geometry)))

    top_face_count = 0
    for top_polygon in top_polygons:
        if _append_mesh_face(mesh, vertex_index, top_polygon.points_xyz):
            top_face_count += 1
    if top_face_count == 0:
        return 0, max(1, len(top_polygons))

    for side in _superstructure_side_polygons(top_polygons, roof_surfaces=roof_surfaces, fallback_z=fallback_z):
        _append_mesh_face(mesh, vertex_index, side)
    return 1, 0


def _superstructure_top_polygons(
    *,
    geometry: BaseGeometry,
    properties: dict[str, Any],
    roof_surfaces: tuple[RoofSurface, ...],
    fallback_z: float,
) -> list[TopPolygon]:
    top_polygons: list[TopPolygon] = []
    for top_face in _top_face_specs(properties):
        top_polygons.extend(_top_polygons_for_plane(top_face[0], top_face[1], roof_surfaces))
    if top_polygons:
        return top_polygons

    top_plane = _plane_from_value(properties.get("top_plane"))
    height_offset = _height_offset(properties)
    for polygon in _as_polygons(geometry):
        if top_plane is not None:
            top_polygons.extend(_top_polygons_for_plane(polygon, top_plane, roof_surfaces))
            continue
        if roof_surfaces:
            for surface in roof_surfaces:
                for part in _as_polygons(safe_intersection(polygon, surface.geometry)):
                    plane = (
                        surface.plane_coeffs[0],
                        surface.plane_coeffs[1],
                        surface.plane_coeffs[2] + height_offset,
                    )
                    top_polygons.extend(_top_polygons_for_plane(part, plane, (surface,)))
        else:
            plane = (0.0, 0.0, float(fallback_z) + height_offset)
            top_polygons.extend(_top_polygons_for_plane(polygon, plane, tuple()))
    return top_polygons


def _top_face_specs(properties: dict[str, Any]) -> list[tuple[Polygon, tuple[float, float, float]]]:
    specs = []
    raw_faces = properties.get("top_faces")
    if not isinstance(raw_faces, list):
        return specs
    for item in raw_faces:
        if not isinstance(item, dict):
            continue
        plane = _plane_from_value(item.get("plane"))
        polygon = _polygon_from_xy(item.get("footprint_xy"))
        if plane is not None and polygon is not None:
            specs.append((polygon, plane))
    return specs


def _top_polygons_for_plane(
    polygon: Polygon,
    top_plane: Plane,
    roof_surfaces: tuple[RoofSurface, ...],
) -> list[TopPolygon]:
    if polygon.is_empty or float(polygon.area) <= 1e-8:
        return []
    if not roof_surfaces:
        top_polygon = _top_polygon_from_xy(polygon, top_plane, support_plane=None)
        return [top_polygon] if top_polygon is not None else []

    top_polygons: list[TopPolygon] = []
    for surface in roof_surfaces:
        for overlap in _as_polygons(safe_intersection(polygon, surface.geometry)):
            for clipped in _clip_polygon_above_roof(overlap, top_plane, surface.plane_coeffs):
                top_polygon = _top_polygon_from_xy(clipped, top_plane, support_plane=surface.plane_coeffs)
                if top_polygon is not None:
                    top_polygons.append(top_polygon)
    return top_polygons


def _top_polygon_from_xy(polygon: Polygon, plane: Plane, *, support_plane: Plane | None) -> TopPolygon | None:
    parts = _clean_xy_polygons(polygon)
    if not parts:
        return None
    polygon = max(parts, key=lambda part: float(part.area))
    coords = [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
    points = tuple((x, y, _plane_z(plane, x, y)) for x, y in coords)
    if len(_clean_ring(list(points))) < 3:
        return None
    return TopPolygon(points, support_plane=support_plane)


def _clip_polygon_above_roof(
    polygon: Polygon,
    top_plane: tuple[float, float, float],
    roof_plane: tuple[float, float, float],
) -> list[Polygon]:
    coords = [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
    if len(coords) < 3:
        return []

    def clearance(point: tuple[float, float]) -> float:
        x, y = point
        return _plane_z(top_plane, x, y) - _plane_z(roof_plane, x, y) - ROOF_CONTACT_TOLERANCE_M

    clipped: list[tuple[float, float]] = []
    previous = coords[-1]
    previous_value = clearance(previous)
    previous_inside = previous_value >= -1e-9
    for current in coords:
        current_value = clearance(current)
        current_inside = current_value >= -1e-9
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-12:
                t = previous_value / denominator
                clipped.append(
                    (
                        previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside

    if len(clipped) < 3:
        return []
    return _clean_xy_polygons(Polygon(clipped))


def _superstructure_side_polygons(
    top_polygons: list[TopPolygon],
    *,
    roof_surfaces: tuple[RoofSurface, ...],
    fallback_z: float,
) -> list[tuple[tuple[float, float, float], ...]]:
    sides = []
    edges = _top_edges(top_polygons)
    processed_shared_edges: set[tuple[int, int]] = set()
    for edge_index, edge in enumerate(edges):
        top_left = edge.start
        top_right = edge.end
        match = _matching_top_edge(edges, edge_index)
        if match is not None:
            other_index, other_left, other_right = match
            key = tuple(sorted((edge_index, other_index)))
            if key in processed_shared_edges:
                continue
            processed_shared_edges.add(key)
            left_gap = abs(top_left[2] - other_left[2])
            right_gap = abs(top_right[2] - other_right[2])
            if max(left_gap, right_gap) > VERTICAL_DROP_TOLERANCE_M:
                sides.append((top_left, top_right, other_right, other_left))
            continue

        base_left = _base_vertex_for_top_vertex(
            top_left,
            support_plane=edge.support_plane,
            roof_surfaces=roof_surfaces,
            fallback_z=fallback_z,
        )
        base_right = _base_vertex_for_top_vertex(
            top_right,
            support_plane=edge.support_plane,
            roof_surfaces=roof_surfaces,
            fallback_z=fallback_z,
        )
        if max(top_left[2] - base_left[2], top_right[2] - base_right[2]) > VERTICAL_DROP_TOLERANCE_M:
            sides.append((base_left, base_right, top_right, top_left))
    return sides


def _top_edges(
    top_polygons: list[TopPolygon],
) -> list[TopEdge]:
    edges = []
    for polygon in top_polygons:
        points = polygon.points_xyz
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            if _same_xy(point, next_point):
                continue
            edges.append(TopEdge(point, next_point, polygon.support_plane))
    return edges


def _matching_top_edge(
    edges: list[TopEdge],
    edge_index: int,
) -> tuple[int, tuple[float, float, float], tuple[float, float, float]] | None:
    left = edges[edge_index].start
    right = edges[edge_index].end
    for other_index, other_edge in enumerate(edges):
        if other_index == edge_index:
            continue
        other_left = other_edge.start
        other_right = other_edge.end
        if _same_xy(left, other_left) and _same_xy(right, other_right):
            return other_index, other_left, other_right
        if _same_xy(left, other_right) and _same_xy(right, other_left):
            return other_index, other_right, other_left
    return None


def _same_xy(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5 <= TOP_EDGE_MATCH_TOLERANCE_M


def _base_vertex_for_top_vertex(
    top_vertex: tuple[float, float, float],
    *,
    support_plane: Plane | None,
    roof_surfaces: tuple[RoofSurface, ...],
    fallback_z: float,
) -> tuple[float, float, float]:
    x, y, _ = top_vertex
    if support_plane is not None:
        return (x, y, _plane_z(support_plane, x, y))
    return (x, y, _roof_z_at(roof_surfaces, x, y, fallback_z=fallback_z))


def _plane_from_value(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _polygon_from_xy(value: Any) -> Polygon | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    coords = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) < 2:
            return None
        try:
            coords.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return None
    parts = _clean_xy_polygons(Polygon(coords))
    return max(parts, key=lambda part: float(part.area), default=None)


def _height_offset(properties: dict[str, Any]) -> float:
    try:
        return max(float(properties.get("height_offset_m")), 0.0)
    except (TypeError, ValueError):
        return 0.1


def _as_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]  # type: ignore[list-item]
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return []
    return [polygon for part in geoms for polygon in _as_polygons(part)]


def _clean_xy_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    cleaned = geometry
    try:
        cleaned = shapely.set_precision(cleaned, GEOMETRY_PRECISION_M)
    except Exception:
        cleaned = geometry
    try:
        cleaned = cleaned.buffer(0)
    except Exception:
        try:
            cleaned = geometry.buffer(0)
        except Exception:
            return []
    return [part for part in _as_polygons(cleaned) if float(part.area) > MIN_POLYGON_AREA_M2]


def _load_roof_surfaces(path: Path) -> tuple[RoofSurface, ...]:
    segment_surfaces: list[RoofSurface] = []
    face_surfaces: list[RoofSurface] = []
    for properties, geometry in iter_scaffold_geometries(path, kinds=("roof_segment", "roof_face")):
        kind = properties.get("kind")
        plane = properties.get("plane_coeffs")
        if not isinstance(plane, list) or len(plane) != 3:
            continue
        surface = RoofSurface(
            geometry=geometry,
            plane_coeffs=(float(plane[0]), float(plane[1]), float(plane[2])),
        )
        if kind == "roof_segment":
            segment_surfaces.append(surface)
        else:
            face_surfaces.append(surface)
    return tuple(segment_surfaces or face_surfaces)


def _roof_z_at(surfaces: tuple[RoofSurface, ...], x: float, y: float, *, fallback_z: float) -> float:
    if not surfaces:
        return float(fallback_z)
    point = Point(float(x), float(y))
    containing = [surface for surface in surfaces if surface.geometry.buffer(1e-7).covers(point)]
    candidates = containing or sorted(surfaces, key=lambda surface: float(surface.geometry.distance(point)))[:1]
    z_values = [_plane_z(surface.plane_coeffs, x, y) for surface in candidates]
    return max(z_values) if z_values else float(fallback_z)


def _plane_z(plane: tuple[float, float, float], x: float, y: float) -> float:
    return float(plane[0] * float(x) + plane[1] * float(y) + plane[2])


def _add_vertex(
    mesh: Mesh,
    vertex_index: dict[Point3, int],
    point: Point3,
) -> int:
    key = _vertex_key(point)
    existing = vertex_index.get(key)
    if existing is not None:
        return existing
    vertex_index[key] = len(mesh.vertices)
    mesh.vertices.append((float(point[0]), float(point[1]), float(point[2])))
    return vertex_index[key]


def _vertex_key(point: Point3) -> Point3:
    return (round(float(point[0]), 6), round(float(point[1]), 6), round(float(point[2]), 6))


def _geojson_feature_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    return len(features) if isinstance(features, list) else 0
