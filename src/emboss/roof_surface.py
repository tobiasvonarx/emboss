"""Upper exposed roof geometry, independent of mesh connectivity and semantics.

The original building remains the shading mesh. These functions construct a
separate surface for placement by partitioning projected boundaries and every
line where overlapping planes exchange vertical order. By default geometric
arithmetic uses a local origin; outputs retain the input coordinate system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import fsum

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

AREA_EPSILON = 1e-10
HEIGHT_EPSILON = 1e-8


@dataclass(frozen=True)
class RoofPiece:
    polygon_xy: Polygon
    source_index: int


@dataclass(frozen=True)
class ExposedRoof:
    triangles: np.ndarray
    source_face_ids: np.ndarray


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > AREA_EPSILON else []
    return [
        part for child in getattr(geometry, "geoms", ()) for part in _polygons(child)
    ]


def _ordered(polygons: Sequence[Polygon]) -> list[Polygon]:
    return sorted(polygons, key=lambda p: (*p.bounds, shapely.normalize(p).wkb_hex))


def _crossing_line(delta: np.ndarray, overlap: BaseGeometry) -> BaseGeometry | None:
    a, b, c = delta
    norm_squared = a * a + b * b
    if norm_squared <= 1e-18:
        return None
    minx, miny, maxx, maxy = overlap.bounds
    center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    origin = center - (a * center[0] + b * center[1] + c) * delta[:2] / norm_squared
    direction = np.array([-b, a]) / np.sqrt(norm_squared)
    span = max(maxx - minx, maxy - miny, 1.0) * 4
    return LineString(
        [origin - direction * span, origin + direction * span]
    ).intersection(overlap)


def _uppermost_local(
    polygons: Sequence[Polygon],
    planes: np.ndarray,
    source_ids: Sequence[int],
    *,
    intersection: Callable[[BaseGeometry, BaseGeometry], BaseGeometry] | None = None,
) -> list[RoofPiece]:
    if not polygons:
        return []
    tree = STRtree(polygons)
    intersect = intersection or (lambda left, right: left.intersection(right))
    boundaries: list[BaseGeometry] = [polygon.boundary for polygon in polygons]
    for left, polygon in enumerate(polygons):
        for right in sorted(int(i) for i in tree.query(polygon) if i > left):
            overlap = intersect(polygon, polygons[right])
            if overlap.area <= AREA_EPSILON:
                continue
            crossing = _crossing_line(planes[left] - planes[right], overlap)
            if crossing is not None and not crossing.is_empty:
                boundaries.append(crossing)
    selected = []
    for cell in polygonize(unary_union(boundaries)):
        if cell.area <= AREA_EPSILON:
            continue
        point = cell.representative_point()
        covering = [int(i) for i in tree.query(point) if polygons[int(i)].covers(point)]
        if not covering:
            continue
        heights = {
            i: float(planes[i, 0] * point.x + planes[i, 1] * point.y + planes[i, 2])
            for i in covering
        }
        highest = max(heights.values())
        # Numerical ties must not depend on triangulation or STRtree traversal.
        winner = min(
            (
                i
                for i in covering
                if highest - heights[i] <= (0.0 if intersection else HEIGHT_EPSILON)
            ),
            key=lambda i: (source_ids[i], i),
        )
        for polygon in _polygons(intersect(cell, polygons[winner])):
            selected.append(RoofPiece(polygon, winner))
    return selected


def uppermost_pieces(
    polygons_xy: Sequence[Polygon],
    planes: Sequence[Sequence[float]],
    *,
    source_ids: Sequence[int] | None = None,
    intersection: Callable[[BaseGeometry, BaseGeometry], BaseGeometry] | None = None,
) -> tuple[RoofPiece, ...]:
    """Partition polygon planes ``z=a*x+b*y+c`` and retain their upper envelope.

    ``source_index`` addresses the input sequence, including when two pieces
    share a source ID. Source IDs provide deterministic tie breaking only.
    A caller with an established coordinate precision grid may supply its
    intersection operation. In that case its original coordinates and exact
    height ordering are retained, avoiding shifts of the caller's grid.
    """
    if len(polygons_xy) != len(planes):
        raise ValueError("Every roof polygon requires one plane")
    if source_ids is not None and len(source_ids) != len(polygons_xy):
        raise ValueError("Every roof polygon requires one source ID")
    if not polygons_xy:
        return ()
    coefficients = np.asarray(planes, dtype=np.float64)
    if (
        coefficients.shape != (len(polygons_xy), 3)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("Roof planes must be finite triples")
    if intersection is not None:
        return tuple(
            _uppermost_local(
                polygons_xy,
                coefficients,
                source_ids if source_ids is not None else range(len(polygons_xy)),
                intersection=intersection,
            )
        )
    bounds = unary_union(polygons_xy).bounds
    x0, y0 = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    local_planes = coefficients.copy()
    local_planes[:, 2] = [fsum((a * x0, b * y0, c)) for a, b, c in coefficients]
    local = [translate(polygon, xoff=-x0, yoff=-y0) for polygon in polygons_xy]
    selected = _uppermost_local(
        local, local_planes, source_ids if source_ids is not None else range(len(local))
    )
    return tuple(
        RoofPiece(translate(p.polygon_xy, xoff=x0, yoff=y0), p.source_index)
        for p in selected
    )


def _triangle_planes(
    triangles: np.ndarray, min_abs_normal_z: float
) -> tuple[np.ndarray, np.ndarray]:
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(cross, axis=1)
    valid = (lengths > 1e-12) & (np.abs(cross[:, 2]) > 2 * AREA_EPSILON)
    valid &= np.abs(cross[:, 2]) >= min_abs_normal_z * lengths
    ids = np.flatnonzero(valid)
    normals = cross[ids]
    slopes = -normals[:, :2] / normals[:, 2, None]
    intercepts = triangles[ids, 0, 2] - np.sum(slopes * triangles[ids, 0, :2], axis=1)
    return ids, np.column_stack((slopes, intercepts))


def exposed_roof(
    vertices: np.ndarray, faces: np.ndarray, *, min_abs_normal_z: float = 0.15
) -> ExposedRoof:
    """Return uppermost rooflike triangles and their original face indices.

    Upward and downward windings are equivalent. Vertical walls and degenerate
    triangles are ignored; semantic exclusions (for example existing PV modules)
    must be applied by the caller before constructing a placement surface.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Vertices must be a finite Nx3 array")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError("Faces must be an integer Mx3 triangle array")
    if not 0 <= min_abs_normal_z <= 1:
        raise ValueError("Minimum absolute normal Z must lie in [0, 1]")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Face index outside the vertex array")
    if not len(faces):
        return ExposedRoof(np.empty((0, 3, 3)), np.empty(0, dtype=np.int64))
    origin = vertices[faces].reshape(-1, 3).mean(axis=0)
    triangles = vertices[faces] - origin
    ids, planes = _triangle_planes(triangles, min_abs_normal_z)
    polygons = [Polygon(triangles[i, :, :2]) for i in ids]
    pieces = _uppermost_local(polygons, planes, ids)
    output, sources = [], []
    for piece in pieces:
        plane = planes[piece.source_index]
        for polygon in _ordered(
            _polygons(shapely.constrained_delaunay_triangles(piece.polygon_xy))
        ):
            xy = np.asarray(polygon.exterior.coords)[:3]
            xyz = np.column_stack((xy, xy @ plane[:2] + plane[2])) + origin
            if np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])[2] < 0:
                xyz = xyz[[0, 2, 1]]
            output.append(xyz)
            sources.append(ids[piece.source_index])
    return ExposedRoof(
        np.asarray(output).reshape(-1, 3, 3), np.asarray(sources, dtype=np.int64)
    )


def covered_footprint(
    polygon_xy: BaseGeometry,
    plane_origin: Sequence[float],
    plane_normal: Sequence[float],
    occluder_triangles: np.ndarray,
    *,
    clearance_m: float = 1e-7,
) -> BaseGeometry:
    """Clip occluder footprints to portions above a particular roof plane.

    Clearance is a vertical height difference, not a setback. Callers apply
    physical setbacks in their roof's metric plane after this projection. An
    exposed dormer's own top therefore does not obstruct its placement surface.
    """
    origin = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    triangles = np.asarray(occluder_triangles, dtype=np.float64)
    if (
        origin.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite([origin, normal]).all()
        or abs(normal[2]) <= 1e-12
    ):
        raise ValueError("A finite nonvertical target plane is required")
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or not np.isfinite(triangles).all()
    ):
        raise ValueError("Occluders must be finite Nx3x3 triangles")
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("Clearance must be finite and nonnegative")
    if polygon_xy.is_empty or not len(triangles):
        return Polygon()
    target = translate(polygon_xy, xoff=-origin[0], yoff=-origin[1])
    local = triangles - origin
    covered = []
    for triangle in local:
        if (
            np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
            <= 1e-12
        ):
            continue
        # Clip in 3D before projection. This also retains elevated vertical
        # walls as lines, so a later metric buffer enforces their clearance.
        heights = (
            triangle[:, 2]
            + triangle[:, :2] @ (normal[:2] / normal[2])
            - clearance_m
            - HEIGHT_EPSILON
        )
        clipped = []
        for index in range(3):
            previous = (index - 1) % 3
            start, end = triangle[previous], triangle[index]
            start_height, end_height = heights[previous], heights[index]
            if (start_height >= 0) != (end_height >= 0):
                fraction = start_height / (start_height - end_height)
                clipped.append(start + fraction * (end - start))
            if end_height >= 0:
                clipped.append(end)
        if len(clipped) < 2:
            continue
        projected = MultiPoint(np.asarray(clipped)[:, :2]).convex_hull
        overlap = target.intersection(projected)
        if not overlap.is_empty:
            covered.append(overlap)
    return (
        translate(unary_union(covered), xoff=origin[0], yoff=origin[1])
        if covered
        else Polygon()
    )
