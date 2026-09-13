"""Shared scaffold geometry loading and overlay helpers for eval methods."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import shapely
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from building_data.geometry import GEOMETRY_PRECISION_GRID_M
from building_data.geometry import clean_polygonal_geometry


ROOF_GEOMETRY_KINDS = frozenset({"roof_envelope", "roof_segment", "roof_face"})


def clean_eval_geometry(geometry: BaseGeometry | None) -> BaseGeometry:
    try:
        return clean_polygonal_geometry(geometry)
    except (GEOSException, ValueError):
        return Polygon()


def shape_eval_geometry(payload: Any) -> BaseGeometry:
    try:
        return clean_eval_geometry(shape(payload))
    except (GEOSException, TypeError, ValueError):
        return Polygon()


def polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(polygon for polygon in geometry.geoms if not polygon.is_empty)
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return ()
    return tuple(polygon for item in geoms for polygon in polygon_parts(item))


def safe_union(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    cleaned = [clean_eval_geometry(geometry) for geometry in geometries]
    cleaned = [geometry for geometry in cleaned if not geometry.is_empty]
    if not cleaned:
        return Polygon()
    try:
        return clean_eval_geometry(unary_union(cleaned))
    except (GEOSException, ValueError):
        try:
            return clean_eval_geometry(shapely.union_all(cleaned, grid_size=GEOMETRY_PRECISION_GRID_M))
        except (GEOSException, ValueError):
            return clean_eval_geometry(MultiPolygon([polygon for geometry in cleaned for polygon in polygon_parts(geometry)]))


def safe_intersection(left: BaseGeometry, right: BaseGeometry) -> BaseGeometry:
    left_clean = clean_eval_geometry(left)
    right_clean = clean_eval_geometry(right)
    if left_clean.is_empty or right_clean.is_empty:
        return Polygon()
    try:
        return clean_eval_geometry(left_clean.intersection(right_clean))
    except (GEOSException, ValueError):
        try:
            return clean_eval_geometry(
                shapely.intersection(left_clean, right_clean, grid_size=GEOMETRY_PRECISION_GRID_M)
            )
        except (GEOSException, ValueError):
            return Polygon()


def iter_scaffold_geometries(
    path: str | Path,
    *,
    kinds: set[str] | frozenset[str] | tuple[str, ...] | None = None,
) -> tuple[tuple[dict[str, Any], BaseGeometry], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(kinds) if kinds is not None else None
    rows: list[tuple[dict[str, Any], BaseGeometry]] = []
    for item in payload.get("features", []):
        properties = dict(item.get("properties") or {})
        if allowed is not None and properties.get("kind") not in allowed:
            continue
        geometry = item.get("geometry")
        if not geometry:
            continue
        cleaned = shape_eval_geometry(geometry)
        if cleaned.is_empty:
            continue
        rows.append((properties, cleaned))
    return tuple(rows)


def load_scaffold_geometry(
    path: str | Path,
    *,
    preferred_kinds: set[str] | frozenset[str] | tuple[str, ...] = ROOF_GEOMETRY_KINDS,
    fallback_kinds: set[str] | frozenset[str] | tuple[str, ...] | None = None,
) -> BaseGeometry:
    preferred_set = set(preferred_kinds)
    fallback_set = set(fallback_kinds) if fallback_kinds is not None else None
    preferred: list[BaseGeometry] = []
    fallback: list[BaseGeometry] = []
    for properties, geometry in iter_scaffold_geometries(path):
        kind = properties.get("kind")
        if kind in preferred_set:
            preferred.append(geometry)
        if fallback_set is None or kind in fallback_set:
            fallback.append(geometry)

    geometry = safe_union(preferred or fallback)
    if geometry.is_empty:
        raise ValueError(f"No usable scaffold geometry in {path}")
    return geometry
