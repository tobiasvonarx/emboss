"""Canonical metric building geometry and imagery inputs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon

@dataclass(frozen=True)
class RoofFace:
    face_id: str
    building_fid: int
    polygon_xy: Polygon
    points_xyz: tuple[tuple[float, float, float], ...]
    plane_coeffs: tuple[float, float, float]
    vertex_rmse_m: float
    area_m2: float

    def z_at(self, x: float, y: float) -> float:
        a, b, c = self.plane_coeffs
        return float(a * float(x) + b * float(y) + c)


@dataclass(frozen=True)
class RoofSegment:
    segment_id: str
    face_ids: tuple[str, ...]
    polygon_xy: Polygon | MultiPolygon
    plane_coeffs: tuple[float, float, float]
    normal: tuple[float, float, float]
    area_m2: float
    is_base: bool = False

    def z_at(self, x: float, y: float) -> float:
        a, b, c = self.plane_coeffs
        return float(a * float(x) + b * float(y) + c)

    def vertical_residual(self, xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=np.float64)
        return pts[:, 2] - (
            self.plane_coeffs[0] * pts[:, 0]
            + self.plane_coeffs[1] * pts[:, 1]
            + self.plane_coeffs[2]
        )


@dataclass(frozen=True)
class SurfaceFace:
    face_id: str
    building_fid: int
    surface_kind: str
    points_xyz: tuple[tuple[float, float, float], ...]
    area_m2: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class VectorHouse:
    building_fid: int
    object_type: str
    roof_faces: tuple[RoofFace, ...]
    roof_segments: tuple[RoofSegment, ...]
    roof_envelope: Polygon | MultiPolygon
    bounds_xy: tuple[float, float, float, float]
    bounds_xyz: tuple[float, float, float, float, float, float]
    is_independent: bool
    touching_building_fids: tuple[int, ...] = ()
    wall_faces: tuple[SurfaceFace, ...] = ()
    floor_faces: tuple[SurfaceFace, ...] = ()

    @property
    def base_segments(self) -> tuple[RoofSegment, ...]:
        return tuple(segment for segment in self.roof_segments if segment.is_base)


@dataclass(frozen=True)
class HouseIndexRow:
    building_fid: int
    object_type: str
    surface_count: int
    roof_face_count: int
    footprint_area_m2: float
    is_independent: bool
    touching_building_fids: tuple[int, ...]
    bounds_xy: tuple[float, float, float, float]
    bounds_xyz: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class OrthophotoCrop:
    rgb: np.ndarray
    extent_lv95: tuple[float, float, float, float]
    metadata: dict[str, Any]
    path: Path
