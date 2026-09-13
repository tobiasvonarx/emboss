"""Emboss reconstruction result types."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from shapely.geometry import Point, Polygon
from building_data.models import RoofFace, RoofSegment, SurfaceFace, VectorHouse, HouseIndexRow, OrthophotoCrop

@dataclass(frozen=True)
class ReturnSupportModel:
    roof_band_m: float
    super_threshold_m: float
    return_xy: np.ndarray
    residuals: np.ndarray
    return_area_weights_m2: np.ndarray
    source_indices: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RectangularTopFace:
    face_id: str
    footprint_xy: tuple[tuple[float, float], ...]
    plane: tuple[float, float, float]

    def z_at(self, x: float, y: float) -> float:
        a, b, c = self.plane
        return float(a * float(x) + b * float(y) + c)


@dataclass(frozen=True)
class SuperstructureSolid:
    solid_id: str
    footprint_xy: tuple[tuple[float, float], ...]
    top_plane: tuple[float, float, float]
    host_segment_ids: tuple[str, ...]
    point_count: int
    height_offset_m: float
    fit_terms: dict[str, float]
    top_faces: tuple[RectangularTopFace, ...] = field(default_factory=tuple)
    source: str = "lidar"
    class_id: int | None = None
    class_label: str | None = None
    height_model: str = "lidar_fit"

    def z_top_at(self, x: float, y: float) -> float:
        if self.top_faces:
            point = Point(float(x), float(y))
            containing = [
                face
                for face in self.top_faces
                if Polygon(face.footprint_xy).buffer(1e-8).covers(point)
            ]
            if containing:
                return max(face.z_at(x, y) for face in containing)
            nearest = min(self.top_faces, key=lambda face: Polygon(face.footprint_xy).distance(point))
            return nearest.z_at(x, y)
        a, b, c = self.top_plane
        return float(a * float(x) + b * float(y) + c)


@dataclass(frozen=True)
class HouseResult:
    building_fid: int
    output_dir: Path
    vector_house: VectorHouse
    solids: tuple[SuperstructureSolid, ...]
    artifacts: dict[str, str]
