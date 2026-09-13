"""Shared types for deterministic SWISSIMAGE Bildsturz correction."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np


LV95_BOUNDS = tuple[float, float, float, float]


def _default_cache_root() -> Path:
    return Path.home() / '.cache' / 'building-data' / 'orthophoto-correction'


def _default_mobile_sam_checkpoint() -> Path:
    return Path.home() / '.cache' / 'building-data' / 'models' / 'mobile_sam.pt'


@dataclass(frozen=True)
class CorrectionConfig:
    """Runtime configuration for first-pass orthophoto correction."""

    flight_height_m: float = 2400.0
    target_gsd_m: float = 0.10
    height_gsd_m: float = 0.5
    min_height_m: float = 0.5
    max_strip_distance_m: float = 1100.0
    strip_query_padding_m: float = 1200.0
    strip_id_override: str | None = None
    mobile_sam_checkpoint: Path = field(default_factory=_default_mobile_sam_checkpoint)
    mobile_sam_prompt_padding_m: float = 0.50
    min_crop_side_m: float = 18.0
    stac_api_root: str = "https://data.geo.admin.ch/api/stac/v1"
    geoadmin_mapserver_root: str = "https://api3.geo.admin.ch/rest/services/api/MapServer"
    swissimage_collection: str = "ch.swisstopo.swissimage-dop10"
    surface_collection: str = "ch.swisstopo.swisssurface3d-raster"
    terrain_collection: str = "ch.swisstopo.swissalti3d"
    strip_layer_id: str = "ch.swisstopo.lubis-bildstreifen"
    cache_root: Path = field(default_factory=_default_cache_root)
    local_strip_catalog: Path | None = None
    prefer_local_strip_catalog: bool = True
    user_agent: str = "pv-placement-emboss/orthophoto-correction"
    timeout_s: float = 30.0


@dataclass(frozen=True)
class RasterAsset:
    """One STAC raster asset selected for a correction crop."""

    collection: str
    item_id: str
    asset_key: str
    href: str
    gsd_m: float
    year: int | None
    epsg: int = 2056
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StripCandidate:
    """One LUBIS aerial strip centerline candidate."""

    strip_id: str
    flight_year: int
    flight_date: str
    gsd_m: float
    goal: str
    geometry_xy: np.ndarray
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DisplacementField:
    """Source sampling offsets for one flight strip in LV95 meters."""

    dx_m: np.ndarray
    dy_m: np.ndarray
    signed_distance_m: np.ndarray
    height_m: np.ndarray
    valid_mask: np.ndarray
    corrected_height_mask: np.ndarray

    @property
    def displacement_m(self) -> np.ndarray:
        return np.hypot(self.dx_m, self.dy_m)


@dataclass(frozen=True)
class CorrectionResult:
    """Corrected crop arrays plus provenance metadata."""

    rgb_corrected: np.ndarray
    valid_mask: np.ndarray
    occlusion_mask: np.ndarray
    corrected_height_mask: np.ndarray
    displacement: DisplacementField
    extent_lv95: LV95_BOUNDS
    transform: tuple[float, float, float, float, float, float]
    metadata: dict[str, Any]
    rgb_raw: np.ndarray | None = None

    @property
    def width(self) -> int:
        return int(self.rgb_corrected.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb_corrected.shape[0])
