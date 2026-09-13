"""Source-provider boundary. Coordinates must be metric, with declared height datum."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

Progress = Callable[[str], None]
Bounds = tuple[float, float, float, float]


class BuildingProvider(Protocol):
    name: str
    crs: str
    vertical_crs: str
    revision: str

    def validate_selection(self, geometry) -> None:
        """Reject a longitude/latitude selection outside this provider's coverage."""
        ...

    def house_id(self, building_fid: int) -> str:
        """Stable, filesystem-safe identity including the vector source revision."""
        ...

    def buildings(self, bounds: Bounds, output: Path, progress: Progress) -> Path: ...

    def terrain(
        self, bounds: Bounds, output: Path, *, pinned_sources: tuple[dict, ...] = ()
    ) -> Path | None:
        """Optional ground raster covering bounds, in this provider's XY/height datum.

        Bounds already include the scene's ground-sampling margin. Return None
        when this provider does not supply terrain; do not substitute zero heights.
        """
        ...

    def lidar(
        self,
        bounds: Bounds,
        reference_year: int | None = None,
        progress: Progress = print,
        *,
        pinned_sources: tuple[dict, ...] = (),
    ) -> tuple[tuple[Path, ...], tuple[dict, ...]]: ...

    def points_for_bounds(
        self,
        bounds: Bounds,
        reference_year: int | None = None,
        progress: Progress = print,
        *,
        pinned_sources: tuple[dict, ...] = (),
    ) -> tuple[pd.DataFrame, tuple[dict, ...]]: ...
