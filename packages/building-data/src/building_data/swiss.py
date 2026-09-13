"""Swiss public data provider. Full source tiles are cached and selections are pinned."""

from __future__ import annotations

import json
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from filelock import FileLock
from pyproj import Transformer

from . import building_repository
from .acquisition_sources import search_source_candidates
from .points import materialize_las_artifact, read_las_points
from .providers import Bounds, Progress
from .storage import download, write_json

TO_WGS84 = Transformer.from_crs(2056, 4326, always_xy=True)
TILE_PATTERN = re.compile(r"swisssurface3d_(\d{4})_(\d+)-(\d+)$")


def intersecting_tiles(bounds: Bounds) -> list[tuple[int, int]]:
    """Positive-area intersections with the Swiss 1 km grid (original convention)."""
    x0, y0, x1, y1 = bounds
    if not np.isfinite(bounds).all() or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid LiDAR bounds: {bounds}")
    return [
        (x, y)
        for x in range(
            int(np.floor(x0 / 1000)) * 1000, int(np.ceil(x1 / 1000)) * 1000, 1000
        )
        for y in range(
            int(np.floor(y0 / 1000)) * 1000, int(np.ceil(y1 / 1000)) * 1000, 1000
        )
    ]


class SwissProvider:
    name = "switzerland"
    crs = "EPSG:2056"
    vertical_crs = "EPSG:5728"
    revision = "swissbuildings3d-2-2024-05"

    def __init__(self, cache: Path, workers: int = 2):
        self.cache = Path(cache).resolve()
        self.workers = max(1, int(workers))
        self._tile_slots = threading.BoundedSemaphore(self.workers)
        self.cache.mkdir(parents=True, exist_ok=True)

    def validate_selection(self, geometry) -> None:
        from shapely.geometry import box

        if not box(5.8, 45.7, 10.7, 47.9).covers(geometry):
            raise ValueError("The installed data provider supports Switzerland only")

    def house_id(self, building_fid: int) -> str:
        return f"ch-sb3d2-202405-{int(building_fid)}"

    def terrain(
        self, bounds: Bounds, output: Path, *, pinned_sources: tuple[dict, ...] = ()
    ) -> Path | None:
        from .terrain import swiss_terrain

        if not pinned_sources:
            _, pinned_sources = self.lidar(bounds)
        return swiss_terrain(bounds, output, self.cache, pinned_sources)

    def buildings(
        self, bounds: Bounds, output: Path, progress: Progress = print
    ) -> Path:
        from .swiss_vectors import _clip_surface_layers

        root = self.cache / "swissbuildings3d" / building_repository.RELEASE
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / "acquire.lock")):
            source = building_repository.source_path(root)
            if source is None:
                progress(
                    "Preparing swissBUILDINGS3D 2.0 (3.6 GB download, cached after first use)"
                )
                building_repository.install(root)
                source = building_repository.source_path(root)
        progress("Selecting complete building geometry")
        return _clip_surface_layers(
            source_path=source, destination=output, tile_bounds_xy=bounds
        )

    def _tile(self, cell: tuple[int, int], reference_year: int | None) -> dict:
        # All area/house requests share one download budget, including nested calls.
        with self._tile_slots:
            return self._prepare_tile(cell, reference_year)

    def _prepare_tile(self, cell: tuple[int, int], reference_year: int | None) -> dict:
        x, y = cell
        catalog = (
            self.cache
            / "lidar-selections"
            / f"{x}_{y}_{reference_year or 'latest'}.json"
        )
        catalog.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(catalog) + ".lock"):
            if catalog.exists():
                tile = json.loads(catalog.read_text())
            else:
                lon, lat = TO_WGS84.transform(x + 500, y + 500)
                candidates = search_source_candidates(
                    "surfacePointCloud", longitude=lon, latitude=lat
                )
                matching = []
                for candidate in candidates:
                    match = TILE_PATTERN.fullmatch(candidate.item_id)
                    if (
                        match
                        and (int(match[2]) * 1000, int(match[3]) * 1000) == cell
                        and "2056_5728" in candidate.file_name
                    ):
                        matching.append((int(match[1]), candidate))
                if not matching:
                    raise ValueError(
                        f"No Swiss LiDAR coverage for required tile {x}, {y}"
                    )
                year, candidate = min(
                    matching,
                    key=lambda item: (
                        abs(item[0] - reference_year)
                        if reference_year is not None
                        else -item[0],
                        -item[0],
                        not item[1].file_name.endswith(".copc.laz"),
                        item[1].file_name,
                    ),
                )
                tile = {
                    "tile": candidate.item_id,
                    "year": year,
                    "bounds": [x, y, x + 1000, y + 1000],
                    "source": asdict(candidate),
                    "crs": self.crs,
                    "vertical_crs": self.vertical_crs,
                }
                write_json(catalog, tile)
        if tile["bounds"] != [x, y, x + 1000, y + 1000]:
            raise ValueError("Cached source tile has incorrect bounds")
        source = tile["source"]
        asset = download(source["asset_href"], self.cache / "assets")
        with FileLock(str(asset) + ".extract.lock"):
            path = materialize_las_artifact(asset, self.cache / "unpacked")
        return {**tile, "path": str(path)}

    def lidar(
        self,
        bounds: Bounds,
        reference_year: int | None = None,
        progress: Progress = print,
        *,
        pinned_sources: tuple[dict, ...] = (),
    ) -> tuple[tuple[Path, ...], tuple[dict, ...]]:
        cells = intersecting_tiles(bounds)
        progress(f"Preparing {len(cells)} LiDAR tile(s)")
        pinned = {tuple(s["bounds"][:2]): s for s in pinned_sources}
        if reference_year is None:
            center = (
                int(((bounds[0] + bounds[2]) / 2) // 1000) * 1000,
                int(((bounds[1] + bounds[3]) / 2) // 1000) * 1000,
            )
            anchor = pinned.get(center) or self._tile(center, None)
            pinned[center] = anchor
            reference_year = anchor["year"]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            sources = tuple(
                executor.map(
                    lambda cell: pinned.get(cell) or self._tile(cell, reference_year),
                    cells,
                )
            )
        for source in sources:
            path = Path(source["path"])
            x, y, x1, y1 = source["bounds"]
            with laspy.open(path) as reader:
                low = np.maximum(bounds[:2], [x, y])
                high = np.minimum(bounds[2:], [x1, y1])
                if np.any(reader.header.mins[:2] > low + 0.02) or np.any(
                    reader.header.maxs[:2] < high - 0.02
                ):
                    raise ValueError(
                        f"LiDAR file does not cover the requested extent: {source['tile']}"
                    )
            if reference_year is not None and source["year"] != reference_year:
                warnings.warn(
                    f"{source['tile']} uses year {source['year']}; target uses {reference_year}",
                    stacklevel=2,
                )
        return tuple(Path(s["path"]) for s in sources), sources

    def points_for_bounds(
        self,
        bounds: Bounds,
        reference_year: int | None = None,
        progress: Progress = print,
        *,
        pinned_sources: tuple[dict, ...] = (),
    ) -> tuple[pd.DataFrame, tuple[dict, ...]]:
        paths, sources = self.lidar(
            bounds, reference_year, progress, pinned_sources=pinned_sources
        )
        frames = []
        for path, source in zip(paths, sources, strict=True):
            frame = read_las_points(path, bounds_xy=bounds)
            x, y, x1, y1 = source["bounds"]
            frame = frame.loc[
                (frame.x >= x) & (frame.x < x1) & (frame.y >= y) & (frame.y < y1)
            ]
            if frame.empty:
                raise ValueError(
                    f"No LiDAR returns in required part of {source['tile']}"
                )
            frames.append(frame)
        # Repeated returns within a source are evidence with meaningful multiplicity.
        # Half-open ownership above removes overlaps between tiles without deleting them.
        return pd.concat(frames, ignore_index=True), sources
