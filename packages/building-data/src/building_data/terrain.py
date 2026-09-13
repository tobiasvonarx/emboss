"""Terrain-only scene export on the original annotation export's sampling grid.

The original acquisition warped swissALTI3D to the LiDAR header extent plus 32 m
at 0.5 m spacing. The final exporter took a rounded pixel window without another
resample. Keep those two operations separate: a direct warp to each house would
shift terrain samples and change ground extrusion elevations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import laspy
from filelock import FileLock
from osgeo import gdal

from .orthophoto_correction.models import CorrectionConfig, RasterAsset
from .orthophoto_correction.raster import read_raster_crop, write_geotiff
from .orthophoto_correction.stac import search_raster_assets
from .providers import Bounds
from .storage import download, stac_sha256, write_json


def terrain_patch_bounds(bounds: Bounds, sources: tuple[dict, ...]) -> Bounds:
    """Use the center source's real header, expanding only by full pixels."""
    if not sources:
        raise ValueError("Terrain grid requires a pinned LiDAR source")
    x, y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    anchor = next(
        (
            s
            for s in sources
            if s["bounds"][0] <= x < s["bounds"][2]
            and s["bounds"][1] <= y < s["bounds"][3]
        ),
        sources[0],
    )
    with laspy.open(anchor["path"]) as reader:
        x0, y0 = reader.header.mins[:2] - 32
        x1, y1 = reader.header.maxs[:2] + 32
    # Expansion cannot change the original bilinear target-grid phase.
    return (
        float(x0 - max(0, math.ceil((x0 - bounds[0]) / 0.5)) * 0.5),
        float(y0 - max(0, math.ceil((y0 - bounds[1]) / 0.5)) * 0.5),
        float(x1 + max(0, math.ceil((bounds[2] - x1) / 0.5)) * 0.5),
        float(y1 + max(0, math.ceil((bounds[3] - y1) / 0.5)) * 0.5),
    )


def crop_terrain(source: Path, output: Path, bounds: Bounds) -> Path:
    """GDAL equivalent of original from_bounds().round_offsets().round_lengths()."""
    dataset = gdal.Open(str(source))
    gt = dataset.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        raise ValueError("Terrain must use a north-up metric grid")
    x0 = math.floor((bounds[0] - gt[0]) / gt[1] + 0.001)
    y0 = math.floor((gt[3] - bounds[3]) / -gt[5] + 0.001)
    width = math.floor((bounds[2] - bounds[0]) / gt[1] + 0.5)
    height = math.floor((bounds[3] - bounds[1]) / -gt[5] + 0.5)
    x1, y1 = min(x0 + width, dataset.RasterXSize), min(y0 + height, dataset.RasterYSize)
    x0, y0 = max(x0, 0), max(y0, 0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Terrain does not intersect the requested scene")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = gdal.Translate(
        str(output),
        dataset,
        format="GTiff",
        srcWin=[x0, y0, x1 - x0, y1 - y0],
        creationOptions=["COMPRESS=DEFLATE"],
    )
    if result is None:
        raise RuntimeError("Could not write scene terrain")
    result = None
    dataset = None
    return output


def swiss_terrain(
    bounds: Bounds, output: Path, cache: Path, sources: tuple[dict, ...]
) -> Path:
    patch_bounds = terrain_patch_bounds(bounds, sources)
    identity = {
        "version": 1,
        "bounds": patch_bounds,
        "gsd": 0.5,
        "collection": "ch.swisstopo.swissalti3d",
        "vertical_crs": "EPSG:5728",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = cache / "terrain" / key
    root.mkdir(parents=True, exist_ok=True)
    patch = root / "terrain.tif"
    selection = root / "sources.json"
    with FileLock(str(root / "prepare.lock")):
        if not patch.exists():
            if selection.exists():
                assets = [
                    RasterAsset(**a)
                    for a in json.loads(selection.read_text())["assets"]
                ]
            else:
                assets = search_raster_assets(
                    patch_bounds,
                    collection=identity["collection"],
                    target_gsd_m=0.5,
                    config=CorrectionConfig(cache_root=cache),
                )
                if not assets:
                    raise ValueError("No swissALTI3D coverage for scene terrain")
                if any("2056_5728" not in a.href for a in assets):
                    raise ValueError(
                        "Terrain source does not declare the required EPSG:5728 height datum"
                    )
                write_json(
                    selection,
                    {"identity": identity, "assets": [asdict(a) for a in assets]},
                )
            paths = {
                a.href: download(
                    a.href,
                    cache / "imagery" / "raw-raster-assets",
                    sha256=stac_sha256(a.metadata.get("checksum")),
                    expected_size=a.metadata.get("size", a.metadata.get("file:size")),
                )
                for a in assets
            }
            values, extent = read_raster_crop(
                assets,
                bounds_lv95=patch_bounds,
                gsd_m=0.5,
                resample_alg="bilinear",
                source_paths=paths,
            )
            temporary = patch.with_suffix(".tmp.tif")
            try:
                write_geotiff(temporary, values, bounds_lv95=extent)
                temporary.replace(patch)
            finally:
                temporary.unlink(missing_ok=True)
    return crop_terrain(patch, output, bounds)
