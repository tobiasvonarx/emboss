"""Prepare pinned tile rasters before house-level orthophoto correction.

Preserves the original tile+32 m raster grid, RGB direct window extraction,
and DSM/DTM 0.5 m -> 0.1 m two-stage resampling.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from filelock import FileLock

from building_data.storage import contained_path, download, stac_sha256, write_json

from .models import LV95_BOUNDS, CorrectionConfig, RasterAsset, StripCandidate
from .raster import read_raster_crop, write_geotiff
from .stac import (
    assets_to_metadata,
    resolve_correction_assets,
    source_year_from_id,
    source_years_by_proximity,
)
from .strips import filter_swissimage_strips, query_lubis_strips, strips_to_metadata


def _expand_bounds(bounds_xy: LV95_BOUNDS, padding_m: float) -> LV95_BOUNDS:
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_xy]
    pad = max(float(padding_m), 0.0)
    return (min_x - pad, min_y - pad, max_x + pad, max_y + pad)


def _primary_asset_year(assets: Iterable[RasterAsset]) -> int | None:
    years = sorted({int(asset.year) for asset in assets if asset.year is not None})
    return years[-1] if years else None


def _raster_assets_by_year(
    assets: Iterable[RasterAsset],
    *,
    target_gsd_m: float | None = None,
) -> dict[int, tuple[RasterAsset, ...]]:
    grouped: dict[int, list[RasterAsset]] = {}
    for asset in assets:
        if asset.year is None:
            continue
        if (
            target_gsd_m is not None
            and abs(float(asset.gsd_m) - float(target_gsd_m)) > 1e-6
        ):
            continue
        grouped.setdefault(int(asset.year), []).append(asset)
    return {
        int(year): tuple(sorted(values, key=lambda item: item.item_id))
        for year, values in sorted(grouped.items(), reverse=True)
    }


def _strips_by_year(
    strips: Iterable[StripCandidate],
) -> dict[int, tuple[StripCandidate, ...]]:
    grouped: dict[int, list[StripCandidate]] = {}
    for strip in strips:
        grouped.setdefault(int(strip.flight_year), []).append(strip)
    return {
        int(year): tuple(sorted(values, key=lambda item: item.strip_id))
        for year, values in sorted(grouped.items(), reverse=True)
    }


def select_compatible_swissimage_assets(
    assets: Iterable[RasterAsset],
    strips: Iterable[StripCandidate],
    *,
    target_gsd_m: float = 0.1,
    lidar_year: int | None = None,
) -> tuple[int, tuple[RasterAsset, ...], tuple[StripCandidate, ...]]:
    """Select the compatible SWISSIMAGE year closest to the LiDAR survey."""
    assets_by_year = _raster_assets_by_year(assets, target_gsd_m=float(target_gsd_m))
    strips_by_year = _strips_by_year(
        filter_swissimage_strips(
            tuple(strips),
            flight_year=None,
            target_gsd_m=float(target_gsd_m),
        )
    )
    compatible_years = set(assets_by_year) & set(strips_by_year)
    for year in source_years_by_proximity(compatible_years, lidar_year):
        return int(year), assets_by_year[year], strips_by_year[year]

    raise RuntimeError(
        "No SWISSIMAGE source year has matching SWISSIMAGE LUBIS strip metadata."
    )


def _correction_raster_dataset_specs(
    config: CorrectionConfig,
) -> dict[str, dict[str, Any]]:
    return {
        "swissimage": {
            "collection": config.swissimage_collection,
            "gsd_m": float(config.target_gsd_m),
            "resample_alg": "bilinear",
            "label": "SWISSIMAGE",
        },
        "surface": {
            "collection": config.surface_collection,
            "gsd_m": float(config.height_gsd_m),
            "resample_alg": "bilinear",
            "label": "swissSURFACE3D Raster DSM",
        },
        "terrain": {
            "collection": config.terrain_collection,
            "gsd_m": float(config.height_gsd_m),
            "resample_alg": "bilinear",
            "label": "swissALTI3D DTM",
        },
    }


def _patch_bounds(
    bounds: LV95_BOUNDS, required: LV95_BOUNDS | None, cfg: CorrectionConfig
) -> LV95_BOUNDS:
    expanded = list(_expand_bounds(bounds, 32.0))
    if required is not None:
        # Grow an edge crop by whole pixels of BOTH raster grids. Default 0.5 m
        # steps retain the 0.1 m RGB grid and the 0.5 m DSM/DTM grid exactly.
        from fractions import Fraction

        rgb_step = Fraction(str(cfg.target_gsd_m))
        height_step = Fraction(str(cfg.height_gsd_m))
        denominator = math.lcm(rgb_step.denominator, height_step.denominator)
        step = (
            math.lcm(int(rgb_step * denominator), int(height_step * denominator))
            / denominator
        )
        for axis in (0, 1):
            expanded[axis] -= (
                max(0, math.ceil((expanded[axis] - required[axis]) / step)) * step
            )
            expanded[axis + 2] += (
                max(0, math.ceil((required[axis + 2] - expanded[axis + 2]) / step))
                * step
            )
    return tuple(expanded)


def _restore_local_assets(
    root: Path, payload: dict
) -> dict[str, tuple[RasterAsset, ...]] | None:
    output = {}
    for key in ("swissimage", "surface", "terrain"):
        record = dict(payload.get("local_assets", {}).get(key, {}))
        if not record:
            return None
        path = contained_path(root, str(record["href"]))
        if not path.is_file():
            return None
        record["href"] = str(path)
        output[key] = (RasterAsset(**record),)
    return output


def prepare_assets(
    *,
    tile_key: str,
    bounds_lv95: LV95_BOUNDS,
    cache: Path,
    config: CorrectionConfig | None = None,
    progress: Callable[[str], None] = print,
    required_bounds_lv95: LV95_BOUNDS | None = None,
) -> dict[str, tuple[RasterAsset, ...]]:
    """Materialize the original correction grid with persistent source selections.

    ``bounds_lv95`` is the actual LiDAR tile-header bounds, as in the original
    acquisition path. ``required_bounds_lv95`` optionally grows that grid for a
    whole-house crop crossing the padded tile boundary, without shifting pixels.
    Completed caches contain only relative local paths and can be moved.
    """
    cfg = config or CorrectionConfig()
    patch_bounds = _patch_bounds(bounds_lv95, required_bounds_lv95, cfg)
    specs = _correction_raster_dataset_specs(cfg)
    identity = {
        "schema_version": 1,
        "tile_key": tile_key,
        "tile_bounds_lv95": list(bounds_lv95),
        "patch_bounds_lv95": list(patch_bounds),
        "datasets": specs,
        "stac_api_root": cfg.stac_api_root,
        "strip_layer_id": cfg.strip_layer_id,
        "strip_query_padding_m": cfg.strip_query_padding_m,
        "geoadmin_mapserver_root": cfg.geoadmin_mapserver_root,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(cache).expanduser().resolve() / "correction-rasters" / key
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    selection_path = root / "source-selection.json"
    with FileLock(str(root / "prepare.lock")):
        if manifest.is_file():
            payload = json.loads(manifest.read_text())
            if payload.get("identity") == identity:
                cached = _restore_local_assets(root, payload)
                if cached:
                    return cached
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text())
            if selection.get("identity") != identity:
                raise ValueError(
                    "Cached correction source identity does not match the request."
                )
        else:
            remote = resolve_correction_assets(patch_bounds, config=cfg)
            strips = query_lubis_strips(patch_bounds, config=cfg, flight_year=None)
            lidar_year = source_year_from_id(tile_key)
            year, primary, primary_strips = select_compatible_swissimage_assets(
                remote.get("swissimage", ()),
                strips,
                target_gsd_m=float(cfg.target_gsd_m),
                lidar_year=lidar_year,
            )
            selection = {
                "identity": identity,
                "source_assets": {
                    key: [
                        asdict(asset)
                        for asset in (
                            primary if key == "swissimage" else remote.get(key, ())
                        )
                    ]
                    for key in specs
                },
                "rgb_metadata": {
                    "source_year": year,
                    "lidar_source_year": lidar_year,
                    "orthophoto_lidar_year_delta": None
                    if lidar_year is None
                    else year - lidar_year,
                    "target_gsd_m": float(cfg.target_gsd_m),
                    "matching_lubis_strip_ids": [
                        strip.strip_id for strip in primary_strips
                    ],
                    "matching_lubis_strips": strips_to_metadata(primary_strips),
                },
            }
            write_json(selection_path, selection)
        local_records = {}
        for dataset_key, spec in specs.items():
            assets = [
                RasterAsset(**record)
                for record in selection["source_assets"][dataset_key]
            ]
            if not assets:
                raise RuntimeError(
                    f"No {spec['label']} assets found for the selected source patch."
                )
            progress(f"Preparing {spec['label']} correction raster for {tile_key}")
            source_paths = {}
            for asset in assets:
                if asset.href.startswith(("https://", "http://")):
                    progress(f"Caching complete raster {asset.item_id}")
                    source_paths[asset.href] = download(
                        asset.href,
                        Path(cache) / "raw-raster-assets",
                        sha256=stac_sha256(asset.metadata.get("checksum")),
                    )
            # These are the original resampling and serialization operations.
            array, extent = read_raster_crop(
                assets,
                bounds_lv95=patch_bounds,
                gsd_m=float(spec["gsd_m"]),
                resample_alg=str(spec["resample_alg"]),
                source_paths=source_paths,
            )
            patch_path = root / f"{dataset_key}.tif"
            temporary = root / f"{dataset_key}.partial.tif"
            write_geotiff(temporary, array, bounds_lv95=extent)
            temporary.replace(patch_path)
            source_ids = "+".join(asset.item_id for asset in assets)
            metadata = {
                "kind": "local_correction_raster_patch",
                "dataset_key": dataset_key,
                "tile_key": tile_key,
                "patch_bounds_lv95": list(patch_bounds),
                "extent_lv95": list(extent),
                "source_assets": assets_to_metadata(assets),
            }
            if dataset_key == "swissimage":
                metadata.update(selection["rgb_metadata"])
            local_records[dataset_key] = asdict(
                RasterAsset(
                    collection=str(spec["collection"]),
                    item_id=f"{tile_key}:{dataset_key}:{hashlib.sha256(source_ids.encode()).hexdigest()[:12]}",
                    asset_key=f"{dataset_key}_patch",
                    href=patch_path.name,
                    gsd_m=float(spec["gsd_m"]),
                    year=_primary_asset_year(assets),
                    epsg=2056,
                    metadata=metadata,
                )
            )
        payload = {"identity": identity, "local_assets": local_records}
        write_json(manifest, payload)
        restored = _restore_local_assets(root, payload)
        if restored is None:
            raise RuntimeError(
                "Prepared correction rasters disappeared before loading."
            )
        return restored
