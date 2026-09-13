"""GeoAdmin STAC v1 helpers for correction source rasters."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

from pyproj import Transformer

from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.models import LV95_BOUNDS
from building_data.orthophoto_correction.models import RasterAsset


LV95_TO_WGS84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def bounds_lv95_to_wgs84_bbox(bounds_lv95: LV95_BOUNDS) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_lv95]
    xs = [min_x, max_x, max_x, min_x]
    ys = [min_y, min_y, max_y, max_y]
    lon, lat = LV95_TO_WGS84.transform(xs, ys)
    return (float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat)))


def _get_json(url: str, *, params: dict[str, Any], config: CorrectionConfig) -> dict[str, Any]:
    query = urlencode(params, doseq=True)
    request_url = f"{url}?{query}" if query else url
    request = Request(request_url, headers={"User-Agent": config.user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=float(config.timeout_s)) as response:
        return json.loads(response.read().decode("utf-8"))


def _item_year(feature: dict[str, Any]) -> int | None:
    datetime_value = str(feature.get("properties", {}).get("datetime", "") or "")
    if len(datetime_value) >= 4 and datetime_value[:4].isdigit():
        return int(datetime_value[:4])
    return source_year_from_id(str(feature.get("id", "")))


def source_year_from_id(value: str) -> int | None:
    for part in str(value).split("_"):
        if len(part) == 4 and part.isdigit():
            return int(part)
    return None


def source_years_by_proximity(years: set[int], reference_year: int | None) -> list[int]:
    if reference_year is None:
        return sorted(years, reverse=True)
    return sorted(years, key=lambda year: (abs(year - reference_year), -year))


def _tile_suffix(item_id: str) -> str:
    parts = str(item_id).split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return "_".join(parts[2:])
    return str(item_id)


def asset_from_stac_feature(
    feature: dict[str, Any],
    *,
    collection: str,
    target_gsd_m: float,
    gsd_tolerance_m: float = 1e-6,
) -> RasterAsset | None:
    item_id = str(feature.get("id", ""))
    year = _item_year(feature)
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for key, payload in dict(feature.get("assets", {})).items():
        href = str(payload.get("href", "") or "")
        if not href.lower().endswith((".tif", ".tiff")):
            continue
        epsg = int(payload.get("proj:epsg", 2056) or 2056)
        if epsg != 2056:
            continue
        try:
            gsd = float(payload.get("gsd"))
        except (TypeError, ValueError):
            continue
        delta = abs(gsd - float(target_gsd_m))
        candidates.append((delta, str(key), dict(payload)))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0] > float(gsd_tolerance_m), item[0], item[1]))
    _delta, asset_key, asset_payload = candidates[0]
    return RasterAsset(
        collection=str(collection),
        item_id=item_id,
        asset_key=asset_key,
        href=str(asset_payload["href"]),
        gsd_m=float(asset_payload.get("gsd", target_gsd_m)),
        year=year,
        epsg=int(asset_payload.get("proj:epsg", 2056) or 2056),
        metadata={
            "created": asset_payload.get("created"),
            "updated": asset_payload.get("updated"),
            "checksum": asset_payload.get("file:checksum"),
            "size": asset_payload.get("file:size"),
        },
    )


def dedupe_latest_assets(assets: list[RasterAsset]) -> list[RasterAsset]:
    by_tile: dict[str, RasterAsset] = {}
    for asset in assets:
        suffix = _tile_suffix(asset.item_id)
        current = by_tile.get(suffix)
        if current is None:
            by_tile[suffix] = asset
            continue
        current_year = current.year if current.year is not None else -1
        asset_year = asset.year if asset.year is not None else -1
        if asset_year >= current_year:
            by_tile[suffix] = asset
    return sorted(by_tile.values(), key=lambda asset: asset.item_id)


def raster_assets_from_features(
    features: list[dict[str, Any]],
    *,
    collection: str,
    target_gsd_m: float,
    dedupe_latest: bool = True,
) -> list[RasterAsset]:
    assets = [
        asset
        for feature in features
        if (asset := asset_from_stac_feature(feature, collection=collection, target_gsd_m=target_gsd_m)) is not None
    ]
    return dedupe_latest_assets(assets) if bool(dedupe_latest) else sorted(assets, key=lambda asset: asset.item_id)


def search_raster_assets(
    bounds_lv95: LV95_BOUNDS,
    *,
    collection: str,
    target_gsd_m: float,
    config: CorrectionConfig,
    limit: int = 100,
    dedupe_latest: bool = True,
) -> list[RasterAsset]:
    bbox = bounds_lv95_to_wgs84_bbox(bounds_lv95)
    payload = _get_json(
        f"{config.stac_api_root.rstrip('/')}/collections/{collection}/items",
        params={
            "bbox": ",".join(f"{value:.8f}" for value in bbox),
            "limit": int(limit),
        },
        config=config,
    )
    return raster_assets_from_features(
        list(payload.get("features", ())),
        collection=collection,
        target_gsd_m=float(target_gsd_m),
        dedupe_latest=bool(dedupe_latest),
    )


def resolve_correction_assets(bounds_lv95: LV95_BOUNDS, *, config: CorrectionConfig) -> dict[str, list[RasterAsset]]:
    return {
        "swissimage": search_raster_assets(
            bounds_lv95,
            collection=config.swissimage_collection,
            target_gsd_m=float(config.target_gsd_m),
            config=config,
            dedupe_latest=False,
        ),
        "surface": search_raster_assets(
            bounds_lv95,
            collection=config.surface_collection,
            target_gsd_m=float(config.height_gsd_m),
            config=config,
        ),
        "terrain": search_raster_assets(
            bounds_lv95,
            collection=config.terrain_collection,
            target_gsd_m=float(config.height_gsd_m),
            config=config,
        ),
    }


def assets_to_metadata(assets: list[RasterAsset]) -> list[dict[str, Any]]:
    return [
        {
            "collection": asset.collection,
            "item_id": asset.item_id,
            "asset_key": asset.asset_key,
            "href": asset.href,
            "gsd_m": float(asset.gsd_m),
            "year": asset.year,
            "epsg": int(asset.epsg),
            "metadata": asset.metadata,
        }
        for asset in assets
    ]
