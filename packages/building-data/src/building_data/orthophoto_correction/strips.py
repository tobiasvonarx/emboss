"""GeoAdmin LUBIS strip retrieval and parsing helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

import numpy as np

from building_data.orthophoto_correction.geometry import expand_bounds
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.models import LV95_BOUNDS
from building_data.orthophoto_correction.models import StripCandidate


SWISS_LV95_QUERY_BOUNDS: LV95_BOUNDS = (2480000.0, 1070000.0, 2845000.0, 1305000.0)
DEFAULT_IDENTIFY_LIMIT = 200
DEFAULT_MIN_TILE_SIDE_M = 5000.0
DEFAULT_SEED_TILE_SIDE_M = 20000.0
DEFAULT_LUBIS_CATALOG_RELATIVE_PATH = Path("lubis") / "switzerland" / "lubis_strips.geojson"


def _get_json(url: str, *, params: dict[str, Any], config: CorrectionConfig) -> dict[str, Any]:
    query = urlencode(params, doseq=True)
    request_url = f"{url}?{query}" if query else url
    request = Request(request_url, headers={"User-Agent": config.user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=float(config.timeout_s)) as response:
        return json.loads(response.read().decode("utf-8"))


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _properties(feature: dict[str, Any]) -> dict[str, Any]:
    if isinstance(feature.get("properties"), dict):
        return dict(feature["properties"])
    if isinstance(feature.get("attributes"), dict):
        return dict(feature["attributes"])
    if isinstance(feature.get("feature"), dict):
        nested = feature["feature"]
        if isinstance(nested.get("properties"), dict):
            return dict(nested["properties"])
        if isinstance(nested.get("attributes"), dict):
            return dict(nested["attributes"])
    return {}


def _geometry(feature: dict[str, Any]) -> dict[str, Any]:
    if isinstance(feature.get("geometry"), dict):
        return dict(feature["geometry"])
    if isinstance(feature.get("feature"), dict) and isinstance(feature["feature"].get("geometry"), dict):
        return dict(feature["feature"]["geometry"])
    return {}


def _as_xy_array(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float64)
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    return values[:, :2]


def _longest_sequence(sequences: Any) -> np.ndarray:
    records = list(sequences) if sequences is not None else []
    longest = max(records, key=len) if records else []
    return _as_xy_array(longest)


def _centerline_from_ring(ring: Any) -> np.ndarray:
    points = _as_xy_array(ring)
    if len(points) > 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64)

    center = np.mean(points, axis=0)
    centered = points - center
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((0, 2), dtype=np.float64)
    axis = vh[0]
    projections = centered @ axis
    span = float(np.max(projections) - np.min(projections))
    if span <= 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [
            center + np.min(projections) * axis,
            center + np.max(projections) * axis,
        ],
        dtype=np.float64,
    )


def _centerline_from_rings(rings: Any) -> np.ndarray:
    return _centerline_from_ring(_longest_sequence(rings))


def _line_from_geometry(geometry: dict[str, Any]) -> np.ndarray:
    if "paths" in geometry:
        return _longest_sequence(geometry.get("paths"))
    if "rings" in geometry:
        return _centerline_from_rings(geometry.get("rings"))

    geometry_type = str(geometry.get("type", "") or "")
    coords = geometry.get("coordinates")
    if geometry_type == "LineString":
        return _as_xy_array(coords)
    if geometry_type == "MultiLineString":
        return _longest_sequence(coords)
    if geometry_type == "Polygon":
        return _centerline_from_rings(coords)
    if geometry_type == "MultiPolygon":
        rings = [ring for polygon in coords or [] for ring in polygon]
        return _centerline_from_rings(rings)
    return np.empty((0, 2), dtype=np.float64)


def _feature_id(feature: dict[str, Any]) -> str:
    props = _properties(feature)
    return str(props.get("id") or feature.get("featureId") or feature.get("id") or "")


def _canonical_geojson_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    feature_id = _feature_id(feature)
    geometry = _geometry(feature)
    props = _properties(feature)
    if not feature_id or not geometry:
        return None
    canonical_props = dict(props)
    canonical_props.setdefault("id", feature_id)
    if "bbox" in feature:
        canonical_props.setdefault("bbox", feature.get("bbox"))
    if feature.get("layerBodId"):
        canonical_props.setdefault("layer_bod_id", feature.get("layerBodId"))
    if feature.get("layerName"):
        canonical_props.setdefault("layer_name", feature.get("layerName"))
    if feature.get("featureId"):
        canonical_props.setdefault("feature_id", feature.get("featureId"))
    return {
        "type": "Feature",
        "id": feature_id,
        "bbox": feature.get("bbox"),
        "geometry": geometry,
        "properties": canonical_props,
    }


def _subdivide_bounds(bounds_lv95: LV95_BOUNDS) -> tuple[LV95_BOUNDS, ...]:
    min_x, min_y, max_x, max_y = bounds_lv95
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    return (
        (min_x, min_y, mid_x, mid_y),
        (mid_x, min_y, max_x, mid_y),
        (min_x, mid_y, mid_x, max_y),
        (mid_x, mid_y, max_x, max_y),
    )


def _seed_grid_bounds(bounds_lv95: LV95_BOUNDS, tile_side_m: float) -> tuple[LV95_BOUNDS, ...]:
    min_x, min_y, max_x, max_y = bounds_lv95
    step = max(float(tile_side_m), 1.0)
    tiles: list[LV95_BOUNDS] = []
    x = float(min_x)
    while x < float(max_x):
        next_x = min(x + step, float(max_x))
        y = float(min_y)
        while y < float(max_y):
            next_y = min(y + step, float(max_y))
            tiles.append((x, y, next_x, next_y))
            y = next_y
        x = next_x
    return tuple(tiles)


def fetch_lubis_identify_features(
    bounds_lv95: LV95_BOUNDS,
    *,
    config: CorrectionConfig,
    limit: int = DEFAULT_IDENTIFY_LIMIT,
) -> tuple[dict[str, Any], ...]:
    min_x, min_y, max_x, max_y = bounds_lv95
    map_extent = f"{min_x:.3f},{min_y:.3f},{max_x:.3f},{max_y:.3f}"
    payload = _get_json(
        f"{config.geoadmin_mapserver_root.rstrip('/')}/identify",
        params={
            "geometryType": "esriGeometryEnvelope",
            "geometry": map_extent,
            "geometryFormat": "geojson",
            "imageDisplay": "1200,1200,96",
            "layers": f"all:{config.strip_layer_id}",
            "mapExtent": map_extent,
            "returnGeometry": "true",
            "sr": "2056",
            "tolerance": 0,
            "limit": int(limit),
            "lang": "en",
        },
        config=config,
    )
    return tuple(payload.get("results", ()))


def strip_from_feature(feature: dict[str, Any]) -> StripCandidate | None:
    props = _properties(feature)
    geometry_xy = _line_from_geometry(_geometry(feature))
    if geometry_xy.ndim != 2 or geometry_xy.shape[0] < 2 or geometry_xy.shape[1] < 2:
        return None
    strip_id = str(props.get("id") or feature.get("featureId") or feature.get("id") or "")
    if not strip_id:
        return None
    return StripCandidate(
        strip_id=strip_id,
        flight_year=_integer(props.get("bgdi_flugjahr")),
        flight_date=str(props.get("flugdatum") or props.get("toposhop_date") or ""),
        gsd_m=_numeric(props.get("gsd", props.get("resolution")), default=0.0),
        goal=str(props.get("goal") or ""),
        geometry_xy=geometry_xy[:, :2].astype(np.float64, copy=False),
        properties=props,
    )


def parse_lubis_features(payload: dict[str, Any]) -> tuple[StripCandidate, ...]:
    if str(payload.get("type", "")) == "FeatureCollection":
        records = list(payload.get("features", ()))
    else:
        records = list(payload.get("results", ()))
    strips = [strip for record in records if (strip := strip_from_feature(record)) is not None]
    return tuple(strips)


def filter_swissimage_strips(
    strips: tuple[StripCandidate, ...],
    *,
    flight_year: int | None,
    target_gsd_m: float = 0.1,
    gsd_tolerance_m: float = 1e-6,
) -> tuple[StripCandidate, ...]:
    output = []
    for strip in strips:
        if strip.goal.strip().upper() != "SWISSIMAGE":
            continue
        if abs(float(strip.gsd_m) - float(target_gsd_m)) > float(gsd_tolerance_m):
            continue
        if flight_year is not None and int(strip.flight_year) != int(flight_year):
            continue
        output.append(strip)
    return tuple(output)


def default_lubis_catalog_path(config: CorrectionConfig) -> Path:
    """Return the local nationwide LUBIS strip catalog path for one config."""
    return Path(config.local_strip_catalog) if config.local_strip_catalog else Path(config.cache_root) / DEFAULT_LUBIS_CATALOG_RELATIVE_PATH


def load_lubis_catalog(path: str | Path) -> tuple[StripCandidate, ...]:
    """Load cached LUBIS strip metadata from a local GeoJSON catalog."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_lubis_features(payload)


def ensure_lubis_catalog(
    *,
    config: CorrectionConfig,
    root_bounds_lv95: LV95_BOUNDS = SWISS_LV95_QUERY_BOUNDS,
) -> Path:
    """Ensure the small local LUBIS strip catalog exists and return its path."""
    catalog_path = default_lubis_catalog_path(config)
    if catalog_path.exists():
        return catalog_path
    download_lubis_catalog(
        catalog_path.parent,
        config=config,
        root_bounds_lv95=root_bounds_lv95,
    )
    return catalog_path


def _strip_intersects_bounds(strip: StripCandidate, bounds_lv95: LV95_BOUNDS) -> bool:
    if strip.geometry_xy.size == 0:
        return False
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_lv95]
    strip_min_x = float(np.min(strip.geometry_xy[:, 0]))
    strip_max_x = float(np.max(strip.geometry_xy[:, 0]))
    strip_min_y = float(np.min(strip.geometry_xy[:, 1]))
    strip_max_y = float(np.max(strip.geometry_xy[:, 1]))
    return not (
        strip_max_x < min_x
        or strip_min_x > max_x
        or strip_max_y < min_y
        or strip_min_y > max_y
    )


def query_lubis_strips_from_catalog(
    bounds_lv95: LV95_BOUNDS,
    *,
    config: CorrectionConfig,
    flight_year: int | None,
    catalog_path: str | Path | None = None,
) -> tuple[StripCandidate, ...]:
    expanded_bounds = expand_bounds(bounds_lv95, float(config.strip_query_padding_m))
    path = Path(catalog_path) if catalog_path is not None else ensure_lubis_catalog(config=config)
    spatial_matches = tuple(strip for strip in load_lubis_catalog(path) if _strip_intersects_bounds(strip, expanded_bounds))
    return filter_swissimage_strips(
        spatial_matches,
        flight_year=flight_year,
        target_gsd_m=float(config.target_gsd_m),
    )


def query_lubis_strips_live(
    bounds_lv95: LV95_BOUNDS,
    *,
    config: CorrectionConfig,
    flight_year: int | None,
) -> tuple[StripCandidate, ...]:
    expanded_bounds = expand_bounds(bounds_lv95, float(config.strip_query_padding_m))
    return filter_swissimage_strips(
        parse_lubis_features(
            {"results": list(fetch_lubis_identify_features(expanded_bounds, config=config, limit=DEFAULT_IDENTIFY_LIMIT))}
        ),
        flight_year=flight_year,
        target_gsd_m=float(config.target_gsd_m),
    )


def query_lubis_strips(
    bounds_lv95: LV95_BOUNDS,
    *,
    config: CorrectionConfig,
    flight_year: int | None,
) -> tuple[StripCandidate, ...]:
    if bool(config.prefer_local_strip_catalog):
        return query_lubis_strips_from_catalog(
            bounds_lv95,
            config=config,
            flight_year=flight_year,
        )
    return query_lubis_strips_live(
        bounds_lv95,
        config=config,
        flight_year=flight_year,
    )


def download_lubis_catalog(
    output_dir: str | Path,
    *,
    config: CorrectionConfig | None = None,
    root_bounds_lv95: LV95_BOUNDS = SWISS_LV95_QUERY_BOUNDS,
    request_limit: int = DEFAULT_IDENTIFY_LIMIT,
    min_tile_side_m: float = DEFAULT_MIN_TILE_SIDE_M,
    seed_tile_side_m: float = DEFAULT_SEED_TILE_SIDE_M,
) -> dict[str, Any]:
    cfg = config or CorrectionConfig()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    saturation_threshold = min(int(request_limit), DEFAULT_IDENTIFY_LIMIT)
    pending: list[LV95_BOUNDS] = list(reversed(_seed_grid_bounds(root_bounds_lv95, seed_tile_side_m)))
    visited: set[LV95_BOUNDS] = set()
    features_by_id: dict[str, dict[str, Any]] = {}
    tile_log: list[dict[str, Any]] = []
    saturated_tiles: list[dict[str, Any]] = []

    while pending:
        bounds = pending.pop()
        if bounds in visited:
            continue
        visited.add(bounds)
        min_x, min_y, max_x, max_y = bounds
        width_m = max_x - min_x
        height_m = max_y - min_y
        records = fetch_lubis_identify_features(bounds, config=cfg, limit=request_limit)
        hit_limit = len(records) >= int(saturation_threshold)
        can_subdivide = width_m > float(min_tile_side_m) or height_m > float(min_tile_side_m)
        if hit_limit and can_subdivide:
            tile_log.append(
                {
                    "bounds_lv95": [float(min_x), float(min_y), float(max_x), float(max_y)],
                    "width_m": float(width_m),
                    "height_m": float(height_m),
                    "result_count": int(len(records)),
                    "subdivided": True,
                }
            )
            pending.extend(reversed(_subdivide_bounds(bounds)))
            continue

        tile_log.append(
            {
                "bounds_lv95": [float(min_x), float(min_y), float(max_x), float(max_y)],
                "width_m": float(width_m),
                "height_m": float(height_m),
                "result_count": int(len(records)),
                "subdivided": False,
            }
        )
        if hit_limit:
            saturated_tiles.append(tile_log[-1])
        for record in records:
            feature = _canonical_geojson_feature(record)
            if feature is None:
                continue
            features_by_id.setdefault(str(feature["id"]), feature)

    features = sorted(
        features_by_id.values(),
        key=lambda feature: (
            int(feature["properties"].get("bgdi_flugjahr") or 0),
            str(feature["id"]),
        ),
    )
    year_counts = Counter(int(feature["properties"].get("bgdi_flugjahr") or 0) for feature in features)
    goal_counts = Counter(str(feature["properties"].get("goal") or "") for feature in features)
    geojson = {
        "type": "FeatureCollection",
        "name": cfg.strip_layer_id,
        "crs": {"type": "name", "properties": {"name": "EPSG:2056"}},
        "features": features,
    }
    summary = {
        "layer_id": cfg.strip_layer_id,
        "source_url": f"{cfg.geoadmin_mapserver_root.rstrip('/')}/identify",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_bounds_lv95": [float(value) for value in root_bounds_lv95],
        "request_limit": int(request_limit),
        "saturation_threshold": int(saturation_threshold),
        "min_tile_side_m": float(min_tile_side_m),
        "seed_tile_side_m": float(seed_tile_side_m),
        "feature_count": int(len(features)),
        "tiles_queried": int(len(tile_log)),
        "saturated_tile_count": int(len(saturated_tiles)),
        "year_counts": {str(year): int(count) for year, count in sorted(year_counts.items())},
        "goal_counts": {goal: int(count) for goal, count in sorted(goal_counts.items()) if goal},
        "files": {
            "geojson": "lubis_strips.geojson",
            "index": "lubis_strips_index.json",
            "fetch_report": "lubis_fetch_report.json",
        },
    }
    fetch_report = {
        "summary": summary,
        "tile_log": tile_log,
        "saturated_tiles": saturated_tiles,
    }
    (target_dir / "lubis_strips.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target_dir / "lubis_strips_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target_dir / "lubis_fetch_report.json").write_text(
        json.dumps(fetch_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def strips_to_metadata(strips: tuple[StripCandidate, ...]) -> list[dict[str, Any]]:
    return [
        {
            "id": strip.strip_id,
            "flight_year": int(strip.flight_year),
            "flight_date": strip.flight_date,
            "gsd_m": float(strip.gsd_m),
            "goal": strip.goal,
            "point_count": int(len(strip.geometry_xy)),
        }
        for strip in strips
    ]
