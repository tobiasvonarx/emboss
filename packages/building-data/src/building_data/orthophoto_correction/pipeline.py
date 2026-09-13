"""Public orchestration API for Emboss orthophoto correction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from building_data.orthophoto_correction.geometry import crop_bounds_lv95
from building_data.orthophoto_correction.geometry import height_above_terrain
from building_data.orthophoto_correction.geometry import lv95_geotransform
from building_data.orthophoto_correction.mobile_sam import MODEL_REPOSITORY
from building_data.orthophoto_correction.mobile_sam import MODEL_REVISION
from building_data.orthophoto_correction.mobile_sam import MODEL_SHA256
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.models import CorrectionResult
from building_data.orthophoto_correction.models import LV95_BOUNDS
from building_data.orthophoto_correction.models import RasterAsset
from building_data.orthophoto_correction.models import StripCandidate
from building_data.orthophoto_correction.raster import read_raster_crop
from building_data.orthophoto_correction.selection import RAW_CANDIDATE_ID
from building_data.orthophoto_correction.selection import select_orthophoto_candidate
from building_data.orthophoto_correction.stac import assets_to_metadata
from building_data.orthophoto_correction.stac import resolve_correction_assets
from building_data.orthophoto_correction.stac import source_year_from_id
from building_data.orthophoto_correction.stac import source_years_by_proximity
from building_data.orthophoto_correction.strips import query_lubis_strips
from building_data.orthophoto_correction.strips import strips_to_metadata
from building_data.orthophoto_correction.validation import correction_quality_summary


class CorrectionUnavailable(RuntimeError):
    """Raised when the required public source data cannot support a crop."""


def _as_rgb(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.ndim != 3:
        raise ValueError("SWISSIMAGE crop must be a 2D or 3D array.")
    if values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    if values.shape[2] < 3:
        raise ValueError("SWISSIMAGE crop must have one grayscale band or at least three RGB bands.")
    return np.clip(values[:, :, :3], 0, 255).astype(np.uint8, copy=False)


def _single_band(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"{label} crop must be a single-band raster.")
    return values.astype(np.float32, copy=False)


def _require_assets(asset_map: dict[str, list[RasterAsset]], key: str) -> list[RasterAsset]:
    assets = list(asset_map.get(key, ()))
    if not assets:
        raise CorrectionUnavailable(f"No {key} assets found for correction crop.")
    return assets


def _rgb_assets_by_year(assets: list[RasterAsset]) -> dict[int, list[RasterAsset]]:
    grouped: dict[int, list[RasterAsset]] = {}
    for asset in assets:
        if asset.year is not None:
            grouped.setdefault(int(asset.year), []).append(asset)
    return {
        year: sorted(values, key=lambda asset: asset.item_id)
        for year, values in sorted(grouped.items(), reverse=True)
    }


def _select_primary_source(
    bounds_lv95: LV95_BOUNDS,
    *,
    rgb_assets: list[RasterAsset],
    config: CorrectionConfig,
    lidar_year: int | None,
) -> tuple[int, list[RasterAsset], tuple[StripCandidate, ...]]:
    assets_by_year = _rgb_assets_by_year(rgb_assets)
    if not assets_by_year:
        raise CorrectionUnavailable("No year-tagged SWISSIMAGE assets found for correction crop.")
    checked_years: list[int] = []
    for year in source_years_by_proximity(set(assets_by_year), lidar_year):
        year_assets = assets_by_year[year]
        checked_years.append(year)
        strips = tuple(query_lubis_strips(bounds_lv95, config=config, flight_year=year))
        if strips:
            return year, year_assets, strips
    checked = ", ".join(str(year) for year in checked_years) or "none"
    raise CorrectionUnavailable(
        f"No year-tagged SWISSIMAGE source with matching LUBIS strips was found. Checked years: {checked}."
    )


def _correct_primary_crop(
    bounds_lv95: LV95_BOUNDS,
    *,
    config: CorrectionConfig,
    rgb_assets: list[RasterAsset],
    surface_assets: list[RasterAsset],
    terrain_assets: list[RasterAsset],
    flight_year: int,
    lidar_year: int | None,
    candidate_strips: tuple[StripCandidate, ...],
    roof_envelope: Any,
) -> CorrectionResult:
    rgb_raw, extent_lv95 = read_raster_crop(
        rgb_assets,
        bounds_lv95=bounds_lv95,
        gsd_m=float(config.target_gsd_m),
        resample_alg="bilinear",
    )
    rgb = _as_rgb(rgb_raw)
    surface_raw, _ = read_raster_crop(
        surface_assets,
        bounds_lv95=extent_lv95,
        gsd_m=float(config.target_gsd_m),
        resample_alg="bilinear",
    )
    terrain_raw, _ = read_raster_crop(
        terrain_assets,
        bounds_lv95=extent_lv95,
        gsd_m=float(config.target_gsd_m),
        resample_alg="bilinear",
    )
    surface = _single_band(surface_raw, "swissSURFACE3D")
    terrain = _single_band(terrain_raw, "swissALTI3D")
    if surface.shape != rgb.shape[:2] or terrain.shape != rgb.shape[:2]:
        raise CorrectionUnavailable("Height rasters did not resample to the SWISSIMAGE crop grid.")

    height_m = height_above_terrain(surface, terrain, min_height_m=float(config.min_height_m))
    include_raw = not config.strip_id_override or config.strip_id_override == RAW_CANDIDATE_ID
    strips = candidate_strips if not config.strip_id_override else ()
    if config.strip_id_override and config.strip_id_override != RAW_CANDIDATE_ID:
        strips = tuple(strip for strip in candidate_strips if strip.strip_id == config.strip_id_override)
        if not strips:
            available = ", ".join(strip.strip_id for strip in candidate_strips)
            raise CorrectionUnavailable(
                f"LUBIS strip override `{config.strip_id_override}` is unavailable in {flight_year}. "
                f"Available strips: {available}."
            )

    selected, candidates = select_orthophoto_candidate(
        rgb,
        height_m=height_m,
        bounds_lv95=extent_lv95,
        roof_envelope=roof_envelope,
        strips=strips,
        config=config,
        include_raw=include_raw,
    )
    corrected = selected.rgb.copy()
    corrected[~selected.valid_mask | selected.occlusion_mask] = 0
    transform = lv95_geotransform(extent_lv95, corrected.shape[1], corrected.shape[0])
    metadata: dict[str, Any] = {
        "status": "complete",
        "method": "primary_year_per_candidate_mobile_sam_v1",
        "crs": "EPSG:2056",
        "source_year": flight_year,
        "lidar_source_year": lidar_year,
        "orthophoto_lidar_year_delta": None if lidar_year is None else flight_year - lidar_year,
        "source_assets": {
            "swissimage": assets_to_metadata(rgb_assets),
            "surface": assets_to_metadata(surface_assets),
            "terrain": assets_to_metadata(terrain_assets),
        },
        "strips": strips_to_metadata((selected.strip,)) if selected.strip is not None else [],
        "strip_candidates": strips_to_metadata(candidate_strips),
        "candidate_selection": {
            "mode": (
                "manual_override"
                if config.strip_id_override
                else "mobile_sam_per_candidate_roof_iou"
            ),
            "override_id": config.strip_id_override,
            "selected_id": selected.candidate_id,
            "candidate_scores": [
                {
                    "candidate_id": candidate.candidate_id,
                    "kind": "raw" if candidate.strip is None else "corrected",
                    "roof_iou": (
                        None
                        if candidate.roof_iou is None
                        else float(candidate.roof_iou)
                    ),
                    "mean_strip_distance_m": candidate.mean_strip_distance_m,
                }
                for candidate in sorted(candidates, key=lambda value: value.candidate_id)
            ],
        },
        "config": {
            "flight_height_m": float(config.flight_height_m),
            "target_gsd_m": float(config.target_gsd_m),
            "height_gsd_m": float(config.height_gsd_m),
            "min_height_m": float(config.min_height_m),
            "max_strip_distance_m": float(config.max_strip_distance_m),
            "strip_query_padding_m": float(config.strip_query_padding_m),
            "strip_id_override": config.strip_id_override,
            "mobile_sam": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "sha256": MODEL_SHA256,
                "prompt_padding_m": float(config.mobile_sam_prompt_padding_m),
            },
            "min_crop_side_m": float(config.min_crop_side_m),
        },
        "extent_lv95": [float(value) for value in extent_lv95],
        "transform": list(transform),
        "width": int(corrected.shape[1]),
        "height": int(corrected.shape[0]),
    }
    result = CorrectionResult(
        rgb_corrected=corrected,
        valid_mask=selected.valid_mask,
        occlusion_mask=selected.occlusion_mask,
        corrected_height_mask=selected.displacement.corrected_height_mask,
        displacement=selected.displacement,
        extent_lv95=extent_lv95,
        transform=transform,
        metadata=metadata,
        rgb_raw=rgb,
    )
    metadata["quality"] = correction_quality_summary(result)
    return replace(result, metadata=metadata)


def correct_orthophoto_crop(
    bounds_lv95: LV95_BOUNDS,
    roof_envelope: Any,
    config: CorrectionConfig | None = None,
    *,
    asset_map: dict[str, list[RasterAsset] | tuple[RasterAsset, ...]] | None = None,
    lidar_year: int | None = None,
) -> CorrectionResult:
    """Correct one crop from one primary SWISSIMAGE year."""

    cfg = config or CorrectionConfig()
    resolved_assets = asset_map or resolve_correction_assets(bounds_lv95, config=cfg)
    rgb_assets = _require_assets(resolved_assets, "swissimage")
    surface_assets = _require_assets(resolved_assets, "surface")
    terrain_assets = _require_assets(resolved_assets, "terrain")
    flight_year, primary_assets, candidate_strips = _select_primary_source(
        bounds_lv95,
        rgb_assets=rgb_assets,
        config=cfg,
        lidar_year=lidar_year,
    )
    return _correct_primary_crop(
        bounds_lv95,
        config=cfg,
        rgb_assets=primary_assets,
        surface_assets=surface_assets,
        terrain_assets=terrain_assets,
        flight_year=flight_year,
        lidar_year=lidar_year,
        candidate_strips=candidate_strips,
        roof_envelope=roof_envelope,
    )


def _workspace_correction_assets(workspace: Any | None) -> dict[str, tuple[RasterAsset, ...]]:
    if workspace is None:
        return {}
    assets = getattr(workspace, "correction_raster_assets", None)
    if not assets:
        return {}
    output: dict[str, tuple[RasterAsset, ...]] = {}
    for key in ("swissimage", "surface", "terrain"):
        values = tuple(assets.get(key, ()))
        if values:
            output[key] = values
    return output if all(key in output for key in ("swissimage", "surface", "terrain")) else {}


def correct_house_orthophoto(
    state: dict[str, Any],
    workspace: Any | None = None,
    *,
    padding_m: float,
    config: CorrectionConfig | None = None,
) -> CorrectionResult:
    cfg = config or CorrectionConfig()
    house = state["house"]
    bounds_xy = tuple(float(value) for value in house.bounds_xy)
    crop_bounds = crop_bounds_lv95(
        bounds_xy,  # type: ignore[arg-type]
        padding_m=float(padding_m),
        min_side_m=float(cfg.min_crop_side_m),
    )
    workspace_assets = _workspace_correction_assets(workspace)
    lidar_year = source_year_from_id(str(getattr(workspace, "tile_key", "")))
    if workspace_assets:
        result = correct_orthophoto_crop(
            crop_bounds,
            house.roof_envelope,
            cfg,
            asset_map=workspace_assets,
            lidar_year=lidar_year,
        )
    else:
        result = correct_orthophoto_crop(
            crop_bounds,
            house.roof_envelope,
            cfg,
            lidar_year=lidar_year,
        )
    if workspace is None:
        return result
    metadata = dict(result.metadata)
    metadata["workspace"] = {
        "tile_key": str(getattr(workspace, "tile_key", "")),
        "tile_bounds_lv95": [float(value) for value in getattr(workspace, "tile_bounds_xy", ())],
    }
    return replace(result, metadata=metadata)
