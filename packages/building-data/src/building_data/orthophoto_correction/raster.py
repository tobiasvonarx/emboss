"""GDAL-backed raster crop and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from building_data.orthophoto_correction.geometry import lv95_geotransform
from building_data.orthophoto_correction.models import (
    LV95_BOUNDS,
    CorrectionResult,
    RasterAsset,
)

try:  # pragma: no cover - availability depends on local GDAL installation
    from osgeo import gdal, osr

    gdal.UseExceptions()
    GDAL_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    gdal = None
    osr = None
    GDAL_IMPORT_ERROR = exc


def _require_gdal() -> Any:
    if gdal is None:
        detail = ""
        if GDAL_IMPORT_ERROR:
            detail = f" Import failed with: {GDAL_IMPORT_ERROR}"
            if GDAL_IMPORT_ERROR.__context__:
                detail += f" Context: {GDAL_IMPORT_ERROR.__context__}"
        raise RuntimeError(
            f"GDAL is required for live orthophoto correction raster reads.{detail}"
        )
    return gdal


def _gdal_path(href: str) -> str:
    if href.startswith(("http://", "https://")):
        return f"/vsicurl/{href}"
    return href


def _dataset_bounds(dataset: Any) -> LV95_BOUNDS:
    gt = dataset.GetGeoTransform()
    width = int(dataset.RasterXSize)
    height = int(dataset.RasterYSize)
    min_x = float(gt[0])
    max_y = float(gt[3])
    max_x = min_x + float(gt[1]) * width
    min_y = max_y + float(gt[5]) * height
    return (min_x, min_y, max_x, max_y)


def _array_from_gdal(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3:
        values = np.moveaxis(values, 0, -1)
        if values.shape[2] > 3:
            values = values[:, :, :3]
    return values


def _local_asset_path(asset: RasterAsset) -> Path | None:
    href = str(asset.href)
    if href.startswith(("http://", "https://")):
        return None
    path = Path(href)
    return path if path.exists() else None


def _dataset_epsg(dataset: Any) -> int | None:
    if osr is None:
        return None
    try:
        projection = str(dataset.GetProjection() or "")
    except Exception:  # noqa: BLE001 - optional GDAL projection introspection
        return None
    if not projection:
        return None
    spatial_ref = osr.SpatialReference()
    try:
        spatial_ref.ImportFromWkt(projection)
        spatial_ref.AutoIdentifyEPSG()
    except Exception:  # noqa: BLE001 - optional GDAL projection introspection
        return None
    for key in (None, "PROJCS", "GEOGCS"):
        try:
            authority = spatial_ref.GetAuthorityCode(key)
        except Exception:  # noqa: BLE001 - optional GDAL projection introspection
            authority = None
        if authority:
            try:
                return int(authority)
            except ValueError:
                return None
    return None


def _direct_grid_window_crop(
    assets: list[RasterAsset],
    *,
    bounds_lv95: LV95_BOUNDS,
    gsd_m: float,
) -> tuple[np.ndarray, LV95_BOUNDS] | None:
    if len(assets) != 1:
        return None
    asset = assets[0]
    if int(asset.epsg) != 2056:
        return None
    path = _local_asset_path(asset)
    if path is None:
        return None

    gdal_module = _require_gdal()
    dataset = gdal_module.Open(str(path))
    if dataset is None:
        return None
    epsg = _dataset_epsg(dataset)
    if epsg is not None and epsg != 2056:
        dataset = None
        return None

    gt = dataset.GetGeoTransform()
    origin_x = float(gt[0])
    pixel_width = float(gt[1])
    rotation_x = float(gt[2])
    origin_y = float(gt[3])
    rotation_y = float(gt[4])
    pixel_height = float(gt[5])
    tolerance = max(abs(float(gsd_m)) * 1e-6, 1e-9)
    if (
        abs(rotation_x) > tolerance
        or abs(rotation_y) > tolerance
        or pixel_width <= 0.0
        or pixel_height >= 0.0
        or abs(pixel_width - float(gsd_m)) > tolerance
        or abs(abs(pixel_height) - float(gsd_m)) > tolerance
    ):
        dataset = None
        return None

    min_x, min_y, max_x, max_y = [float(value) for value in bounds_lv95]
    pixel_y = abs(pixel_height)
    eps = 1e-9
    col0 = int(np.floor((min_x - origin_x) / pixel_width + eps))
    col1 = int(np.ceil((max_x - origin_x) / pixel_width - eps))
    row0 = int(np.floor((origin_y - max_y) / pixel_y + eps))
    row1 = int(np.ceil((origin_y - min_y) / pixel_y - eps))
    width = col1 - col0
    height = row1 - row0
    if width <= 0 or height <= 0:
        dataset = None
        return None
    if (
        col0 < 0
        or row0 < 0
        or col1 > int(dataset.RasterXSize)
        or row1 > int(dataset.RasterYSize)
    ):
        dataset = None
        return None

    array = dataset.ReadAsArray(col0, row0, width, height)
    if array is None:
        dataset = None
        raise RuntimeError(
            f"GDAL returned no raster data for aligned local crop `{path}`."
        )
    snapped_bounds = (
        origin_x + col0 * pixel_width,
        origin_y + row1 * pixel_height,
        origin_x + col1 * pixel_width,
        origin_y + row0 * pixel_height,
    )
    dataset = None
    return _array_from_gdal(np.asarray(array)), snapped_bounds


def read_raster_crop(
    assets: list[RasterAsset],
    *,
    bounds_lv95: LV95_BOUNDS,
    gsd_m: float,
    resample_alg: str = "bilinear",
    source_paths: dict[str, str | Path] | None = None,
) -> tuple[np.ndarray, LV95_BOUNDS]:
    if not assets:
        raise ValueError("At least one raster asset is required.")

    direct_crop = _direct_grid_window_crop(
        assets,
        bounds_lv95=bounds_lv95,
        gsd_m=float(gsd_m),
    )
    if direct_crop is not None:
        return direct_crop

    gdal_module = _require_gdal()
    # IO substitutions apply ONLY to BuildVRT. Retain remote asset.href above so
    # downloading raw bytes cannot activate the local direct-grid crop branch.
    paths = [
        str(source_paths[asset.href])
        if source_paths and asset.href in source_paths
        else _gdal_path(asset.href)
        for asset in assets
    ]
    vrt = gdal_module.BuildVRT("", paths)
    if vrt is None:
        raise RuntimeError("Could not build GDAL VRT for correction assets.")
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_lv95]
    dataset = gdal_module.Warp(
        "",
        vrt,
        format="MEM",
        dstSRS="EPSG:2056",
        outputBounds=(min_x, min_y, max_x, max_y),
        xRes=float(gsd_m),
        yRes=float(gsd_m),
        resampleAlg=resample_alg,
        multithread=True,
    )
    if dataset is None:
        raise RuntimeError("Could not crop correction raster assets.")
    array = dataset.ReadAsArray()
    if array is None:
        raise RuntimeError("GDAL returned no raster data for correction crop.")
    return _array_from_gdal(np.asarray(array)), _dataset_bounds(dataset)


def read_geotiff_array(path: Path) -> np.ndarray:
    """Read a local GeoTIFF into the same array layout used by correction results."""
    gdal_module = _require_gdal()
    dataset = gdal_module.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Could not open GeoTIFF `{path}`.")
    array = dataset.ReadAsArray()
    if array is None:
        raise RuntimeError(f"GeoTIFF `{path}` contains no readable array data.")
    values = _array_from_gdal(np.asarray(array))
    dataset = None
    return values


def _gdal_dtype(array: np.ndarray) -> Any:
    gdal_module = _require_gdal()
    dtype = np.asarray(array).dtype
    if dtype == np.uint8 or dtype == bool:
        return gdal_module.GDT_Byte
    if dtype == np.uint16:
        return gdal_module.GDT_UInt16
    if dtype == np.int16:
        return gdal_module.GDT_Int16
    if dtype == np.float64:
        return gdal_module.GDT_Float64
    return gdal_module.GDT_Float32


def write_geotiff(
    path: Path,
    array: np.ndarray,
    *,
    bounds_lv95: LV95_BOUNDS,
    nodata: float | None = None,
) -> Path:
    gdal_module = _require_gdal()
    arr = np.asarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        height, width = arr.shape
        band_count = 1
        write_array = arr[None, :, :]
    elif arr.ndim == 3:
        height, width, band_count = arr.shape
        write_array = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError("GeoTIFF arrays must be 2D or 3D.")

    driver = gdal_module.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        int(width),
        int(height),
        int(band_count),
        _gdal_dtype(arr),
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    dataset.SetGeoTransform(lv95_geotransform(bounds_lv95, int(width), int(height)))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(2056)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    for band_index in range(band_count):
        band = dataset.GetRasterBand(band_index + 1)
        band.WriteArray(write_array[band_index])
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
    dataset.FlushCache()
    dataset = None
    return path


def write_correction_outputs(
    result: CorrectionResult, output_dir: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rgb_raw": output_dir / "rgb_raw.tif",
        "rgb_corrected": output_dir / "rgb_corrected.tif",
        "valid_mask": output_dir / "correction_valid_mask.tif",
        "occlusion_mask": output_dir / "correction_occlusion_mask.tif",
        "corrected_height_mask": output_dir / "correction_height_mask.tif",
        "displacement_m": output_dir / "correction_displacement_m.tif",
        "height_m": output_dir / "correction_height_m.tif",
        "metadata": output_dir / "orthophoto_correction.json",
    }
    if result.rgb_raw is not None:
        write_geotiff(
            paths["rgb_raw"],
            result.rgb_raw.astype(np.uint8),
            bounds_lv95=result.extent_lv95,
            nodata=0,
        )
    write_geotiff(
        paths["rgb_corrected"],
        result.rgb_corrected.astype(np.uint8),
        bounds_lv95=result.extent_lv95,
        nodata=0,
    )
    write_geotiff(
        paths["valid_mask"],
        result.valid_mask.astype(np.uint8),
        bounds_lv95=result.extent_lv95,
        nodata=0,
    )
    write_geotiff(
        paths["occlusion_mask"],
        result.occlusion_mask.astype(np.uint8),
        bounds_lv95=result.extent_lv95,
        nodata=0,
    )
    write_geotiff(
        paths["corrected_height_mask"],
        result.corrected_height_mask.astype(np.uint8),
        bounds_lv95=result.extent_lv95,
        nodata=0,
    )
    write_geotiff(
        paths["displacement_m"],
        result.displacement.displacement_m.astype(np.float32),
        bounds_lv95=result.extent_lv95,
    )
    write_geotiff(
        paths["height_m"],
        result.displacement.height_m.astype(np.float32),
        bounds_lv95=result.extent_lv95,
    )
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        key: str(value)
        for key, value in paths.items()
        if key != "rgb_raw" or Path(value).exists()
    }
