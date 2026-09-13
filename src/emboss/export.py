"""Emboss-native result export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import tifffile
from shapely.geometry import mapping
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .models import HouseResult
from .models import SurfaceFace
from .models import SuperstructureSolid
from .models import VectorHouse
from .models import ReturnSupportModel
from building_data.raster import rasterize_polygon
from building_data.serialization import json_ready


SCHEMA_VERSION = "emboss-result-v5"
FIT_DIAGNOSTIC_FIELDS = (
    "fit_rmse_m",
    "height_m",
    "footprint_area_m2",
    "top_face_count",
    "sloped_top_face_count",
    "support_return_count",
    "support_weight_m2",
    "strong_lidar_return_count",
    "strong_lidar_support_weight_m2",
    "roof_contradiction_weight_m2",
    "elevated_support_weight_m2",
    "observed_support_weight_m2",
    "elevated_support_fraction",
    "image_overlap_area_m2",
    "image_overlap_footprint_fraction",
    "image_overlap_component_fraction",
)


def _feature(geometry: Any, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": json_ready(properties),
    }


def _feature_collection(features: list[dict[str, Any]], *, crs: str = "EPSG:2056") -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": "emboss",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }


def _surface_feature(face: SurfaceFace, *, kind: str) -> dict[str, Any]:
    coords = [
        [float(x), float(y), float(z)]
        for x, y, z in face.points_xyz
    ]
    if len(coords) >= 3 and coords[0] != coords[-1]:
        coords.append(coords[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": json_ready(
            {
                "kind": kind,
                "face_id": face.face_id,
                "building_fid": face.building_fid,
                "surface_kind": face.surface_kind,
                "area_m2": face.area_m2,
                "z_min": face.z_min,
                "z_max": face.z_max,
            }
        ),
    }


def _write_geojson(path: Path, features: list[dict[str, Any]], *, crs: str = "EPSG:2056") -> None:
    path.write_text(
        json.dumps(_feature_collection(features, crs=crs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_vector_house_geojson(path: Path, house: VectorHouse, *, crs: str = "EPSG:2056") -> None:
    features = []
    features.append(
        _feature(
            house.roof_envelope,
            {
                "kind": "roof_envelope",
                "building_fid": house.building_fid,
                "is_independent": house.is_independent,
                "touching_building_fids": house.touching_building_fids,
            },
        )
    )
    for face in house.roof_faces:
        features.append(
            _feature(
                face.polygon_xy,
                {
                    "kind": "roof_face",
                    "face_id": face.face_id,
                    "building_fid": face.building_fid,
                    "plane_coeffs": face.plane_coeffs,
                    "vertex_rmse_m": face.vertex_rmse_m,
                    "area_m2": face.area_m2,
                },
            )
        )
    for segment in house.roof_segments:
        features.append(
            _feature(
                segment.polygon_xy,
                {
                    "kind": "roof_segment",
                    "segment_id": segment.segment_id,
                    "face_ids": segment.face_ids,
                    "plane_coeffs": segment.plane_coeffs,
                    "normal": segment.normal,
                    "area_m2": segment.area_m2,
                    "is_base": segment.is_base,
                },
            )
        )
    for face in house.wall_faces:
        features.append(_surface_feature(face, kind="wall_face"))
    for face in house.floor_faces:
        features.append(_surface_feature(face, kind="floor_face"))
    _write_geojson(path, features, crs=crs)


def write_superstructures_geojson(
    path: Path,
    solids: tuple[SuperstructureSolid, ...],
    *,
    crs: str = "EPSG:2056",
) -> None:
    features = []
    for solid in solids:
        geometry = _solid_visible_geometry(solid)
        features.append(
            _feature(
                geometry,
                {
                    "solid_id": solid.solid_id,
                    "top_plane": solid.top_plane,
                    "top_faces": [
                        {
                            "face_id": face.face_id,
                            "footprint_xy": face.footprint_xy,
                            "plane": face.plane,
                        }
                        for face in solid.top_faces
                    ],
                    "host_segment_ids": solid.host_segment_ids,
                    "point_count": solid.point_count,
                    "height_offset_m": solid.height_offset_m,
                    "fit_terms": solid.fit_terms,
                    "source": solid.source,
                    "class_id": solid.class_id,
                    "class_label": solid.class_label,
                    "height_model": solid.height_model,
                },
            )
        )
    _write_geojson(path, features, crs=crs)


def _solid_visible_geometry(solid: SuperstructureSolid):
    top_parts = [
        Polygon(face.footprint_xy).buffer(0)
        for face in solid.top_faces
        if len(face.footprint_xy) >= 3
    ]
    top_parts = [part for part in top_parts if not part.is_empty and float(part.area) > 0.0]
    if top_parts:
        return unary_union(top_parts).buffer(0)
    return Polygon(solid.footprint_xy).buffer(0)


def _solid_top_z(solid: SuperstructureSolid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``SuperstructureSolid.z_top_at``."""

    if not solid.top_faces:
        a, b, c = solid.top_plane
        return a * x + b * y + c
    planes = np.asarray([face.plane for face in solid.top_faces], dtype=np.float64)
    per_face_z = x[:, None] * planes[None, :, 0] + y[:, None] * planes[None, :, 1] + planes[None, :, 2]
    polygons = [Polygon(face.footprint_xy) for face in solid.top_faces]
    contains = np.column_stack(
        [
            np.asarray(shapely.intersects_xy(polygon.buffer(1e-8), x, y), dtype=bool)
            for polygon in polygons
        ]
    )
    top_z = np.max(np.where(contains, per_face_z, -np.inf), axis=1)
    missing = ~contains.any(axis=1)
    if np.any(missing):
        points = shapely.points(x[missing], y[missing])
        distances = np.column_stack(
            [np.asarray(shapely.distance(points, polygon), dtype=np.float64) for polygon in polygons]
        )
        nearest = np.argmin(distances, axis=1)
        top_z[missing] = per_face_z[missing, nearest]
    return top_z


def _height_raster(
    solids: tuple[SuperstructureSolid, ...],
    house: VectorHouse,
    *,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((height, width), dtype=np.uint8)
    residual = np.zeros((height, width), dtype=np.float32)
    min_x, max_x, min_y, max_y = extent
    segment_candidates = list(house.base_segments or house.roof_segments)
    segment_planes = np.asarray([segment.plane_coeffs for segment in segment_candidates], dtype=np.float64)
    for solid in solids:
        polygon = _solid_visible_geometry(solid)
        solid_mask = rasterize_polygon(polygon, extent=extent, width=width, height=height)
        rows, cols = np.nonzero(solid_mask)
        mask[solid_mask] = 1
        if len(rows) == 0 or not segment_candidates:
            continue
        x = min_x + (cols.astype(np.float64) + 0.5) / float(width) * (max_x - min_x)
        y = max_y - (rows.astype(np.float64) + 0.5) / float(height) * (max_y - min_y)
        points = shapely.points(x, y)
        distances = np.column_stack(
            [
                np.asarray(shapely.distance(points, segment.polygon_xy), dtype=np.float64)
                for segment in segment_candidates
            ]
        )
        choice = np.argmin(np.nan_to_num(distances, nan=np.inf), axis=1)
        roof_z = (
            segment_planes[choice, 0] * x
            + segment_planes[choice, 1] * y
            + segment_planes[choice, 2]
        )
        top_z = _solid_top_z(solid, x, y)
        residual[rows, cols] = np.maximum(0.0, top_z - roof_z).astype(np.float32)
    return mask, residual


def export_house_result(
    *,
    output_dir: Path,
    workspace_label: str,
    house: VectorHouse,
    solids: tuple[SuperstructureSolid, ...],
    support_model: ReturnSupportModel,
    rgb: np.ndarray,
    vector_roof_mask: np.ndarray,
    extent_lv95: tuple[float, float, float, float],
    image_superstructure_mask: np.ndarray,
    image_superstructure_class_map: np.ndarray,
    image_segmentation_source: str,
    crs: str = "EPSG:2056",
) -> HouseResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    rasters = output_dir / "rasters"
    diagnostics = output_dir / "diagnostics"
    rasters.mkdir(exist_ok=True)
    diagnostics.mkdir(exist_ok=True)

    tifffile.imwrite(rasters / "orthophoto.tif", np.asarray(rgb, dtype=np.uint8))
    tifffile.imwrite(rasters / "image_superstructure_mask.tif", np.asarray(image_superstructure_mask, dtype=np.uint8))
    tifffile.imwrite(
        rasters / "image_superstructure_class_map.tif",
        np.asarray(image_superstructure_class_map, dtype=np.uint8),
    )
    tifffile.imwrite(rasters / "vector_roof_mask.tif", vector_roof_mask.astype(np.uint8))
    super_mask, height_raster = _height_raster(
        solids,
        house,
        extent=extent_lv95,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
    )
    tifffile.imwrite(rasters / "superstructure_mask.tif", super_mask)
    tifffile.imwrite(rasters / "height_above_roof.tif", height_raster.astype(np.float32))

    write_vector_house_geojson(output_dir / "vector_house.geojson", house, crs=crs)
    write_superstructures_geojson(output_dir / "superstructures.geojson", solids, crs=crs)

    fit_diagnostics_path = diagnostics / "fit_diagnostics.csv"
    with fit_diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["solid_id", *FIT_DIAGNOSTIC_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for solid in solids:
            row = {"solid_id": solid.solid_id}
            row.update(
                {
                    key: float(solid.fit_terms.get(key, 0.0))
                    for key in fieldnames
                    if key != "solid_id"
                }
            )
            writer.writerow(row)

    artifacts = {
        "result": "result.json",
        "vector_house": "vector_house.geojson",
        "superstructures": "superstructures.geojson",
        "orthophoto": "rasters/orthophoto.tif",
        "image_superstructure_mask": "rasters/image_superstructure_mask.tif",
        "image_superstructure_class_map": "rasters/image_superstructure_class_map.tif",
        "vector_roof_mask": "rasters/vector_roof_mask.tif",
        "superstructure_mask": "rasters/superstructure_mask.tif",
        "height_above_roof": "rasters/height_above_roof.tif",
        "fit_diagnostics": "diagnostics/fit_diagnostics.csv",
    }
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_workspace": workspace_label,
        "building_fid": int(house.building_fid),
        "image_segmentation_provider": "roof_superstructures",
        "image_segmentation_source": image_segmentation_source,
        "independent_house": {
            "is_independent": bool(house.is_independent),
            "touching_building_fids": list(house.touching_building_fids),
        },
        "image_segmentation": {
            "provider": "roof_superstructures",
            "source": image_segmentation_source,
        },
        "return_support": {
            "model_type": "empirical_returns",
            "roof_band_m": support_model.roof_band_m,
            "super_threshold_m": support_model.super_threshold_m,
            "diagnostics": support_model.diagnostics,
        },
        "solid_count": int(len(solids)),
        "artifacts": artifacts,
    }
    (output_dir / "result.json").write_text(
        json.dumps(json_ready(result_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HouseResult(
        building_fid=house.building_fid,
        output_dir=output_dir,
        vector_house=house,
        solids=solids,
        artifacts=artifacts,
    )
