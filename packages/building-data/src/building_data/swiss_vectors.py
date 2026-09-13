"""Original Swiss surface extraction, preserving feature geometry and IDs."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from osgeo import ogr
ogr.UseExceptions()
OGR_IMPORT_ERROR = None

def _point_xyz(point: tuple[float, ...]) -> tuple[float, float, float]:
    x = float(point[0])
    y = float(point[1])
    z = float(point[2]) if len(point) > 2 else 0.0
    return (x, y, z)


def _ring_points_xyz(ring: Any) -> list[tuple[float, float, float]]:
    if ring is None:
        return []
    return [_point_xyz(point) for point in ring.GetPoints()]


def _extract_polygon_surfaces(geometry: Any) -> list[Any]:
    if geometry is None:
        return []

    candidate = geometry.Clone()
    try:
        candidate = candidate.GetLinearGeometry()
    except Exception:
        pass

    geom_type = ogr.GT_Flatten(candidate.GetGeometryType())
    geom_name = candidate.GetGeometryName().upper()

    if geom_type == ogr.wkbPolygon:
        return [candidate.Clone()]

    if geom_type == ogr.wkbMultiPolygon or geom_name in {"TRIANGLE", "TIN", "POLYHEDRALSURFACE", "TRIANGULATEDSURFACE"}:
        polygons: list[Any] = []
        for index in range(candidate.GetGeometryCount()):
            polygons.extend(_extract_polygon_surfaces(candidate.GetGeometryRef(index)))
        return polygons

    if geom_type in {
        ogr.wkbGeometryCollection,
        ogr.wkbMultiSurface,
        ogr.wkbMultiCurve,
        ogr.wkbCompoundCurve,
        ogr.wkbCurvePolygon,
    }:
        polygons = []
        for index in range(candidate.GetGeometryCount()):
            polygons.extend(_extract_polygon_surfaces(candidate.GetGeometryRef(index)))
        return polygons

    return []


def _polygon_outer_ring_points(polygon: Any) -> list[tuple[float, float, float]]:
    if polygon is None or polygon.GetGeometryCount() <= 0:
        return []
    return _ring_points_xyz(polygon.GetGeometryRef(0))


def _path_bounds_xyz(path: list[tuple[float, float, float]]) -> tuple[float, float, float, float, float, float]:
    xs = [point[0] for point in path]
    ys = [point[1] for point in path]
    zs = [point[2] for point in path]
    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
        min(zs),
        max(zs),
    )


def _path_normal_z(path: list[tuple[float, float, float]]) -> float:
    normal_x = 0.0
    normal_y = 0.0
    normal_z = 0.0
    point_count = len(path)
    if point_count < 3:
        return 0.0

    for index in range(point_count):
        current = path[index]
        next_point = path[(index + 1) % point_count]
        normal_x += (current[1] - next_point[1]) * (current[2] + next_point[2])
        normal_y += (current[2] - next_point[2]) * (current[0] + next_point[0])
        normal_z += (current[0] - next_point[0]) * (current[1] + next_point[1])

    magnitude = float(np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z))
    if magnitude <= 1e-9:
        return 0.0
    return normal_z / magnitude


def _classify_surface_polygon(
    path: list[tuple[float, float, float]],
    *,
    feature_bounds_xyz: tuple[float, float, float, float, float, float],
) -> str:
    _min_x, _min_y, _max_x, _max_y, min_z, max_z = feature_bounds_xyz
    normal_z = abs(_path_normal_z(path))
    average_z = sum(point[2] for point in path) / len(path)
    z_span = max(float(max_z) - float(min_z), 0.01)
    if normal_z >= 0.3:
        floor_threshold = float(min_z) + min(1.0, z_span * 0.15)
        return "Floor" if average_z <= floor_threshold else "Roof"
    return "Wall"


def _copy_layer_schema(source_layer: Any, output_layer: Any) -> None:
    source_defn = source_layer.GetLayerDefn()
    for field_index in range(source_defn.GetFieldCount()):
        field_defn = source_defn.GetFieldDefn(field_index)
        output_layer.CreateField(field_defn)


def _create_semantic_output_layers(
    *,
    output_ds: Any,
    source_layer: Any,
    layer_names: tuple[str, ...],
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer_name in layer_names:
        output_layer = output_ds.CreateLayer(
            str(layer_name),
            srs=source_layer.GetSpatialRef(),
            geom_type=ogr.wkbUnknown,
        )
        if output_layer is None:
            raise RuntimeError(f"Could not create the `{layer_name}` layer inside the output GeoPackage.")
        _copy_layer_schema(source_layer, output_layer)
        layers[str(layer_name)] = output_layer
    return layers


def _merge_surface_polygons(polygons: list[Any]) -> Any:
    if len(polygons) == 1:
        return polygons[0].Clone()
    geometry = ogr.Geometry(ogr.wkbMultiPolygon25D)
    for polygon in polygons:
        geometry.AddGeometry(polygon.Clone())
    return geometry


def _copy_existing_semantic_layers(
    *,
    source_ds: Any,
    output_ds: Any,
    tile_bounds_xy: tuple[float, float, float, float],
    layer_names: tuple[str, ...],
) -> dict[str, int]:
    min_x, min_y, max_x, max_y = [float(value) for value in tile_bounds_xy]
    copied_by_layer: dict[str, int] = {}
    for layer_name in layer_names:
        source_layer = source_ds.GetLayerByName(str(layer_name))
        if source_layer is None:
            copied_by_layer[str(layer_name)] = 0
            continue

        source_layer.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
        source_layer.ResetReading()
        output_layer = output_ds.CreateLayer(
            str(layer_name),
            srs=source_layer.GetSpatialRef(),
            geom_type=source_layer.GetGeomType(),
        )
        if output_layer is None:
            raise RuntimeError(f"Could not create the `{layer_name}` layer inside the output GeoPackage.")

        _copy_layer_schema(source_layer, output_layer)
        output_defn = output_layer.GetLayerDefn()
        copied = 0
        for feature in source_layer:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue
            new_feature = ogr.Feature(output_defn)
            new_feature.SetFrom(feature)
            new_feature.SetFID(int(feature.GetFID()))
            new_feature.SetGeometry(geometry.Clone())
            if output_layer.CreateFeature(new_feature) != 0:
                raise RuntimeError(f"Failed to copy a {layer_name} feature into the output GeoPackage.")
            new_feature = None
            copied += 1
        copied_by_layer[str(layer_name)] = copied
        source_layer.SetSpatialFilter(None)
    return copied_by_layer


def _synthesize_semantic_layers_from_generic_source(
    *,
    source_ds: Any,
    output_ds: Any,
    tile_bounds_xy: tuple[float, float, float, float],
    layer_names: tuple[str, ...],
) -> dict[str, int]:
    min_x, min_y, max_x, max_y = [float(value) for value in tile_bounds_xy]
    source_layers = [source_ds.GetLayerByIndex(index) for index in range(source_ds.GetLayerCount())]
    source_layers = [layer for layer in source_layers if layer is not None]
    if not source_layers:
        return {str(layer_name): 0 for layer_name in layer_names}

    output_layers = _create_semantic_output_layers(
        output_ds=output_ds,
        source_layer=source_layers[0],
        layer_names=layer_names,
    )
    copied_by_layer = {str(layer_name): 0 for layer_name in layer_names}

    for source_layer in source_layers:
        source_layer.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
        source_layer.ResetReading()
        for feature in source_layer:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue

            polygon_surfaces = _extract_polygon_surfaces(geometry)
            semantic_polygons: dict[str, list[Any]] = {str(layer_name): [] for layer_name in layer_names}
            feature_paths: list[list[tuple[float, float, float]]] = []
            polygon_paths: list[tuple[Any, list[tuple[float, float, float]]]] = []
            for polygon in polygon_surfaces:
                path = _polygon_outer_ring_points(polygon)
                if len(path) < 3:
                    continue
                polygon_paths.append((polygon, path))
                feature_paths.append(path)

            if not feature_paths:
                continue

            feature_bounds_xyz = (
                min(bounds[0] for bounds in (_path_bounds_xyz(path) for path in feature_paths)),
                min(bounds[1] for bounds in (_path_bounds_xyz(path) for path in feature_paths)),
                max(bounds[2] for bounds in (_path_bounds_xyz(path) for path in feature_paths)),
                max(bounds[3] for bounds in (_path_bounds_xyz(path) for path in feature_paths)),
                min(bounds[4] for bounds in (_path_bounds_xyz(path) for path in feature_paths)),
                max(bounds[5] for bounds in (_path_bounds_xyz(path) for path in feature_paths)),
            )
            for polygon, path in polygon_paths:
                semantic = _classify_surface_polygon(path, feature_bounds_xyz=feature_bounds_xyz)
                if semantic in semantic_polygons:
                    semantic_polygons[semantic].append(polygon)

            for semantic, polygons in semantic_polygons.items():
                if not polygons:
                    continue
                output_layer = output_layers[semantic]
                output_feature = ogr.Feature(output_layer.GetLayerDefn())
                output_feature.SetFrom(feature)
                output_feature.SetFID(int(feature.GetFID()))
                output_feature.SetGeometry(_merge_surface_polygons(polygons))
                if output_layer.CreateFeature(output_feature) != 0:
                    raise RuntimeError(f"Failed to synthesize a {semantic} feature into the output GeoPackage.")
                copied_by_layer[semantic] += 1
        source_layer.SetSpatialFilter(None)

    return copied_by_layer


def _clip_surface_layers(
    *,
    source_path: Path,
    destination: Path,
    tile_bounds_xy: tuple[float, float, float, float],
    layer_names: tuple[str, ...] = ("Roof", "Wall", "Floor"),
) -> Path:
    if ogr is None:
        detail = ""
        if OGR_IMPORT_ERROR:
            detail = f" Import failed with: {OGR_IMPORT_ERROR}"
            if OGR_IMPORT_ERROR.__context__:
                detail += f" Context: {OGR_IMPORT_ERROR.__context__}"
        raise RuntimeError(f"GDAL/OGR is required to build the active surface workspace layer.{detail}")

    min_x, min_y, max_x, max_y = [float(value) for value in tile_bounds_xy]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source_ds = ogr.Open(str(source_path))
    if source_ds is None:
        raise RuntimeError(f"Could not open buildings source `{source_path}`.")

    gpkg_driver = ogr.GetDriverByName("GPKG")
    if gpkg_driver is None:
        raise RuntimeError("The GDAL `GPKG` driver is not available.")
    output_ds = gpkg_driver.CreateDataSource(str(destination))
    if output_ds is None:
        raise RuntimeError(f"Could not create `{destination}`.")

    has_semantic_layers = any(source_ds.GetLayerByName(str(layer_name)) is not None for layer_name in layer_names)
    if has_semantic_layers:
        copied_by_layer = _copy_existing_semantic_layers(
            source_ds=source_ds,
            output_ds=output_ds,
            tile_bounds_xy=tile_bounds_xy,
            layer_names=layer_names,
        )
    else:
        copied_by_layer = _synthesize_semantic_layers_from_generic_source(
            source_ds=source_ds,
            output_ds=output_ds,
            tile_bounds_xy=tile_bounds_xy,
            layer_names=layer_names,
        )

    output_ds = None
    source_ds = None

    if copied_by_layer.get("Roof", 0) <= 0 or not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("No roof geometry remained after selecting features for the active tile.")

    validation_ds = ogr.Open(str(destination))
    if validation_ds is None:
        raise RuntimeError(f"`{destination}` was created but could not be reopened as a valid GeoPackage.")
    validation_roof_layer = validation_ds.GetLayerByName("Roof")
    if validation_roof_layer is None:
        raise RuntimeError(f"`{destination}` does not contain a readable `Roof` layer.")
    validation_ds = None
    return destination
