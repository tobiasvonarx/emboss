"""Building eligibility across shared boundaries, roof forms and object labels.

Synthetic GPKG records use the real geometry reader, acquisition selector,
LiDAR preparation, fitting and export. Only remote source acquisition is replaced.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from building_data.geometry import (
    build_house_index,
    load_vector_house,
    load_vector_house_scaffold,
)
from building_data.raster import rasterize_polygon
from building_data.store import TO_WGS84, AcquisitionStore
from building_data.swiss import SwissProvider
from osgeo import ogr, osr
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import transform

from emboss.lidar import prepare_lidar_observations_from_points
from emboss.mesh import read_ascii_ply_mesh, write_prediction_mesh
from emboss.pipeline import run_emboss_house

# The first two buildings share a complete wall edge. Labels deliberately include
# nonresidential structures; the method must not interpret them as eligibility.
BUILDINGS = {
    1: ("Reihenhaus", (2570100, 1181100, 2570110, 1181110), 0.25),
    2: ("Reihenhaus", (2570110, 1181100, 2570120, 1181110), 0.25),
    3: ("Industriehalle", (2570200, 1181100, 2570300, 1181180), 0.0),
    4: ("Schulhaus", (2570400, 1181100, 2570430, 1181120), 0.20),
}


@pytest.fixture
def surfaces(tmp_path):
    path = tmp_path / "source.gpkg"
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(2056)
    layer = dataset.CreateLayer("Roof", crs, ogr.wkbPolygon25D)
    layer.CreateField(ogr.FieldDefn("OBJEKTART", ogr.OFTString))
    for fid, (kind, bounds, slope) in BUILDINGS.items():
        polygon = Polygon(
            [
                (x, y, 620 + slope * (x - bounds[0]))
                for x, y in box(*bounds).exterior.coords
            ]
        )
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetFID(fid)
        feature.SetField("OBJEKTART", kind)
        feature.SetGeometry(ogr.CreateGeometryFromWkt(polygon.wkt))
        assert layer.CreateFeature(feature) == 0
    feature = layer = dataset = None
    return path


class LocalProvider(SwissProvider):
    def __init__(self, root, surfaces):
        super().__init__(root / "cache")
        self.surfaces = surfaces
        self.lidar_requests = []

    def buildings(self, bounds, output, progress=print):
        shutil.copy2(self.surfaces, output)
        return output

    def lidar(self, bounds, reference_year=None, progress=print, *, pinned_sources=()):
        self.lidar_requests.append(tuple(bounds))
        path = self.cache / "local-fixture.las"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        sources = (
            {
                "path": str(path),
                "tile": "swisssurface3d_2024_2570-1181",
                "year": 2024,
                "bounds": [2570000, 1181000, 2571000, 1182000],
            },
        )
        return (path,), sources


def test_index_retains_attached_flat_large_and_nonresidential_roofs(surfaces):
    index = build_house_index(surfaces).set_index("building_fid")
    assert set(index.index) == set(BUILDINGS)
    for fid, neighbor in [(1, 2), (2, 1)]:
        assert not index.loc[fid, "is_independent"]
        assert index.loc[fid, "touching_building_fids"] == str(neighbor)
        assert index.loc[fid, "roof_type"] == "pitched"
    assert index.loc[3, "roof_type"] == "flat"
    assert index.loc[3, "footprint_area_m2"] == 8000
    assert index.loc[3, "object_type"] == "Industriehalle"
    assert index.loc[4, "object_type"] == "Schulhaus"


def test_area_acquisition_keeps_every_roof_without_use_or_topology_filter(
    surfaces, tmp_path
):
    root = tmp_path / "area-store"
    store = AcquisitionStore(root, provider=LocalProvider(root, surfaces))
    area = transform(TO_WGS84.transform, box(2570099, 1181099, 2570431, 1181181))
    result = store.acquire({"mode": "area", "geometry": mapping(area)})
    assert not result["failures"]
    assert {
        store.house(identifier).building_fid for identifier in result["houses"]
    } == set(BUILDINGS)
    for identifier in result["houses"]:
        house = store.house(identifier)
        assert house.bounds_xy == BUILDINGS[house.building_fid][1]


@pytest.mark.parametrize("fid", BUILDINGS)
def test_point_acquisition_can_select_each_building_category(surfaces, tmp_path, fid):
    root = tmp_path / "point-store"
    store = AcquisitionStore(root, provider=LocalProvider(root, surfaces))
    _, (x0, y0, x1, y1), _ = BUILDINGS[fid]
    longitude, latitude = TO_WGS84.transform((x0 + x1) / 2, (y0 + y1) / 2)
    result = store.acquire(
        {"mode": "house", "longitude": longitude, "latitude": latitude}
    )
    assert not result["failures"]
    assert len(result["houses"]) == 1
    assert store.house(result["houses"][0]).building_fid == fid


@pytest.mark.parametrize("fid", BUILDINGS)
def test_real_fitting_and_export_process_all_building_categories(
    surfaces, tmp_path, fid
):
    index = build_house_index(surfaces)
    house = load_vector_house(surfaces, fid, index=index)
    kind, (x0, y0, x1, y1), slope = BUILDINGS[fid]
    assert house.object_type == kind
    assert house.is_independent == (fid not in (1, 2))
    # Well-distributed roof returns plus a compact raised structure. This exercises
    # canonical calibration and actual fitting, rather than a mocked fit result.
    samples = [
        (x, y, 620 + slope * (x - x0))
        for x in np.linspace(x0 + 0.5, x1 - 0.5, 7)
        for y in np.linspace(y0 + 0.5, y1 - 0.5, 7)
    ]
    samples.extend(
        (x, y, 621 + slope * (x - x0))
        for x in np.linspace(x0 + 2, x0 + 3, 3)
        for y in np.linspace(y0 + 2, y0 + 3, 3)
    )
    points = pd.DataFrame(samples, columns=["x", "y", "z"])
    points["classification"] = 6
    observations = prepare_lidar_observations_from_points(points, house)
    assert len(observations) == len(points)
    labels = np.full((32, 32), 5, dtype=np.uint8)
    output = tmp_path / f"reconstructed-{fid}"
    result = run_emboss_house(
        output_dir=output,
        workspace_label="building-coverage",
        house=house,
        observations=observations,
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        image_superstructure_mask=labels != 5,
        image_superstructure_class_map=labels,
        image_segmentation_source="fixture",
        vector_roof_mask=np.ones(labels.shape, dtype=bool),
        extent_lv95=(x0, x1, y0, y1),
    )
    assert result.solids
    assert all(
        house.roof_envelope.covers(Polygon(solid.footprint_xy))
        for solid in result.solids
    )
    manifest = json.loads((output / "result.json").read_text())
    assert manifest["building_fid"] == fid
    assert manifest["independent_house"]["is_independent"] == (fid not in (1, 2))
    assert manifest["solid_count"] == len(result.solids)
    assert all((output / path).is_file() for path in manifest["artifacts"].values())


@pytest.fixture
def multipart_surfaces(tmp_path):
    path = tmp_path / "multipart.gpkg"
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(2056)
    layer = dataset.CreateLayer("Roof", crs, ogr.wkbMultiPolygon25D)
    polygons = [
        Polygon([(x, y, 620) for x, y in box(*bounds).exterior.coords])
        for bounds in [
            (2570500, 1181100, 2570510, 1181110),
            (2570600, 1181100, 2570605, 1181105),
        ]
    ]
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetFID(5)
    feature.SetGeometry(ogr.CreateGeometryFromWkt(MultiPolygon(polygons).wkt))
    assert layer.CreateFeature(feature) == 0
    feature = layer = dataset = None
    return path


@pytest.mark.parametrize("mode", ["house", "area"])
def test_selection_on_smaller_disconnected_component_acquires_complete_building(
    multipart_surfaces, tmp_path, mode
):
    root = tmp_path / "multipart-store"
    provider = LocalProvider(root, multipart_surfaces)
    store = AcquisitionStore(root, provider=provider)
    lon, lat = TO_WGS84.transform(2570602, 1181102)
    selection = {"mode": "house", "longitude": lon, "latitude": lat}
    if mode == "area":
        selection = {
            "mode": "area",
            "geometry": mapping(
                transform(TO_WGS84.transform, box(2570601, 1181101, 2570603, 1181103))
            ),
        }
    acquired = store.acquire(selection)
    assert not acquired["failures"]
    assert len(acquired["houses"]) == 1
    item = store.house(acquired["houses"][0])
    assert item.building_fid == 5
    assert item.bounds_xy == (2570500, 1181100, 2570605, 1181110)
    assert (2570498, 1181098, 2570607, 1181112) in provider.lidar_requests
    index = build_house_index(multipart_surfaces)
    assert index.iloc[0].footprint_area_m2 == 125


def test_reacquisition_repairs_cached_largest_component_bounds(
    multipart_surfaces, tmp_path
):
    root = tmp_path / "repair-store"
    provider = LocalProvider(root, multipart_surfaces)
    store = AcquisitionStore(root, provider=provider)
    lon, lat = TO_WGS84.transform(2570502, 1181102)
    selection = {"mode": "house", "longitude": lon, "latitude": lat}
    house_id = store.acquire(selection)["houses"][0]
    manifest = store.house(house_id).root / "house.json"
    payload = json.loads(manifest.read_text())
    payload["bounds_xy"] = [2570500, 1181100, 2570510, 1181110]
    manifest.write_text(json.dumps(payload))
    previous_requests = len(provider.lidar_requests)
    assert store.acquire(selection)["houses"] == [house_id]
    assert store.house(house_id).bounds_xy == (2570500, 1181100, 2570605, 1181110)
    assert len(provider.lidar_requests) == previous_requests + 1


def test_disconnected_components_survive_scaffold_masks_evidence_and_fitting(
    multipart_surfaces, tmp_path
):
    house = load_vector_house(multipart_surfaces, 5)
    assert house.roof_envelope.geom_type == "MultiPolygon"
    scaffold = tmp_path / "scaffold.geojson"
    write_vector_house_geojson(scaffold, house)
    house = load_vector_house_scaffold(scaffold, 5)
    assert house.roof_envelope.area == 125
    assert len(house.roof_envelope.geoms) == 2
    assert sum(segment.area_m2 for segment in house.roof_segments) == 125
    x0, y0, x1, y1 = house.bounds_xy
    extent = (x0, x1, y0, y1)
    mask = rasterize_polygon(house.roof_envelope, extent=extent, width=210, height=20)
    assert mask[:, :20].any() and mask[:, -10:].any()
    assert not mask[:, 40:180].any()  # Disconnected gap remains empty.
    samples = []
    for part in house.roof_envelope.geoms:
        a, b, c, d = part.bounds
        samples.extend(
            (x, y, 620)
            for x in np.linspace(a + 0.25, c - 0.25, 7)
            for y in np.linspace(b + 0.25, d - 0.25, 7)
        )
        samples.extend(
            (x, y, 621)
            for x in np.linspace(a + 2, a + 3, 3)
            for y in np.linspace(b + 2, b + 3, 3)
        )
    points = pd.DataFrame(samples, columns=["x", "y", "z"])
    points["classification"] = 6
    observations = prepare_lidar_observations_from_points(points, house)
    assert len(observations) == 116
    assert (observations.x_cal > 2570600).sum() == 58
    classes = np.full(mask.shape, 5, dtype=np.uint8)
    result = run_emboss_house(
        output_dir=tmp_path / "multipart-result",
        workspace_label="coverage",
        house=house,
        observations=observations,
        rgb=np.zeros((*mask.shape, 3), dtype=np.uint8),
        image_superstructure_mask=classes != 5,
        image_superstructure_class_map=classes,
        image_segmentation_source="fixture",
        vector_roof_mask=mask,
        extent_lv95=extent,
    )
    for part in house.roof_envelope.geoms:
        assert any(part.covers(Polygon(solid.footprint_xy)) for solid in result.solids)
    mesh_path = tmp_path / "multipart.ply"
    write_prediction_mesh(
        roof_scaffold=scaffold,
        superstructures_path=tmp_path / "multipart-result/superstructures.geojson",
        output_path=mesh_path,
        crs="EPSG:2056",
    )
    vertices = np.asarray(read_ascii_ply_mesh(mesh_path).vertices)
    assert (vertices[:, 0] < 2570511).any() and (vertices[:, 0] >= 2570600).any()


def test_multipart_segment_survives_scaffold_serialization(
    multipart_surfaces, tmp_path
):
    house = load_vector_house(multipart_surfaces, 5)
    segment = replace(
        house.roof_segments[0],
        face_ids=tuple(face.face_id for face in house.roof_faces),
        polygon_xy=house.roof_envelope,
        area_m2=125.0,
    )
    house = replace(house, roof_segments=(segment,))
    path = tmp_path / "multipart-segment.geojson"
    write_vector_house_geojson(path, house)
    restored = load_vector_house_scaffold(path, 5)
    assert len(restored.roof_segments) == 1
    assert restored.roof_segments[0].polygon_xy.equals(house.roof_envelope)
    assert restored.roof_segments[0].area_m2 == 125


def test_client_carries_actual_attached_topology_into_scaffold(
    surfaces, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    root = tmp_path / "topology-store"
    store = AcquisitionStore(root, provider=LocalProvider(root, surfaces))
    lon, lat = TO_WGS84.transform(2570105, 1181105)
    house_id = store.acquire({"mode": "house", "longitude": lon, "latitude": lat})[
        "houses"
    ][0]
    client = api.Client(root)
    client.store = store
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"not loaded by this geometry-stage check")
    client.segmentation = SimpleNamespace(checkpoint_path=checkpoint)

    class CheckedTopology(Exception):
        pass

    def inspect_scaffold(path, house, **kwargs):
        assert not house.is_independent
        assert house.touching_building_fids == (2,)
        write_vector_house_geojson(path, house, **kwargs)
        reloaded = load_vector_house_scaffold(path, 1)
        assert not reloaded.is_independent and reloaded.touching_building_fids == (2,)
        raise CheckedTopology

    monkeypatch.setattr(api, "write_vector_house_geojson", inspect_scaffold)
    with pytest.raises(CheckedTopology):
        client.reconstruct(house_id)


def test_reconstruct_repairs_legacy_multipart_coverage_before_inference(
    multipart_surfaces, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    root = tmp_path / "automatic-repair"
    provider = LocalProvider(root, multipart_surfaces)
    store = AcquisitionStore(root, provider=provider)
    lon, lat = TO_WGS84.transform(2570502, 1181102)
    house_id = store.acquire({"mode": "house", "longitude": lon, "latitude": lat})[
        "houses"
    ][0]
    manifest = store.house(house_id).root / "house.json"
    payload = json.loads(manifest.read_text())
    payload["bounds_xy"] = [2570500, 1181100, 2570510, 1181110]
    manifest.write_text(json.dumps(payload))
    previous_requests = len(provider.lidar_requests)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"geometry-stage fixture")
    client = api.Client(root)
    client.store = store
    client.segmentation = SimpleNamespace(checkpoint_path=checkpoint)

    class BeforeInference(Exception):
        pass

    def inspect(path, house, **kwargs):
        assert house.roof_envelope.area == 125
        assert store.house(house_id).bounds_xy == (2570500, 1181100, 2570605, 1181110)
        assert len(provider.lidar_requests) == previous_requests + 1
        assert provider.lidar_requests[-1] == (2570498, 1181098, 2570607, 1181112)
        raise BeforeInference

    monkeypatch.setattr(api, "write_vector_house_geojson", inspect)
    with pytest.raises(BeforeInference):
        client.reconstruct(house_id)


from emboss import api
from emboss.export import write_vector_house_geojson
