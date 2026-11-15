"""Cambridge source contracts and the shared multi-dataset picker API."""

import json

import geopandas as gpd
import laspy
import numpy as np
import pytest
from building_data.api import mount_acquisition
from building_data.cambridge import CambridgeProvider
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.store import AcquisitionStore
from fastapi import FastAPI
from fastapi.testclient import TestClient
from osgeo import gdal, osr
from pyproj import Transformer
from shapely.geometry import box, mapping


@pytest.fixture
def provider(tmp_path):
    root = tmp_path / "source"
    (root / "points").mkdir(parents=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "revision": "fixture",
                "vertical_crs": "source heights",
                "coverage": mapping(box(541000, 255000, 542000, 256000)),
                "lidar_bounds": [541000, 255000, 542000, 256000],
                "source_sha256": "fixture-source",
                "imagery_dates": {"image": "06/06/2023"},
            }
        )
    )
    frame = gpd.GeoDataFrame(
        {"toid": ["osgb101", "osgb102"], "building_id": [101, 102]},
        geometry=[
            box(541100, 255100, 541110, 255110),
            box(541130, 255100, 541140, 255110),
        ],
        crs=27700,
    )
    frame.to_file(root / "footprints.gpkg", layer="buildings")
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = [0.01, 0.01, 0.01]
    cloud = laspy.LasData(header)
    cloud.x = [541101] * 7
    cloud.y = [255101] * 7
    cloud.z = [12.31, 13.42, 14.53, 15.64, 16.75, 17.86, 18.97]
    cloud.classification = [1, 2, 3, 4, 5, 6, 7]
    cloud.write(root / "points/541_255.laz")
    image = gdal.GetDriverByName("GTiff").Create(
        str(root / "imagery.vrt"),
        4000,
        4000,
        3,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW"],
    )
    image.SetGeoTransform((541000, 0.25, 0, 256000, 0, -0.25))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(27700)
    image.SetProjection(srs.ExportToWkt())
    for i in range(1, 4):
        image.GetRasterBand(i).Fill(100)
    image = None
    return CambridgeProvider(tmp_path / "app/cache", root)


def test_points_retain_all_vegetation_and_exact_source_heights(provider):
    frame, sources = provider.points_for_bounds((541100, 255100, 541110, 255110), 2023)
    assert frame.classification.tolist() == [2, 3, 4, 5, 6]
    np.testing.assert_allclose(frame.z, [13.42, 14.53, 15.64, 16.75, 17.86], atol=1e-12)
    assert sources[0]["crs"] == "EPSG:27700"
    assert sources[0]["revision"] == provider.revision
    with pytest.raises(ValueError, match="revision"):
        provider.points_for_bounds(
            (541100, 255100, 541110, 255110), pinned_sources=({"revision": "other"},)
        )


def test_complete_context_coverage_and_source_year_are_required(provider):
    with pytest.raises(ValueError, match="beyond TL4055"):
        provider.lidar((540999, 255100, 541110, 255110))
    with pytest.raises(ValueError, match="2023 survey"):
        provider.lidar((541100, 255100, 541110, 255110), 2024)


def test_orthophoto_is_a_georeferenced_crop_without_warping(provider):
    from types import SimpleNamespace

    crop = provider.orthophoto(
        SimpleNamespace(bounds_xy=(541100, 255100, 541110, 255110)), CorrectionConfig()
    )
    assert crop.rgb_corrected.shape == (104, 104, 3)
    assert np.all(crop.rgb_corrected == 100)
    assert crop.extent_lv95 == (541092, 255092, 541118, 255118)
    assert crop.transform == (541092, 0.25, 0, 255118, 0, -0.25)
    assert crop.metadata["crs"] == "EPSG:27700"
    assert not crop.occlusion_mask.any()
    assert not crop.displacement.displacement_m.any()
    with pytest.raises(ValueError, match="full roof image crop"):
        provider.orthophoto(
            SimpleNamespace(bounds_xy=(541001, 255100, 541010, 255110)),
            CorrectionConfig(),
        )


def test_picker_configuration_and_footprints_are_provider_specific(provider, tmp_path):
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    app = FastAPI()
    mount_acquisition(app, store)
    client = TestClient(app)
    dataset = client.get("/api/datasets").json()[0]
    assert dataset["id"] == "cambridge" and dataset["search"] is False
    assert dataset["coverage"]["type"] == "Polygon"
    t = Transformer.from_crs(27700, 4326, always_xy=True)
    x0, y0 = t.transform(541090, 255090)
    x1, y1 = t.transform(541150, 255120)
    r = client.get(
        "/api/datasets/cambridge/footprints", params={"bbox": f"{x0},{y0},{x1},{y1}"}
    )
    assert r.status_code == 200
    assert {f["properties"]["toid"] for f in r.json()["features"]} == {
        "osgb101",
        "osgb102",
    }
    assert (
        client.get("/api/datasets/cambridge/footprints?bbox=nan,0,1,1").status_code
        == 422
    )
    assert (
        client.post(
            "/api/acquisition",
            json={"provider": "missing", "longitude": 0, "latitude": 52},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/search", params={"provider": "cambridge", "q": "Cambridge"}
        ).json()
        == []
    )


def test_selected_footprint_becomes_real_surface_layers_and_keeps_its_provider(
    provider, tmp_path, monkeypatch
):
    from building_data import city3d
    from building_data.geometry import load_vector_house

    requested = []

    def scaffold(_provider, row):
        requested.append(row.toid)
        ring = [(x, y, 20) for x, y in row.geometry.exterior.coords]
        return {"Roof": [ring], "Wall": [], "Floor": []}

    monkeypatch.setattr(city3d, "scaffold_surfaces", scaffold)
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    # A different provider being the default must not change the selected source.
    from building_data.swiss import SwissProvider

    store.provider = SwissProvider(tmp_path / "swiss")
    store.providers[store.provider.name] = store.provider
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(
        541105, 255105
    )
    result = store.acquire(
        {"provider": "cambridge", "mode": "house", "longitude": lon, "latitude": lat}
    )
    assert requested == ["osgb101"]
    assert not result["failures"] and len(result["houses"]) == 1
    item = store.house(result["houses"][0])
    assert item.provider == "cambridge" and item.crs == "EPSG:27700"
    assert store.provider_for_house(item) is provider
    assert store.provider.name == "switzerland"
    assert load_vector_house(item.surfaces_path, item.building_fid).roof_faces


def test_failed_city3d_roof_is_reported_and_never_substituted(
    provider, tmp_path, monkeypatch
):
    from building_data import city3d

    def fail(*_):
        raise RuntimeError("No plane support")

    monkeypatch.setattr(city3d, "scaffold_surfaces", fail)
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(
        541105, 255105
    )
    result = store.acquire(
        {"provider": "cambridge", "mode": "house", "longitude": lon, "latitude": lat}
    )
    assert result["houses"] == []
    assert result["failures"] == [{"building_fid": 101, "error": "No plane support"}]


def test_local_imagery_tiles_are_cached_and_reject_unbounded_requests(provider):
    import io
    import math

    from PIL import Image

    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(
        541105, 255105
    )
    z = 18
    x = int((lon + 180) / 360 * 2**z)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z)
    data = provider.map_tile(z, x, y)
    image = np.asarray(Image.open(io.BytesIO(data)))
    assert image.shape == (256, 256, 4)
    assert (image[:, :, :3] == 100).any()
    assert provider.map_tile(z, x, y) == data
    with pytest.raises(ValueError, match="Invalid imagery tile"):
        provider.map_tile(30, x, y)


def flat_scaffold(_provider, row):
    ring = [(x, y, 20) for x, y in row.geometry.exterior.coords]
    return {"Roof": [ring], "Wall": [], "Floor": [[(x, y, 0) for x, y, _ in ring]]}


def test_explicit_parts_form_one_complete_target_with_stable_identity(provider, tmp_path, monkeypatch):
    from building_data import city3d
    from building_data.geometry import load_vector_house

    monkeypatch.setattr(city3d, "scaffold_surfaces", flat_scaffold)
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(541105, 255105)
    selection = {"provider": "cambridge", "mode": "house", "longitude": lon,
                 "latitude": lat, "source_ids": ["osgb102", "osgb101"]}
    first = store.acquire(selection)
    second = store.acquire({**selection, "longitude": lon + .001,
                            "source_ids": ["osgb101", "osgb102", "osgb101"]})
    assert not first["failures"] and len(first["houses"]) == 1
    assert first["id"] == second["id"]
    assert first["houses"] == second["houses"]
    item = store.house(first["houses"][0])
    assert item.target["source_ids"] == ["osgb101", "osgb102"]
    assert item.target["source_area_m2"] == 200
    assert item.target["coverage"]["missing_area_m2"] == 0
    house = load_vector_house(item.surfaces_path, item.building_fid)
    assert house.roof_envelope.area == 200
    assert house.bounds_xy == (541100, 255100, 541140, 255110)


def test_source_neighbour_topology_survives_without_reconstructing_neighbours(provider, tmp_path, monkeypatch):
    from building_data import city3d
    from building_data.geometry import build_house_index, load_vector_house

    frame = gpd.read_file(provider.root / "footprints.gpkg", layer="buildings")
    frame.loc[frame.toid.eq("osgb102"), "geometry"] = box(541110, 255100, 541120, 255110)
    frame.to_file(provider.root / "footprints.gpkg", layer="buildings")
    requested = []
    def scaffold(p, row):
        requested.append(row.toid)
        if row.toid == "osgb102":
            raise RuntimeError("Neighbour has no usable roof returns")
        return flat_scaffold(p, row)
    monkeypatch.setattr(city3d, "scaffold_surfaces", scaffold)
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(541105, 255105)
    result = store.acquire({"provider": "cambridge", "mode": "house", "longitude": lon,
                            "latitude": lat, "source_ids": ["osgb101"]})
    item = store.house(result["houses"][0])
    assert requested == ["osgb101"]
    house = load_vector_house(item.surfaces_path, item.building_fid)
    assert not house.is_independent
    assert house.touching_building_fids == (102,)
    index = build_house_index(item.surfaces_path)
    assert not bool(index.iloc[0].is_independent)
    assert index.iloc[0].touching_building_fids == "102"


def test_failed_selected_part_rejects_entire_target(provider, tmp_path, monkeypatch):
    from building_data import city3d

    def scaffold(p, row):
        if row.toid == "osgb102":
            raise ValueError("No building returns for osgb102")
        return flat_scaffold(p, row)
    monkeypatch.setattr(city3d, "scaffold_surfaces", scaffold)
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(541105, 255105)
    result = store.acquire({"provider": "cambridge", "mode": "house", "longitude": lon,
                            "latitude": lat, "source_ids": ["osgb101", "osgb102"]})
    assert result["houses"] == [] and result["targets"] == []
    assert result["failures"][0]["source_ids"] == ["osgb101", "osgb102"]
    assert "osgb102" in result["failures"][0]["error"]


def test_target_preview_resolves_source_geometry_and_rejects_unknown_ids(provider, tmp_path):
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    app = FastAPI()
    mount_acquisition(app, store)
    client = TestClient(app)
    url = "/api/datasets/cambridge/target-preview"
    response = client.post(url, json={"source_ids": ["osgb101", "osgb102"]})
    assert response.status_code == 200
    preview = response.json()
    assert preview["area_m2"] == 200
    assert preview["source_ids"] == ["osgb101", "osgb102"]
    assert preview["geometry"]["type"] == "MultiPolygon"
    assert preview["unsupported_source_ids"] == []
    for ids in [[], ["osgb999"], ["osgb101' OR 1=1 --"]]:
        assert client.post(url, json={"source_ids": ids}).status_code == 422


def test_boundary_drift_is_retained_as_measured_provenance(provider, tmp_path, monkeypatch):
    from building_data import city3d

    def enlarged(p, row):
        from types import SimpleNamespace

        return flat_scaffold(p, SimpleNamespace(geometry=row.geometry.buffer(1)))
    monkeypatch.setattr(city3d, "scaffold_surfaces", enlarged)
    store = AcquisitionStore(tmp_path / "app", provider=provider)
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(541105, 255105)
    result = store.acquire({"provider": "cambridge", "longitude": lon, "latitude": lat})
    target = store.house(result["houses"][0]).target
    assert target["source_area_m2"] == 100
    assert target["coverage"]["extra_area_m2"] > 40
    assert target["coverage"]["intersection_over_union"] < 1


def test_topology_ids_remain_exact_above_float_integer_precision():
    from building_data.geometry import (
        _coerce_building_fid,
        _parse_touching_building_fids,
    )

    assert _parse_touching_building_fids("123456789012345678") == (123456789012345678,)
    assert _coerce_building_fid(123456789012345678) == 123456789012345678
    assert _coerce_building_fid("123456789012345678") == 123456789012345678
