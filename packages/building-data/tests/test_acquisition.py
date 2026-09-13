"""Acquisition contracts independent of downloads, plus real LAS boundary fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import laspy
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, mapping
from shapely.ops import transform
from pyproj import Transformer

from building_data import geometry as geometry_module
from building_data import store as store_module
from building_data.api import mount_acquisition
from building_data.store import AcquisitionStore, TO_WGS84, selection_geometry
from building_data.swiss import SwissProvider, intersecting_tiles


def write_las(path: Path, xyz: list[tuple[float, float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(np.min(np.asarray(xyz), axis=0))
    data = laspy.LasData(header)
    data.x, data.y, data.z = np.asarray(xyz).T
    data.classification = np.full(len(xyz), 6, dtype=np.uint8)
    data.write(path)
    return path


def source(path: Path, cell: tuple[int, int], year: int = 2024) -> dict:
    x, y = cell
    return {
        "tile": f"swisssurface3d_{year}_{x // 1000}-{y // 1000}",
        "year": year,
        "bounds": [x, y, x + 1000, y + 1000],
        "path": str(path),
        "crs": "EPSG:2056",
        "vertical_crs": "EPSG:5728",
        "source": {"asset_href": f"https://example.test/{x}-{y}.las"},
    }


def area_selection(bounds: tuple[float, float, float, float]) -> dict:
    polygon = transform(TO_WGS84.transform, box(*bounds))
    return {"mode": "area", "geometry": mapping(polygon)}


def house_geometry(fid: int, bounds: tuple[float, float, float, float]):
    x0, y0, x1, y1 = bounds
    # VectorHouse uses XYZXYZ, unlike the XYXYZZ extraction helper.
    return SimpleNamespace(
        building_fid=fid,
        roof_envelope=box(*bounds),
        roof_faces=(object(),),
        bounds_xy=bounds,
        bounds_xyz=(x0, y0, 610.0, x1, y1, 625.0),
    )


class FixtureProvider(SwissProvider):
    """Deterministic full-tile provider; fake vector loading is separately injected."""

    def __init__(self, root: Path):
        super().__init__(root / "cache", workers=2)
        self.building_requests = []
        self.lidar_requests = []
        self.point_requests = []
        self.validated = []

    def validate_selection(self, geometry):
        self.validated.append(geometry)
        if not box(5.8, 45.7, 10.7, 47.9).covers(geometry):
            raise ValueError("Fixture provider covers Switzerland")

    def house_id(self, fid):
        return f"fixture-v1-{fid}"

    def terrain(self, bounds, output, *, pinned_sources=()):
        return None

    def buildings(self, bounds, output, progress=print):
        self.building_requests.append(tuple(bounds))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "fixture vectors are loaded by the monkeypatched geometry reader"
        )
        return output

    def lidar(self, bounds, reference_year=None, progress=print, *, pinned_sources=()):
        self.lidar_requests.append(tuple(bounds))
        sources = []
        for x, y in intersecting_tiles(bounds):
            path = self.cache / f"{x}_{y}.las"
            if not path.exists():
                write_las(path, [(x, y, 600), (x + 999.999, y + 999.999, 630)])
            sources.append(
                {**source(path, (x, y), reference_year or 2024), "crs": self.crs}
            )
        return tuple(Path(s["path"]) for s in sources), tuple(sources)

    def points_for_bounds(
        self, bounds, reference_year=None, progress=print, *, pinned_sources=()
    ):
        self.point_requests.append((tuple(bounds), reference_year, pinned_sources))
        x, y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        return pd.DataFrame(
            {"x": [x, x], "y": [y, y], "z": [620.0, 620.0], "classification": [6, 6]}
        ), pinned_sources


def fixture_store(tmp_path, monkeypatch, houses):
    provider = FixtureProvider(tmp_path)
    monkeypatch.setattr(
        store_module,
        "build_house_index",
        lambda _path: pd.DataFrame({"building_fid": list(houses)}),
    )
    monkeypatch.setattr(
        store_module, "load_vector_house", lambda _path, fid, **_kwargs: houses[fid]
    )
    return AcquisitionStore(tmp_path, workers=2, provider=provider), provider


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((0, 0, 1000, 1000), [(0, 0)]),
        ((999, 999, 1001, 1001), [(0, 0), (0, 1000), (1000, 0), (1000, 1000)]),
        ((-1, -1, 1, 1), [(-1000, -1000), (-1000, 0), (0, -1000), (0, 0)]),
        ((0, 1, 3000, 999), [(0, 0), (1000, 0), (2000, 0)]),
    ],
)
def test_tile_enumeration_owns_shared_edges_once(bounds, expected):
    assert intersecting_tiles(bounds) == expected


@pytest.mark.parametrize(
    "bounds", [(0, 0, 0, 1), (1, 0, 0, 1), (0, 2, 1, 1), (0, 0, float("nan"), 1)]
)
def test_tile_enumeration_rejects_invalid_extents(bounds):
    with pytest.raises(ValueError):
        intersecting_tiles(bounds)


def test_area_acquires_empty_tiles_and_complete_boundary_houses(tmp_path, monkeypatch):
    # Only the western tile has buildings, while the selected area crosses three tiles.
    houses = {
        11: house_geometry(11, (2570930, 1181105, 2570940, 1181115)),
        12: house_geometry(12, (2570988, 1181105, 2571008, 1181115)),
        13: house_geometry(13, (2570800, 1181110, 2570910, 1181130)),
    }
    store, provider = fixture_store(tmp_path, monkeypatch, houses)
    selection = area_selection((2570900, 1181100, 2572100, 1181120))
    result = store.acquire(selection)
    assert result["failures"] == []
    assert {
        store.house(identifier).building_fid for identifier in result["houses"]
    } == {11, 12, 13}
    roi_bounds = transform(
        Transformer.from_crs(4326, provider.crs, always_xy=True).transform,
        selection_geometry(selection)[1],
    ).bounds
    assert np.allclose(provider.lidar_requests[0], roi_bounds)
    assert set(intersecting_tiles(provider.lidar_requests[0])) == {
        (2570000, 1181000),
        (2571000, 1181000),
        (2572000, 1181000),
    }
    # Building 13 crosses the selected boundary. Its west half must remain complete.
    item = next(
        store.house(identifier)
        for identifier in result["houses"]
        if store.house(identifier).building_fid == 13
    )
    assert item.bounds_xy == houses[13].bounds_xy
    assert (2570798, 1181108, 2570912, 1181132) in provider.lidar_requests
    assert len(result["houses"]) == len(set(result["houses"]))


def test_house_spanning_tile_edge_requests_neighbor_and_ids_are_stable(
    tmp_path, monkeypatch
):
    house = house_geometry(42, (2570995, 1181105, 2571005, 1181115))
    store, provider = fixture_store(tmp_path, monkeypatch, {42: house})
    longitude, latitude = TO_WGS84.transform(2570999, 1181110)
    selection = {"mode": "house", "longitude": longitude, "latitude": latitude}
    first = store.acquire(selection)
    item = store.house(first["houses"][0])
    assert {tuple(s["bounds"][:2]) for s in item.sources} == {
        (2570000, 1181000),
        (2571000, 1181000),
    }
    assert provider.lidar_requests == [(2570993, 1181103, 2571007, 1181117)]
    snapshot = item.as_dict()
    repeated = store.acquire(selection)
    assert repeated["id"] == first["id"]
    assert repeated["houses"] == first["houses"]
    assert len(provider.lidar_requests) == 1  # Existing provenance is reused.
    overlapping = store.acquire(area_selection((2570997, 1181107, 2571003, 1181113)))
    assert overlapping["houses"] == first["houses"]
    assert store.house(first["houses"][0]).as_dict() == snapshot
    assert (
        len(provider.building_requests) == 2
    )  # The repeat reused the original vector clip.


def test_roof_index_groups_multiple_surface_records_by_building_id(
    monkeypatch, tmp_path
):
    def roof(fid, x):
        return {
            "building_fid": fid,
            "object_type": "building",
            "paths": [
                [
                    (x, 1181000, 620),
                    (x + 4, 1181000, 620),
                    (x + 4, 1181004, 620),
                    (x, 1181004, 620),
                    (x, 1181000, 620),
                ]
            ],
        }

    monkeypatch.setattr(
        geometry_module,
        "_iter_roof_features",
        lambda _path: [roof(21, 2570000), roof(21, 2570004), roof(22, 2570020)],
    )
    index = geometry_module.build_house_index(tmp_path / "unused.gpkg")
    assert list(index.building_fid) == [21, 22]
    assert int(index.loc[index.building_fid == 21, "roof_face_count"].iloc[0]) == 2


def test_point_reader_keeps_source_multiplicity_and_assigns_seam_to_one_tile(
    tmp_path, monkeypatch
):
    west = write_las(
        tmp_path / "west.las",
        [
            (0, 0, 0),
            (1000, 1000, 20),
            (995, 15, 10),
            (995, 15, 10),
            (1000, 15, 10),
            (1005, 15, 10),
        ],
    )
    east = write_las(
        tmp_path / "east.las",
        [
            (1000, 0, 0),
            (2000, 1000, 20),
            (1000, 15, 10),
            (1005, 15, 10),
            (1005, 15, 10),
        ],
    )
    pinned = (source(west, (0, 0)), source(east, (1000, 0)))
    provider = SwissProvider(tmp_path / "cache")

    def unexpected_lookup(*_args):
        raise AssertionError("Pinned source files must not be reselected")

    monkeypatch.setattr(provider, "_tile", unexpected_lookup)
    points, used = provider.points_for_bounds(
        (990, 10, 1010, 20), 2024, pinned_sources=pinned
    )
    assert points.groupby("x").size().to_dict() == {995.0: 2, 1000.0: 1, 1005.0: 2}
    assert tuple(s["tile"] for s in used) == tuple(s["tile"] for s in pinned)
    assert list(points.classification) == [6] * 5


def test_lidar_rejects_file_without_requested_spatial_coverage(tmp_path, monkeypatch):
    short = write_las(tmp_path / "short.las", [(990, 10, 0), (995, 20, 10)])
    provider = SwissProvider(tmp_path / "cache")
    monkeypatch.setattr(provider, "_tile", lambda *_args: source(short, (0, 0)))
    with pytest.raises(ValueError, match="cover"):
        provider.lidar((990, 10, 999, 20), 2024)


def test_export_uses_pinned_inputs_and_preserves_original_returns(
    tmp_path, monkeypatch
):
    house = house_geometry(42, (2570995, 1181105, 2571005, 1181115))
    store, provider = fixture_store(tmp_path, monkeypatch, {42: house})
    longitude, latitude = TO_WGS84.transform(2571000, 1181110)
    house_id = store.acquire(
        {"mode": "house", "longitude": longitude, "latitude": latitude}
    )["houses"][0]
    manifest_path = store.export_scene(house_id, tmp_path / "scene-export")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "label3d-scene-v1"
    assert manifest["base_height"] == 610.0
    assert manifest["focus_extent"] == list(house.bounds_xy)
    outline = json.loads(
        (manifest_path.parent / manifest["building_outline"]).read_text()
    )
    assert outline["features"][0]["properties"]["z_min"] == 610.0
    assert provider.point_requests[0][2] == store.house(house_id).sources
    points = laspy.read(manifest_path.parent / manifest["points"])
    assert len(points.points) == 2
    assert list(points.z) == [620.0, 620.0]
    annotation = manifest_path.parent / "annotations.json"
    annotation.write_text('{"vertices":[{"id":"kept"}]}')
    store.export_scene(house_id, manifest_path.parent)
    assert annotation.read_text() == '{"vertices":[{"id":"kept"}]}'
    assert len(provider.point_requests) == 1


@pytest.fixture
def prepared_store(tmp_path, monkeypatch):
    house = house_geometry(42, (2570995, 1181105, 2571005, 1181115))
    store, _ = fixture_store(tmp_path, monkeypatch, {42: house})
    lon, lat = TO_WGS84.transform(2571000, 1181110)
    house_id = store.acquire({"mode": "house", "longitude": lon, "latitude": lat})[
        "houses"
    ][0]
    return store, house_id


@pytest.mark.parametrize(
    "house_id", ["../escape", "a/b", "..", "a\\b", "/etc/passwd", ""]
)
def test_house_lookup_rejects_path_ids(tmp_path, house_id):
    with pytest.raises(ValueError):
        AcquisitionStore(tmp_path).house(house_id)


@pytest.mark.parametrize("field", ["surfaces_path", "source_path"])
def test_house_lookup_rejects_manifest_path_escape(prepared_store, tmp_path, field):
    store, house_id = prepared_store
    path = store.root / "houses" / house_id / "house.json"
    payload = json.loads(path.read_text())
    if field == "source_path":
        payload["sources"][0]["path"] = "../outside.las"
    else:
        payload[field] = "../outside.gpkg"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outside"):
        store.house(house_id)


def test_house_lookup_rejects_symlink_escape(prepared_store, tmp_path):
    store, house_id = prepared_store
    external = tmp_path.parent / f"{tmp_path.name}-outside.gpkg"
    external.write_text("outside")
    link = store.root / "linked.gpkg"
    link.symlink_to(external)
    path = store.root / "houses" / house_id / "house.json"
    payload = json.loads(path.read_text())
    payload["surfaces_path"] = "linked.gpkg"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outside"):
        store.house(house_id)


@pytest.mark.parametrize(
    ("field", "value"), [("schema", "unsupported-v999"), ("id", "another-building")]
)
def test_house_lookup_validates_manifest_identity(prepared_store, field, value):
    store, house_id = prepared_store
    path = store.root / "houses" / house_id / "house.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        store.house(house_id)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"mode": "invalid"},
        {"mode": "house", "longitude": 0, "latitude": 0},
        {"mode": "area", "bbox": [7, 47, 6, 48]},
        {"mode": "area", "geometry": {"type": "Point", "coordinates": [7, 47]}},
        {"mode": "area", "geometry": {"type": "Feature", "geometry": None}},
    ],
)
def test_http_selection_validation_does_not_start_download(tmp_path, payload):
    app = FastAPI()
    store = AcquisitionStore(tmp_path)
    mount_acquisition(app, store)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/acquisition", json=payload)
    assert response.status_code == 422
    assert not list((tmp_path / "jobs" / "acquisition").glob("*.json"))


def test_semantic_vector_selection_keeps_source_fid_and_complete_geometry(tmp_path):
    from osgeo import ogr, osr
    from building_data.swiss_vectors import _clip_surface_layers

    driver = ogr.GetDriverByName("GPKG")
    original = tmp_path / "original.gpkg"
    selected = tmp_path / "selected.gpkg"
    dataset = driver.CreateDataSource(str(original))
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(2056)
    polygon_wkt = "POLYGON Z ((2570000 1181000 620,2570020 1181000 620,2570020 1181020 620,2570000 1181020 620,2570000 1181000 620))"
    for name in ("Roof", "Floor"):
        layer = dataset.CreateLayer(name, crs, ogr.wkbPolygon25D)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetFID(42)
        feature.SetGeometry(ogr.CreateGeometryFromWkt(polygon_wkt))
        assert layer.CreateFeature(feature) == 0
    feature = None
    layer = None
    dataset = None
    # The selected area covers only an interior slice of a larger source building.
    _clip_surface_layers(
        source_path=original,
        destination=selected,
        tile_bounds_xy=(2570005, 1181005, 2570010, 1181010),
    )
    dataset = ogr.Open(str(selected))
    for name in ("Roof", "Floor"):
        layer = dataset.GetLayerByName(name)
        feature = layer.GetNextFeature()
        assert feature.GetFID() == 42
        assert feature.GetGeometryRef().GetEnvelope() == (
            2570000,
            2570020,
            1181000,
            1181020,
        )
        assert layer.GetNextFeature() is None
    feature = None
    layer = None
    dataset = None


def test_invalid_roof_does_not_discard_valid_houses_and_zero_roofs_are_skipped(
    tmp_path, monkeypatch
):
    valid = house_geometry(42, (2570995, 1181105, 2571005, 1181115))
    store, _ = fixture_store(tmp_path, monkeypatch, {42: valid})
    monkeypatch.setattr(
        store_module,
        "build_house_index",
        lambda _path: pd.DataFrame(
            {"building_fid": [42, 43, 44, 42], "roof_face_count": [1, 1, 0, 1]}
        ),
    )
    loaded = []

    def load(_path, fid, **_kwargs):
        loaded.append(fid)
        if fid == 43:
            raise RuntimeError("Unusable roof geometry")
        if fid == 44:
            raise AssertionError("Known zero-roof feature should not be loaded")
        return valid

    monkeypatch.setattr(store_module, "load_vector_house", load)
    result = store.acquire(area_selection((2570990, 1181100, 2571010, 1181120)))
    assert len(result["houses"]) == 1
    assert store.house(result["houses"][0]).building_fid == 42
    assert result["failures"] == [
        {"building_fid": 43, "error": "Unusable roof geometry"}
    ]
    assert loaded == [42, 43]


def test_all_invalid_roofs_return_individual_diagnostics(tmp_path, monkeypatch):
    store, _ = fixture_store(
        tmp_path,
        monkeypatch,
        {42: house_geometry(42, (2570995, 1181105, 2571005, 1181115))},
    )

    def bad_roof(*_args, **_kwargs):
        raise RuntimeError("Invalid roof face")

    monkeypatch.setattr(store_module, "load_vector_house", bad_roof)
    result = store.acquire(area_selection((2570990, 1181100, 2571010, 1181120)))
    assert result["houses"] == []
    assert result["failures"] == [{"building_fid": 42, "error": "Invalid roof face"}]


def test_provider_controls_projection_identity_and_coverage(tmp_path, monkeypatch):
    house = house_geometry(42, (499995, 5199995, 500005, 5200005))
    store, provider = fixture_store(tmp_path, monkeypatch, {42: house})
    provider.crs = "EPSG:32632"
    longitude, latitude = Transformer.from_crs(
        provider.crs, 4326, always_xy=True
    ).transform(500000, 5200000)
    result = store.acquire(
        {"mode": "house", "longitude": longitude, "latitude": latitude}
    )
    assert result["houses"] == ["fixture-v1-42"]
    assert provider.validated[0].x == pytest.approx(longitude)
    assert provider.validated[0].y == pytest.approx(latitude)
    item = store.house(result["houses"][0])
    assert item.crs == provider.crs
    assert item.as_dict()["longitude"] == pytest.approx(longitude)
    assert item.as_dict()["latitude"] == pytest.approx(latitude)
    with pytest.raises(ValueError, match="covers"):
        store.acquire({"mode": "house", "longitude": 0, "latitude": 0})


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Feature", "geometry": None},
        {"type": "Feature", "geometry": []},
        {"type": "Polygon", "coordinates": [[[7, 47]]]},
        {"type": "Polygon", "coordinates": []},
        {"type": 42},
    ],
)
def test_geometry_validation_rejects_malformed_geojson(geometry):
    with pytest.raises(ValueError):
        selection_geometry({"mode": "area", "geometry": geometry})


def test_generic_selection_validation_is_not_limited_to_swiss_provider():
    mode, geometry = selection_geometry(
        {"mode": "house", "longitude": -73.98, "latitude": 40.75}
    )
    assert mode == "house"
    assert geometry.x == -73.98 and geometry.y == 40.75
