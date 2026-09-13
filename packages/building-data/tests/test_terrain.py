"""Scene DEM sampling must retain the existing ground-extrusion elevations."""

import laspy
import numpy as np
import pytest
from building_data import terrain
from building_data.orthophoto_correction.models import RasterAsset
from building_data.orthophoto_correction.raster import write_geotiff
from osgeo import gdal


def source(
    tmp_path, *, minimum=(2570000.013, 1181000.017), maximum=(2570999.987, 1181999.991)
):
    path = tmp_path / "source.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([2570000, 1181000, 500])
    points = laspy.LasData(header)
    points.x = [minimum[0], maximum[0]]
    points.y = [minimum[1], maximum[1]]
    points.z = [500, 600]
    points.write(path)
    return ({"path": str(path), "bounds": [2570000, 1181000, 2571000, 1182000]},)


def test_grid_uses_actual_header_and_expands_without_changing_phase(tmp_path):
    sources = source(tmp_path)
    base = terrain.terrain_patch_bounds((2570500, 1181500, 2570520, 1181520), sources)
    assert base == pytest.approx((2569968.013, 1180968.017, 2571031.987, 1182031.991))
    expanded = terrain.terrain_patch_bounds(
        (2569900, 1180900, 2571100, 1182100), sources
    )
    for before, after in zip(base, expanded):
        assert (after - before) / 0.5 == pytest.approx(round((after - before) / 0.5))
    assert expanded[0] <= 2569900 and expanded[1] <= 1180900
    assert expanded[2] >= 2571100 and expanded[3] >= 1182100


@pytest.mark.parametrize(
    "bounds,window",
    [
        ((100.63, 200.51, 105.89, 205.78), (1, 8, 11, 11)),
        ((99, 199, 105, 205), (0, 9, 10, 11)),
        ((100.4999, 200.49, 105.2499, 205.24), (1, 9, 10, 10)),
    ],
)
def test_crop_matches_original_rounded_window(tmp_path, bounds, window):
    values = np.arange(400, dtype=np.float32).reshape(20, 20)
    original = tmp_path / "patch.tif"
    write_geotiff(original, values, bounds_lv95=(100, 199.99, 110, 209.99))
    output = terrain.crop_terrain(original, tmp_path / "dem.tif", bounds)
    raster = gdal.Open(str(output))
    x, y, width, height = window
    np.testing.assert_array_equal(
        raster.ReadAsArray(), values[y : y + height, x : x + width]
    )
    assert raster.GetGeoTransform() == pytest.approx(
        (100 + x * 0.5, 0.5, 0, 209.99 - y * 0.5, 0, -0.5)
    )


def test_terrain_source_selection_is_pinned_and_preserves_remote_warp_branch(
    tmp_path, monkeypatch
):
    sources = source(tmp_path)
    asset = RasterAsset(
        "ch.swisstopo.swissalti3d",
        "tile2021",
        "dem",
        "https://example.test/dem_2056_5728.tif",
        0.5,
        2021,
        metadata={"checksum": "1220" + "a" * 64, "size": 1234},
    )
    searches = []

    def search(*args, **kwargs):
        searches.append(1)
        return [asset]

    monkeypatch.setattr(terrain, "search_raster_assets", search)

    def download(*args, **kwargs):
        assert kwargs == {"sha256": "a" * 64, "expected_size": 1234}
        return tmp_path / "raw.tif"

    monkeypatch.setattr(terrain, "download", download)

    def read(assets, **kwargs):
        assert assets[0].href.startswith("https://")
        assert kwargs["source_paths"][asset.href] == tmp_path / "raw.tif"
        assert kwargs["gsd_m"] == 0.5 and kwargs["resample_alg"] == "bilinear"
        return np.ones((20, 20), dtype=np.float32) * 600, (
            2570500,
            1181500,
            2570510,
            1181510,
        )

    monkeypatch.setattr(terrain, "read_raster_crop", read)
    bounds = (2570501, 1181501, 2570509, 1181509)
    first = terrain.swiss_terrain(
        bounds, tmp_path / "first.tif", tmp_path / "cache", sources
    )
    second = terrain.swiss_terrain(
        bounds, tmp_path / "second.tif", tmp_path / "cache", sources
    )
    assert len(searches) == 1
    assert first.read_bytes() == second.read_bytes()
    patch = next((tmp_path / "cache" / "terrain").glob("*/terrain.tif"))
    patch.unlink()
    terrain.swiss_terrain(bounds, tmp_path / "third.tif", tmp_path / "cache", sources)
    assert len(searches) == 1


def test_rejects_wrong_height_datum(tmp_path, monkeypatch):
    sources = source(tmp_path)
    asset = RasterAsset(
        "ch.swisstopo.swissalti3d",
        "tile",
        "dem",
        "https://example.test/dem_2056_5710.tif",
        0.5,
        2021,
    )
    monkeypatch.setattr(
        terrain, "search_raster_assets", lambda *args, **kwargs: [asset]
    )
    with pytest.raises(ValueError, match="height datum"):
        terrain.swiss_terrain(
            (2570501, 1181501, 2570509, 1181509),
            tmp_path / "dem.tif",
            tmp_path / "cache",
            sources,
        )


def test_existing_scene_backfills_only_dem_and_keeps_all_user_data(
    tmp_path, monkeypatch
):
    import json

    from test_acquisition import TO_WGS84, fixture_store, house_geometry

    house = house_geometry(42, (2570995, 1181105, 2571005, 1181115))
    store, provider = fixture_store(tmp_path, monkeypatch, {42: house})
    longitude, latitude = TO_WGS84.transform(2571000, 1181110)
    house_id = store.acquire(
        {"mode": "house", "longitude": longitude, "latitude": latitude}
    )["houses"][0]
    path = store.export_scene(house_id, tmp_path / "scene")
    manifest = json.loads(path.read_text())
    manifest.update(earth_alignment={"de": 2, "dn": 3, "dh": 4}, custom={"preserve": 5})
    path.write_text(json.dumps(manifest))
    for name in ["annotations.json", "mesh.ply"]:
        (path.parent / name).write_bytes(b"user data unchanged")
    before = {
        p.name: p.read_bytes() for p in path.parent.iterdir() if p.name != "scene.json"
    }
    requests = []

    def provide(bounds, output, *, pinned_sources=()):
        requests.append((bounds, pinned_sources))
        write_geotiff(
            output, np.ones((10, 10), dtype=np.float32) * 600, bounds_lv95=bounds
        )
        return output

    monkeypatch.setattr(provider, "terrain", provide)
    store.export_scene(house_id, path.parent)
    assert json.loads(path.read_text()) == {**manifest, "dem": "dem.tif"}
    assert requests[0][0] == (2570970, 1181080, 2571030, 1181140)
    assert requests[0][1] == store.house(house_id).sources
    assert all(
        (path.parent / name).read_bytes() == value for name, value in before.items()
    )
    assert len(provider.point_requests) == 1
    dem = path.parent / "dem.tif"
    dem.write_bytes(b"user modified DEM")
    store.export_scene(house_id, path.parent)
    assert len(requests) == 1 and dem.read_bytes() == b"user modified DEM"
