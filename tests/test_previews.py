"""Preview pixels, source choices and isolation from saved reconstruction inputs."""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
import tifffile
from PIL import Image
from shapely.geometry import box

from emboss import previews
from emboss.api import Reconstruction


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "results/house-1"
    (root / "inputs").mkdir(parents=True)
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    labels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    tifffile.imwrite(root / "inputs/orthophoto.tif", rgb)
    tifffile.imwrite(root / "classes.tif", labels)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "orthophoto": "inputs/orthophoto.tif",
                    "segmentation": "classes.tif",
                }
            }
        )
    )
    metadata = {
        "source_year": 2023,
        "requested_strip_override": None,
        "strip_candidates": [
            {"id": "strip-a", "flight_date": "2023-05-29", "flight_year": 2023},
            {"id": "strip-b", "flight_date": "2023-06-05", "flight_year": 2023},
        ],
        "candidate_selection": {
            "selected_id": "strip-a",
            "override_id": None,
            "candidate_scores": [
                {"candidate_id": "strip-a", "roof_iou": 0.8},
                {"candidate_id": "strip-b", "roof_iou": 0.6},
                {"candidate_id": "raw", "roof_iou": 0.3},
            ],
        },
    }
    (root / "inputs/orthophoto_correction.json").write_text(json.dumps(metadata))
    surfaces = tmp_path / "surfaces.gpkg"
    surfaces.write_bytes(b"test geometry identity")
    lidar = tmp_path / "points.las"
    data = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    data.x, data.y, data.z = [0, 10], [0, 10], [0, 1]
    data.write(lidar)
    house_root = tmp_path / "houses/house-1"
    (house_root / "imagery").mkdir(parents=True)
    (house_root / "imagery/rgb_corrected.tif").write_bytes(b"keep saved image")
    item = SimpleNamespace(
        id="house-1",
        root=house_root,
        surfaces_path=surfaces,
        building_fid=1,
        bounds_xy=(1, 1, 3, 3),
        sources=[
            {
                "tile": "swisssurface3d_2024_0-0",
                "path": str(lidar),
                "bounds": [0, 0, 10, 10],
            }
        ],
    )
    result = Reconstruction(root, root / "bundle.json", item)
    client = SimpleNamespace(
        store=SimpleNamespace(root=tmp_path),
        load_result=lambda _house_id: result,
        _inference_lock=threading.Lock(),
    )
    return client, result, rgb, labels


def test_saved_pngs_preserve_grid_pixels_and_class_palette(saved):
    _client, result, rgb, labels = saved
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(previews.orthophoto_png(result)))), rgb
    )
    rgba = np.asarray(Image.open(io.BytesIO(previews.segmentation_png(result))))
    assert rgba.shape == (*labels.shape, 4)
    assert (rgba[labels == 5] == 0).all()
    for class_id, _label, color in previews.CLASS_PALETTE:
        expected = [*(int(color[i : i + 2], 16) for i in (1, 3, 5)), 180]
        np.testing.assert_array_equal(rgba[labels == class_id][0], expected)


def test_sources_use_result_snapshot_and_include_uncorrected_option(saved):
    _client, result, _rgb, _labels = saved
    (result.house_input.root / "imagery/orthophoto_correction.json").write_text(
        '{"source_year":1999}'
    )
    info = previews.imagery_info(result)
    assert info["source_year"] == 2023
    assert info["selected_id"] == "strip-a"
    assert info["override_id"] is None
    assert [row["id"] for row in info["candidates"]] == ["raw", "strip-a", "strip-b"]
    assert info["candidates"][0]["kind"] == "raw"
    assert info["candidates"][1]["flight_date"] == "2023-05-29"
    assert info["candidates"][1]["roof_iou"] == 0.8
    assert [row["id"] for row in info["classes"]] == [0, 1, 2, 3, 4]


def test_preview_reuses_method_and_keeps_saved_files_unchanged(saved, monkeypatch):
    client, result, rgb, _labels = saved
    protected = [*result.root.rglob("*"), *result.house_input.root.rglob("*")]
    before = {path: path.read_bytes() for path in protected if path.is_file()}
    house = SimpleNamespace(roof_envelope=box(1, 1, 3, 3), bounds_xy=(1, 1, 3, 3))
    monkeypatch.setattr(previews, "load_vector_house", lambda *_: house)
    calls = []
    assets = {"swissimage": ()}

    def prepare(**kwargs):
        assert kwargs["bounds_lv95"] == (0.0, 0.0, 10.0, 10.0)
        assert kwargs["required_bounds_lv95"] == (-7.0, -7.0, 11.0, 11.0)
        return assets

    def correct(state, workspace, *, padding_m, config):
        assert state["house"] is house
        assert workspace.correction_raster_assets is assets
        assert padding_m == 8.0
        assert config.strip_id_override == "strip-b"
        assert client._inference_lock.locked()
        calls.append(config.strip_id_override)
        return SimpleNamespace(rgb_corrected=rgb + 20)

    monkeypatch.setattr(previews, "prepare_assets", prepare)
    monkeypatch.setattr(previews, "correct_house_orthophoto", correct)
    png = previews.preview_candidate(client, "house-1", "strip-b")
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png))), rgb + 20)
    assert previews.preview_candidate(client, "house-1", "strip-b") == png
    assert calls == ["strip-b"]
    assert all(path.read_bytes() == content for path, content in before.items())
    # Corrupt previews are rebuilt; changed inputs select a different cache entry.
    next(
        (client.store.root / "cache/source-previews").rglob("orthophoto.png")
    ).write_bytes(b"broken")
    assert previews.preview_candidate(client, "house-1", "strip-b") == png
    result.house_input.surfaces_path.write_bytes(b"changed geometry identity")
    assert previews.preview_candidate(client, "house-1", "strip-b") == png
    assert calls == ["strip-b"] * 3


@pytest.mark.parametrize("candidate_id", ["../strip-a", "unknown", ""])
def test_unknown_preview_rejected_before_preparation(saved, monkeypatch, candidate_id):
    client, _result, _rgb, _labels = saved
    monkeypatch.setattr(
        previews, "prepare_assets", lambda **_: pytest.fail("Unexpected preparation")
    )
    with pytest.raises(ValueError, match="not available"):
        previews.preview_candidate(client, "house-1", candidate_id)
    assert not (client.store.root / "cache/source-previews").exists()


def test_selected_preview_uses_saved_pixels_without_new_correction(saved, monkeypatch):
    client, result, _rgb, _labels = saved
    monkeypatch.setattr(
        previews,
        "correct_house_orthophoto",
        lambda *_args, **_kwargs: pytest.fail("Unexpected correction"),
    )
    assert previews.preview_candidate(
        client, "house-1", "strip-a"
    ) == previews.orthophoto_png(result)


def test_invalid_class_map_is_not_silently_displayed(saved):
    _client, result, _rgb, _labels = saved
    tifffile.imwrite(result.root / "classes.tif", np.array([[255]], dtype=np.uint8))
    with pytest.raises(ValueError, match="RID2"):
        previews.segmentation_png(result)


@pytest.mark.parametrize(
    "transform", [[100, 2, 0, 200, 0, -2], [100, 2, 0.5, 200, 0.25, -2]]
)
def test_footprint_projection_preserves_holes_and_parts(saved, transform):
    _client, result, rgb, _labels = saved
    pixel_rings = [
        [[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]],
        [[1, 0.5], [2, 0.5], [2, 1], [1, 1], [1, 0.5]],
        [[4, 0], [5, 0], [5, 1], [4, 1], [4, 0]],
    ]
    x0, dx, rx, y0, ry, dy = transform
    rings = [
        [[x0 + dx * x + rx * y, y0 + ry * x + dy * y, 50] for x, y in ring]
        for ring in pixel_rings
    ]
    (result.root / "vector_house.geojson").write_text(
        json.dumps(
            {
                "crs": {"properties": {"name": "EPSG:2056"}},
                "features": [
                    {
                        "properties": {"kind": "roof_envelope"},
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [[rings[0], rings[1]], [rings[2]]],
                        },
                    }
                ],
            }
        )
    )
    bundle = json.loads(result.manifest_path.read_text())
    bundle["artifacts"]["vector_house"] = "vector_house.geojson"
    result.manifest_path.write_text(json.dumps(bundle))
    metadata_path = result.root / "inputs/orthophoto_correction.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(transform=transform, crs="EPSG:2056", width=3, height=2)
    metadata_path.write_text(json.dumps(metadata))
    overlay = previews.imagery_info(result)["footprint"]
    assert (overlay["width"], overlay["height"]) == (3, 2)
    np.testing.assert_allclose(overlay["rings"], pixel_rings, atol=1e-10)
    np.testing.assert_array_equal(tifffile.imread(result.orthophoto_path), rgb)


def test_footprint_uses_declared_provider_crs(saved):
    from pyproj import Transformer

    _client, result, _rgb, _labels = saved
    ring = [[8, 47], [8.001, 47], [8.001, 47.001], [8, 47]]
    (result.root / "vector_house.geojson").write_text(
        json.dumps(
            {
                "crs": {"properties": {"name": "EPSG:4326"}},
                "features": [
                    {
                        "properties": {"kind": "roof_envelope"},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                ],
            }
        )
    )
    bundle = json.loads(result.manifest_path.read_text())
    bundle["artifacts"]["vector_house"] = "vector_house.geojson"
    result.manifest_path.write_text(json.dumps(bundle))
    projection = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = projection.transform(8, 47)
    overlay = previews.footprint_overlay(
        result,
        {
            "transform": [x0, 1, 0, y0, 0, -1],
            "crs": "EPSG:3857",
            "width": 300,
            "height": 300,
        },
    )
    expected = [
        [x - x0, y0 - y] for x, y in [projection.transform(*point) for point in ring]
    ]
    np.testing.assert_allclose(overlay["rings"][0], expected)


@pytest.fixture
def gallery(saved, monkeypatch):
    client, result, rgb, labels = saved
    item = result.house_input
    item.crs = "EPSG:2056"
    client.store.house = lambda _house_id: item
    client.load_result = lambda _: (_ for _ in ()).throw(
        FileNotFoundError("No reconstruction")
    )
    calls = []
    house = SimpleNamespace(
        roof_envelope=box(1, 1, 3, 3),
        bounds_xy=(1, 1, 3, 3),
        building_fid=1,
        is_independent=True,
        touching_building_fids=(),
        roof_faces=(),
        roof_segments=(),
        wall_faces=(),
        floor_faces=(),
    )
    monkeypatch.setattr(previews, "load_vector_house", lambda *_: house)
    monkeypatch.setattr(previews, "prepare_assets", lambda **_: {"swissimage": ()})

    def correct(state, workspace, *, padding_m, config):
        assert client._inference_lock.locked()
        calls.append(config.strip_id_override)
        metadata = json.loads(
            (result.root / "inputs/orthophoto_correction.json").read_text()
        )
        metadata.update(
            transform=[1, 2 / 3, 0, 3, 0, -1], crs="EPSG:2056", width=3, height=2
        )
        image = rgb if config.strip_id_override is None else rgb + 10
        return SimpleNamespace(rgb_corrected=image, metadata=metadata)

    monkeypatch.setattr(previews, "correct_house_orthophoto", correct)
    return client, result, rgb, labels, calls


def test_prepare_before_reconstruction_is_cached_and_does_not_choose_saved_inputs(
    gallery,
):
    client, result, rgb, _labels, calls = gallery
    before = {
        path: path.read_bytes()
        for root in [result.root, result.house_input.root]
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileNotFoundError, match="Prepare"):
        previews.house_preview_candidate(client, "house-1", "raw")
    info = previews.prepare_house_imagery(client, "house-1")
    assert not info["has_reconstruction"]
    assert info["current_id"] is None
    assert info["selected_id"] == info["recommended_id"] == "strip-a"
    assert info["footprint"]["width"] == 3
    assert calls == [None]  # Only automatic image correction, never segmentation.
    assert previews.prepare_house_imagery(client, "house-1") == info
    np.testing.assert_array_equal(
        np.asarray(
            Image.open(
                io.BytesIO(
                    previews.house_preview_candidate(client, "house-1", "strip-a")
                )
            )
        ),
        rgb,
    )
    alternate = previews.house_preview_candidate(client, "house-1", "strip-b")
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(alternate))), rgb + 10
    )
    assert calls == [None, "strip-b"]
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not list(
        (client.store.root / "cache/image-galleries").rglob("*segmentation*")
    )
    assert not list((client.store.root / "cache/image-galleries").rglob("mesh.ply"))


def test_gallery_invalidates_changed_source_and_repairs_corrupt_snapshot(gallery):
    client, _result, _rgb, _labels, calls = gallery
    previews.prepare_house_imagery(client, "house-1")
    cache = client.store.root / "cache/image-galleries"
    next(cache.rglob("orthophoto.tif")).write_bytes(b"corrupt image")
    previews.prepare_house_imagery(client, "house-1")
    client.store.house("house-1").surfaces_path.write_bytes(b"changed source")
    previews.prepare_house_imagery(client, "house-1")
    assert calls == [None, None, None]


def test_saved_automatic_result_reused_without_new_correction(saved, monkeypatch):
    client, result, _rgb, _labels = saved
    client.store.house = lambda _: result.house_input
    monkeypatch.setattr(
        previews, "_correct_preview", lambda *_: pytest.fail("Unnecessary correction")
    )
    info = previews.prepare_house_imagery(client, "house-1")
    assert info["has_reconstruction"]
    assert info["current_id"] == info["recommended_id"] == "strip-a"
    assert not (client.store.root / "cache/image-galleries").exists()


def test_manual_saved_choice_remains_current_while_default_is_ranked_separately(
    gallery,
):
    client, result, rgb, _labels, calls = gallery
    client.load_result = lambda _: result
    path = result.root / "inputs/orthophoto_correction.json"
    metadata = json.loads(path.read_text())
    metadata["requested_strip_override"] = "strip-b"
    metadata["candidate_selection"]["selected_id"] = "strip-b"
    path.write_text(json.dumps(metadata))
    # The mock correction's metadata describes automatic ranking independently.
    correct = previews.correct_house_orthophoto

    def default(*args, **kwargs):
        output = correct(*args, **kwargs)
        output.metadata["requested_strip_override"] = None
        output.metadata["candidate_selection"]["selected_id"] = "strip-a"
        return output

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(previews, "correct_house_orthophoto", default)
    try:
        info = previews.prepare_house_imagery(client, "house-1")
        assert info["current_id"] == info["selected_id"] == "strip-b"
        assert info["recommended_id"] == "strip-a"
        assert info["override_id"] == "strip-b"
        np.testing.assert_array_equal(
            np.asarray(
                Image.open(
                    io.BytesIO(
                        previews.house_preview_candidate(client, "house-1", "strip-b")
                    )
                )
            ),
            rgb,
        )
        assert calls == [None]
    finally:
        monkeypatch.undo()
