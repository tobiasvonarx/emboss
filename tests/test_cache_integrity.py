"""Exercise real Client cache decisions with a deterministic computation fixture."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.storage import checksum, write_json
from shapely.geometry import box

from emboss import api


@dataclass(frozen=True)
class CachedHouse:
    roof_envelope: object
    bounds_xy: tuple
    is_independent: bool = True
    touching_building_fids: tuple = ()


@pytest.fixture
def cached_client(tmp_path, monkeypatch):
    client = api.Client(tmp_path, workers=1)
    inputs = tmp_path / "houses/house"
    imagery = inputs / "imagery"
    imagery.mkdir(parents=True)
    point_file = tmp_path / "points.bin"
    point_file.write_text("2")
    (tmp_path / "surfaces.gpkg").write_bytes(b"fixture-vector")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    write_json(
        inputs / "house.json",
        {
            "schema": "building-input-v1",
            "id": "house",
            "building_fid": 1,
            "units": "m",
            "crs": "EPSG:2056",
            "bounds_xy": [100, 100, 104, 104],
            "surfaces_path": "surfaces.gpkg",
            "survey_year": 2024,
            "sources": [
                {
                    "path": "points.bin",
                    "tile": "fixture",
                    "year": 2024,
                    "bounds": [0, 0, 1000, 1000],
                }
            ],
        },
    )
    item = client.store.house("house")
    fingerprint = api._imagery_fingerprint(
        item, CorrectionConfig(), checksum(item.surfaces_path)
    )
    tifffile.imwrite(
        imagery / "rgb_corrected.tif", np.zeros((40, 40, 3), dtype=np.uint8)
    )
    write_json(
        imagery / "orthophoto_correction.json",
        {
            "imagery_fingerprint": fingerprint,
            "extent_lv95": [100, 100, 104, 104],
            "requested_strip_override": None,
        },
    )
    house = CachedHouse(
        roof_envelope=box(100, 100, 104, 104), bounds_xy=(100, 100, 104, 104)
    )
    monkeypatch.setattr(
        api,
        "build_house_index",
        lambda _: pd.DataFrame(
            [
                {
                    "building_fid": 1,
                    "object_type": "fixture",
                    "surface_count": 1,
                    "roof_face_count": 1,
                    "footprint_area_m2": 16,
                    "is_independent": True,
                    "touching_building_fids": "",
                    "min_x": 100,
                    "min_y": 100,
                    "max_x": 104,
                    "max_y": 104,
                    "min_z": 0,
                    "max_z": 0,
                }
            ]
        ),
    )
    monkeypatch.setattr(api, "load_vector_house", lambda *_: house)
    monkeypatch.setattr(api, "load_vector_house_scaffold", lambda *_: house)
    monkeypatch.setattr(
        api,
        "write_vector_house_geojson",
        lambda path, *_, **__: path.write_text("fixture-scaffold"),
    )
    monkeypatch.setattr(
        api, "prepare_lidar_observations_from_points", lambda frame, _: frame
    )
    monkeypatch.setattr(
        client.store.provider,
        "points_for_bounds",
        lambda *_, **__: (
            pd.DataFrame({"classification": [6] * int(point_file.read_text())}),
            (),
        ),
    )
    client.segmentation = SimpleNamespace(
        checkpoint_path=checkpoint,
        predict_hard_segmentation=lambda *_, **__: SimpleNamespace(
            foreground_class_map=np.full((40, 40), 5, dtype=np.uint8), diagnostics={}
        ),
    )
    calls = []

    def computation(*, output_dir, observations, **_):
        count = len(observations)
        calls.append(count)
        (output_dir / "rasters").mkdir()
        write_json(output_dir / "result.json", {"solid_count": count})
        for name in ("superstructures.geojson", "vector_house.geojson"):
            write_json(output_dir / name, {"type": "FeatureCollection", "features": []})
        for name in ("superstructure_mask.tif", "image_superstructure_class_map.tif"):
            tifffile.imwrite(
                output_dir / "rasters" / name, np.zeros((40, 40), dtype=np.uint8)
            )
        return SimpleNamespace(solids=[None] * count)

    def mesh(*, output_path, **_):
        output_path.write_bytes(b"fixture-mesh")
        return {}

    monkeypatch.setattr(api, "run_emboss_house", computation)
    monkeypatch.setattr(api, "write_swissbuildings3d_mesh", mesh)
    monkeypatch.setattr(api, "write_prediction_mesh", mesh)
    result = client.reconstruct("house")
    assert result.as_dict()["solid_count"] == 2
    assert calls == [2]
    return client, result, point_file, calls


def test_unchanged_inputs_reuse_complete_result(cached_client):
    client, result, _, calls = cached_client
    assert client.reconstruct("house").root == result.root
    assert calls == [2]


def test_changed_lidar_content_with_same_size_and_timestamp_is_recomputed(
    cached_client,
):
    client, _, path, calls = cached_client
    before = path.stat()
    path.write_text("0")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    result = client.reconstruct("house")
    assert result.as_dict()["solid_count"] == 0
    assert calls == [2, 0]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["provenance"]["lidar_sha256"] == [checksum(path)]


@pytest.mark.parametrize(
    "artifact", ["mesh.ply", "result.json", "inputs/orthophoto.tif"]
)
def test_changed_result_artifact_is_detected_and_rebuilt(cached_client, artifact):
    client, result, _, calls = cached_client
    (result.root / artifact).write_bytes(b"corrupted-output")
    with pytest.raises(ValueError, match="checksum mismatch"):
        client.load_result("house")
    assert client.reconstruct("house").as_dict()["solid_count"] == 2
    assert calls == [2, 2]


def test_missing_result_artifact_is_rebuilt(cached_client):
    client, result, _, calls = cached_client
    result.mesh_path.unlink()
    assert client.reconstruct("house").mesh_path.exists()
    assert calls == [2, 2]


def test_interrupted_manifest_is_rebuilt(cached_client):
    client, result, _, calls = cached_client
    result.manifest_path.write_text("{")
    assert client.reconstruct("house").as_dict()["solid_count"] == 2
    assert calls == [2, 2]


@pytest.mark.parametrize(
    "field,value", [("artifacts", {}), ("artifacts", []), ("artifact_sha256", None)]
)
def test_invalid_artifact_manifest_is_rebuilt(cached_client, field, value):
    client, result, _, calls = cached_client
    bundle = json.loads(result.manifest_path.read_text())
    bundle[field] = value
    write_json(result.manifest_path, bundle)
    assert client.reconstruct("house").as_dict()["solid_count"] == 2
    assert calls == [2, 2]
