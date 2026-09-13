"""Opt-in original-source numerical and export parity; no original checkout execution."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from shapely.geometry import Polygon, box, mapping
from test_segmentation_parity import _reference_module, _reference_root


def _original_mesh():
    return _reference_module(
        _reference_root() / "src/emboss/eval/cad_mesh.py",
        "emboss",
        (
            ("from .schema import EvalCase", "from typing import Any as EvalCase"),
            ("from ..geometry import", "from building_data.geometry import"),
            (
                "from ..zurich_dachmodell import ZURICH_DACHMODELL_BODY_GROUND_LAYER",
                "ZURICH_DACHMODELL_BODY_GROUND_LAYER = 'unused'",
            ),
            (
                "from ..zurich_dachmodell import ZURICH_DACHMODELL_BODY_WALL_LAYER",
                "ZURICH_DACHMODELL_BODY_WALL_LAYER = 'unused'",
            ),
            (
                "from ..zurich_dachmodell import ZURICH_DACHMODELL_MESH_ROOF_LAYER",
                "ZURICH_DACHMODELL_MESH_ROOF_LAYER = 'unused'",
            ),
        ),
    )


def _write_features(path, features):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(geometry),
                        "properties": properties,
                    }
                    for geometry, properties in features
                ],
            }
        )
    )


@pytest.mark.parametrize("shape", ["empty", "concave", "step", "ridge", "duplicates"])
def test_original_prediction_mesh_bytes_match(shape, tmp_path):
    from emboss.mesh import write_prediction_mesh

    original = _original_mesh()
    scaffold = tmp_path / "scaffold.geojson"
    details = tmp_path / "details.geojson"
    _write_features(
        scaffold,
        [(box(0, 0, 4, 5), {"kind": "roof_segment", "plane_coeffs": [0.0, 0.0, 0.0]})],
    )
    features = []
    if shape == "concave":
        features = [
            (
                Polygon([(0, 0), (3, 0), (3, 3), (1, 1), (0, 3)]),
                {"height_offset_m": 0.2},
            )
        ]
    elif shape in {"step", "ridge"}:
        planes = (
            ([0.0, 0.0, 2.0], [0.0, 0.0, 3.0])
            if shape == "step"
            else ([1.0, 0.0, 1.0], [-1.0, 0.0, 3.0])
        )
        features = [
            (
                box(0, 0, 2, 1),
                {
                    "top_faces": [
                        {
                            "footprint_xy": list(box(0, 0, 1, 1).exterior.coords[:-1]),
                            "plane": planes[0],
                        },
                        {
                            "footprint_xy": list(box(1, 0, 2, 1).exterior.coords[:-1]),
                            "plane": planes[1],
                        },
                    ]
                },
            )
        ]
    elif shape == "duplicates":
        points = [
            (2.733, 1.579),
            (3.646, 3.2),
            (1.801, 4.239),
            (1.586, 3.856),
            (0.0, 1.038),
            (0.061, 1.004),
            (2.073, 1.925),
            (2.073, 1.925),
            (0.061, 1.004),
            (1.845, 0.0),
            (2.366, 0.926),
            (2.494, 1.153),
        ]
        features = [
            (
                box(0, 0, 4, 5),
                {"top_faces": [{"footprint_xy": points, "plane": [0.0, 0.0, 2.0]}]},
            )
        ]
    _write_features(details, features)
    kwargs = {"superstructures_path": details, "crs": "EPSG:2056"}
    expected = original.write_prediction_mesh_from_case(
        case=SimpleNamespace(roof_scaffold=scaffold, crs="EPSG:2056"),
        output_path=tmp_path / "old.ply",
        **kwargs,
    )
    actual = write_prediction_mesh(
        roof_scaffold=scaffold, output_path=tmp_path / "new.ply", **kwargs
    )
    assert actual == expected
    assert (tmp_path / "old.ply").read_bytes() == (tmp_path / "new.ply").read_bytes()


def _original_core():
    root = _reference_root() / "src/emboss"
    lidar = _reference_module(root / "lidar.py", "emboss")
    support = _reference_module(
        root / "return_support.py",
        "emboss",
        (("from .geometry import", "from building_data.geometry import"),),
    )
    support.residual_thresholds = lidar.residual_thresholds
    fitter = _reference_module(
        root / "superstructure_fitting/core.py",
        "emboss.superstructure_fitting",
        (("from ..geometry import", "from building_data.geometry import"),),
    )
    fitter.return_support_partition = support.return_support_partition
    export = _reference_module(
        root / "export.py",
        "emboss",
        (
            ("from .raster import", "from building_data.raster import"),
            ("from .serialization import", "from building_data.serialization import"),
        ),
    )
    tree = ast.parse((root / "pipeline.py").read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_emboss_house"
    )
    namespace = {
        "build_return_support_model": support.build_return_support_model,
        "fit_superstructures": fitter.fit_superstructures,
        "export_house_result": export.export_house_result,
        "Path": Path,
        "pd": pd,
        "VectorHouse": object,
        "HouseResult": object,
    }
    exec(  # noqa: S102 - execute explicitly selected original source function
        compile(
            ast.Module(body=[node], type_ignores=[]), "isolated_original_core", "exec"
        ),
        namespace,
    )
    return lidar, namespace["run_emboss_house"]


def _normalized_payload(value, output_dir):
    if isinstance(value, dict):
        return {
            key: _normalized_payload(item, output_dir) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalized_payload(item, output_dir) for item in value]
    if isinstance(value, str):
        return value.replace(str(output_dir), "<OUTPUT>")
    return value


@pytest.mark.parametrize(
    "configured",
    (
        os.environ.get("EMBOSS_RECONSTRUCTION_FIXTURES")
        or os.environ.get("EMBOSS_RECONSTRUCTION_FIXTURE")
        or ""
    ).split(os.pathsep),
    ids=lambda value: Path(value).name if value else "unconfigured",
)
def test_real_house_core_and_full_building_mesh_match_original(tmp_path, configured):
    if not configured:
        pytest.skip(
            "Set EMBOSS_RECONSTRUCTION_FIXTURE to copied house fixture directory."
        )
    fixture = Path(configured)
    from building_data.geometry import load_vector_house, load_vector_house_scaffold
    from building_data.raster import rasterize_polygon

    from emboss import lidar, mesh
    from emboss.export import write_vector_house_geojson
    from emboss.pipeline import run_emboss_house

    fixture_metadata_path = fixture / "fixture.json"
    fixture_metadata = (
        json.loads(fixture_metadata_path.read_text())
        if fixture_metadata_path.is_file()
        else {}
    )
    fid = int(fixture_metadata.get("building_fid", 2220796))
    point_cloud = fixture / fixture_metadata.get("point_cloud", "points.copc.laz")
    output_root = os.environ.get("EMBOSS_RECONSTRUCTION_OUTPUT_ROOT")
    if output_root:
        tmp_path = Path(output_root) / str(fid)
        tmp_path.mkdir(parents=True, exist_ok=True)
    native_house = load_vector_house(fixture / "surfaces.gpkg", fid)
    scaffold = tmp_path / "scaffold.geojson"
    write_vector_house_geojson(scaffold, native_house)
    house = load_vector_house_scaffold(scaffold, fid)
    assert native_house.roof_envelope.equals_exact(
        house.roof_envelope, 0.0, normalize=True
    )
    assert native_house.bounds_xy == house.bounds_xy
    original_lidar, original_run = _original_core()
    workspace = SimpleNamespace(las_path_or_zip=point_cloud)
    points = original_lidar.read_las_points_for_house(workspace, house)
    expected_observations = original_lidar.prepare_lidar_observations_from_points(
        points, house
    )
    actual_observations = lidar.prepare_lidar_observations_from_points(points, house)
    pd.testing.assert_frame_equal(
        actual_observations, expected_observations, check_exact=True
    )
    rgb = tifffile.imread(fixture / "rgb_corrected.tif")
    classes = tifffile.imread(fixture / "emboss/image_superstructures_class_map.tif")
    metadata = json.loads((fixture / "orthophoto_correction.json").read_text())
    x0, y0, x1, y1 = metadata["extent_lv95"]
    extent = (x0, x1, y0, y1)
    roof_mask = rasterize_polygon(
        house.roof_envelope, extent=extent, width=rgb.shape[1], height=rgb.shape[0]
    )
    kwargs = {
        "workspace_label": "isolated-fixture",
        "house": house,
        "rgb": rgb,
        "image_superstructure_mask": classes != 5,
        "image_superstructure_class_map": classes,
        "image_segmentation_source": "computed",
        "vector_roof_mask": roof_mask,
        "extent_lv95": extent,
    }
    expected = original_run(
        output_dir=tmp_path / "old", observations=expected_observations, **kwargs
    )
    actual = run_emboss_house(
        output_dir=tmp_path / "new", observations=actual_observations, **kwargs
    )
    assert [asdict(solid) for solid in actual.solids] == [
        asdict(solid) for solid in expected.solids
    ]
    old_files = {
        p.relative_to(expected.output_dir)
        for p in expected.output_dir.rglob("*")
        if p.is_file()
    }
    new_files = {
        p.relative_to(actual.output_dir)
        for p in actual.output_dir.rglob("*")
        if p.is_file()
    }
    assert old_files == new_files
    for relative in old_files:
        old, new = expected.output_dir / relative, actual.output_dir / relative
        if old.suffix in {".json", ".geojson"}:
            assert _normalized_payload(
                json.loads(old.read_text()), expected.output_dir
            ) == _normalized_payload(json.loads(new.read_text()), actual.output_dir), (
                str(relative)
            )
        else:
            assert old.read_bytes() == new.read_bytes(), str(relative)
    original_mesh = _original_mesh()
    old_base, new_base = tmp_path / "old-base.ply", tmp_path / "new-base.ply"
    original_mesh.write_swissbuildings3d_mesh(
        surfaces_vector_path=fixture / "surfaces.gpkg",
        building_fid=fid,
        output_path=old_base,
    )
    mesh.write_swissbuildings3d_mesh(
        surfaces_vector_path=fixture / "surfaces.gpkg",
        building_fid=fid,
        output_path=new_base,
    )
    assert old_base.read_bytes() == new_base.read_bytes()
    mesh_kwargs = {
        "superstructures_path": actual.output_dir / "superstructures.geojson",
        "base_mesh_path": new_base,
        "crs": "EPSG:2056",
    }
    original_mesh.write_prediction_mesh_from_case(
        case=SimpleNamespace(roof_scaffold=scaffold, crs="EPSG:2056"),
        output_path=tmp_path / "original.ply",
        **mesh_kwargs,
    )
    stats = mesh.write_prediction_mesh(
        roof_scaffold=scaffold, output_path=tmp_path / "standalone.ply", **mesh_kwargs
    )
    assert (tmp_path / "original.ply").read_bytes() == (
        tmp_path / "standalone.ply"
    ).read_bytes()
    report = {
        "building_fid": fid,
        "points": len(points),
        "observations": len(actual_observations),
        "solids": len(actual.solids),
        "input_class_pixel_counts": {
            str(int(key)): int(count)
            for key, count in zip(*np.unique(classes, return_counts=True), strict=True)
        },
        "fitted_class_labels": [solid.class_label for solid in actual.solids],
        "core_artifacts": len(old_files),
        "core_artifacts_identical_except_output_paths": True,
        "lidar_observations_exact": True,
        "all_solid_coordinates_and_fit_terms_exact": True,
        "base_and_augmented_meshes_byte_identical": True,
        "mesh": stats,
        "scaffold_reload_preserves_envelope_exactly": True,
    }
    report_path = os.environ.get("EMBOSS_RECONSTRUCTION_REPORT")
    if report_path:
        target = Path(report_path)
        if os.environ.get("EMBOSS_RECONSTRUCTION_FIXTURES"):
            target = target.with_name(f"{target.stem}-{fid}{target.suffix}")
        target.write_text(json.dumps(report, indent=2) + "\n")


def test_provider_preserves_duplicate_returns_within_one_source(tmp_path, monkeypatch):
    from building_data import swiss

    points = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.8],
            "y": [0.5, 0.5, 0.8],
            "z": [1.0, 1.0, 2.0],
            "classification": [6, 6, 6],
        }
    )
    provider = swiss.SwissProvider(tmp_path)
    source = {"bounds": [0.0, 0.0, 1.0, 1.0], "tile": "fixture", "path": "unused"}
    monkeypatch.setattr(
        provider, "lidar", lambda *args, **kwargs: ((Path("unused"),), (source,))
    )
    monkeypatch.setattr(swiss, "read_las_points", lambda *args, **kwargs: points.copy())
    actual, _ = provider.points_for_bounds((0.0, 0.0, 1.0, 1.0))
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), points, check_exact=True
    )


def test_imagery_fingerprint_covers_geometry_config_and_is_portable(tmp_path):
    from dataclasses import replace

    from building_data.orthophoto_correction.models import CorrectionConfig

    from emboss.api import _imagery_fingerprint

    item = SimpleNamespace(
        building_fid=1,
        bounds_xy=(0.0, 0.0, 1.0, 1.0),
        sources=({"path": "/old/cache/source.laz", "tile": "tile-2024", "year": 2024},),
    )
    cfg = CorrectionConfig()
    original = _imagery_fingerprint(item, cfg, "original-vector-sha")
    assert _imagery_fingerprint(item, cfg, "changed-vector-sha") != original
    assert (
        _imagery_fingerprint(
            item, replace(cfg, target_gsd_m=0.2), "original-vector-sha"
        )
        != original
    )
    assert (
        _imagery_fingerprint(
            item, replace(cfg, strip_id_override="strip-a"), "original-vector-sha"
        )
        != original
    )
    relocated = SimpleNamespace(
        **{
            **vars(item),
            "sources": ({**item.sources[0], "path": "/new/cache/source.laz"},),
        }
    )
    assert (
        _imagery_fingerprint(
            relocated, replace(cfg, cache_root=tmp_path), "original-vector-sha"
        )
        == original
    )


def test_imagery_state_detects_changed_pixels_metadata_and_identity(tmp_path):
    from emboss.api import _imagery_state

    image = tmp_path / "rgb_corrected.tif"
    metadata = tmp_path / "orthophoto_correction.json"
    tifffile.imwrite(image, np.zeros((4, 4, 3), dtype=np.uint8))
    metadata.write_text(
        json.dumps(
            {"imagery_fingerprint": "matching", "requested_strip_override": None}
        )
    )
    original = _imagery_state(tmp_path, "matching")
    assert original is not None
    assert _imagery_state(tmp_path, "changed-config-or-geometry") is None
    tifffile.imwrite(image, np.ones((4, 4, 3), dtype=np.uint8))
    changed_image = _imagery_state(tmp_path, "matching")
    assert changed_image != original
    metadata.write_text(
        json.dumps(
            {
                "imagery_fingerprint": "matching",
                "requested_strip_override": None,
                "quality": "updated",
            }
        )
    )
    assert _imagery_state(tmp_path, "matching") != changed_image


@pytest.mark.parametrize(
    ("reset", "requested", "expected"),
    [
        (False, None, "saved-strip"),
        (True, None, None),
        (False, "new-strip", "new-strip"),
    ],
)
def test_client_preserves_or_explicitly_resets_strip_override(
    tmp_path, monkeypatch, reset, requested, expected
):
    from building_data.orthophoto_correction.models import CorrectionConfig

    from emboss import api

    root = tmp_path / "data"
    house_root = root / "houses/house-1"
    images = house_root / "imagery"
    images.mkdir(parents=True)
    (images / "orthophoto_correction.json").write_text(
        json.dumps({"requested_strip_override": "saved-strip"})
    )
    surfaces = root / "surfaces.gpkg"
    surfaces.write_bytes(b"geometry-checksum")
    checkpoint = root / "weights.pt"
    checkpoint.write_bytes(b"checkpoint-checksum")
    (house_root / "house.json").write_text("{}")
    item = SimpleNamespace(
        id="house-1",
        root=house_root,
        surfaces_path=surfaces,
        lidar_paths=(),
        building_fid=1,
        sources=(),
        bounds_xy=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:2056",
    )
    client = api.Client(root)
    monkeypatch.setattr(client.store, "house", lambda _: item)
    client.segmentation = SimpleNamespace(checkpoint_path=checkpoint)
    selected = []

    def config(**kwargs):
        selected.append(kwargs["strip_id_override"])
        return CorrectionConfig(**kwargs)

    monkeypatch.setattr(api, "CorrectionConfig", config)

    class StopAfterConfiguration(Exception):
        pass

    def stop(*args):
        raise StopAfterConfiguration

    monkeypatch.setattr(api, "load_vector_house", stop)
    with pytest.raises(StopAfterConfiguration):
        client.reconstruct(
            "house-1", strip_id_override=requested, reset_strip_override=reset
        )
    assert selected == [expected]
