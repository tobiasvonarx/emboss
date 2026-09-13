# ruff: noqa: C408 - preserve original regression fixture expressions
"""Selected unchanged reference-method regression cases, imports adapted to standalone packages."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from building_data.models import RoofFace, RoofSegment, VectorHouse
from emboss.image_superstructures import CLASS_NAMES, image_class_components
from emboss.lidar import (
    CLASS_BUILDING,
    assign_points_to_segments,
    calibrate_residuals,
    residual_thresholds,
)
from emboss.models import ReturnSupportModel
from emboss.roof_superstructures import (
    RoofSuperstructureMaskClient,
    RoofSuperstructureSegmentation,
    RoofSuperstructureSegmentationInput,
    load_or_compute_roof_superstructure_segmentation_many,
)
from emboss.segmentation.schema import RID2_BACKGROUND_CLASS_ID
from emboss.superstructure_fitting import fit_superstructures
from emboss.superstructure_fitting.core import _clip_face_to_roof_parts, _FaceFit
from shapely.geometry import Polygon, box


def _flat_house() -> VectorHouse:
    roof = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    face = RoofFace(
        face_id="face_00",
        building_fid=1,
        polygon_xy=roof,
        points_xyz=tuple(),
        plane_coeffs=(0.0, 0.0, 0.0),
        vertex_rmse_m=0.0,
        area_m2=float(roof.area),
    )
    segment = RoofSegment(
        segment_id="segment_00",
        face_ids=(face.face_id,),
        polygon_xy=roof,
        plane_coeffs=face.plane_coeffs,
        normal=(0.0, 0.0, 1.0),
        area_m2=float(roof.area),
        is_base=True,
    )
    return VectorHouse(
        building_fid=1,
        object_type="building",
        roof_faces=(face,),
        roof_segments=(segment,),
        roof_envelope=roof,
        bounds_xy=(0.0, 0.0, 4.0, 4.0),
        bounds_xyz=(0.0, 0.0, 0.0, 4.0, 4.0, 0.0),
        is_independent=True,
    )


def _drop_house() -> VectorHouse:
    left = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)])
    right = Polygon([(2.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0)])
    faces = (
        RoofFace(
            face_id="face_00",
            building_fid=1,
            polygon_xy=left,
            points_xyz=tuple(),
            plane_coeffs=(0.0, 0.0, 1.0),
            vertex_rmse_m=0.0,
            area_m2=float(left.area),
        ),
        RoofFace(
            face_id="face_01",
            building_fid=1,
            polygon_xy=right,
            points_xyz=tuple(),
            plane_coeffs=(0.0, 0.0, 0.0),
            vertex_rmse_m=0.0,
            area_m2=float(right.area),
        ),
    )
    segments = (
        RoofSegment(
            segment_id="segment_00",
            face_ids=("face_00",),
            polygon_xy=left,
            plane_coeffs=(0.0, 0.0, 1.0),
            normal=(0.0, 0.0, 1.0),
            area_m2=float(left.area),
            is_base=True,
        ),
        RoofSegment(
            segment_id="segment_01",
            face_ids=("face_01",),
            polygon_xy=right,
            plane_coeffs=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            area_m2=float(right.area),
            is_base=True,
        ),
    )
    return VectorHouse(
        building_fid=1,
        object_type="building",
        roof_faces=faces,
        roof_segments=segments,
        roof_envelope=left.union(right),
        bounds_xy=(0.0, 0.0, 4.0, 4.0),
        bounds_xyz=(0.0, 0.0, 0.0, 4.0, 4.0, 1.0),
        is_independent=True,
    )


def _house_with_modeled_details(*footprints: Polygon) -> VectorHouse:
    house = _flat_house()
    details = tuple(
        RoofSegment(
            segment_id=f"detail_{index:02d}",
            face_ids=(),
            polygon_xy=footprint,
            plane_coeffs=(0.0, 0.0, 1.0),
            normal=(0.0, 0.0, 1.0),
            area_m2=float(footprint.area),
            is_base=False,
        )
        for index, footprint in enumerate(footprints)
    )
    return replace(house, roof_segments=(*house.roof_segments, *details))


def _continuous_two_segment_house() -> VectorHouse:
    left = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)])
    right = Polygon([(2.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0)])
    faces = (
        RoofFace(
            face_id="face_00",
            building_fid=1,
            polygon_xy=left,
            points_xyz=tuple(),
            plane_coeffs=(0.0, 0.0, 0.0),
            vertex_rmse_m=0.0,
            area_m2=float(left.area),
        ),
        RoofFace(
            face_id="face_01",
            building_fid=1,
            polygon_xy=right,
            points_xyz=tuple(),
            plane_coeffs=(0.0, 0.0, 0.0),
            vertex_rmse_m=0.0,
            area_m2=float(right.area),
        ),
    )
    segments = (
        RoofSegment(
            segment_id="segment_00",
            face_ids=("face_00",),
            polygon_xy=left,
            plane_coeffs=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            area_m2=float(left.area),
            is_base=True,
        ),
        RoofSegment(
            segment_id="segment_01",
            face_ids=("face_01",),
            polygon_xy=right,
            plane_coeffs=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            area_m2=float(right.area),
            is_base=True,
        ),
    )
    return VectorHouse(
        building_fid=1,
        object_type="building",
        roof_faces=faces,
        roof_segments=segments,
        roof_envelope=left.union(right),
        bounds_xy=(0.0, 0.0, 4.0, 4.0),
        bounds_xyz=(0.0, 0.0, 0.0, 4.0, 4.0, 0.0),
        is_independent=True,
    )


def _observations(residuals: list[float]) -> pd.DataFrame:
    xy = [(1.4, 1.4), (1.8, 1.4), (1.8, 1.8), (1.4, 1.8)]
    return pd.DataFrame(
        [
            {
                "x_cal": x,
                "y_cal": y,
                "classification": CLASS_BUILDING,
                "is_base_segment": True,
                "segment_id": "segment_00",
                "residual_calibrated": residual,
            }
            for (x, y), residual in zip(xy, residuals, strict=True)
        ]
    )


def _raw_lidar_points(samples: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "classification": CLASS_BUILDING,
            }
            for x, y, z in samples
        ]
    )


def _support_model(
    observations: pd.DataFrame,
    *,
    threshold: float = 0.17,
    weight_m2: float = 0.04,
) -> ReturnSupportModel:
    count = len(observations)
    residuals = observations["residual_calibrated"].to_numpy(dtype=np.float64)
    return_xy = observations[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    return ReturnSupportModel(
        roof_band_m=0.08,
        super_threshold_m=float(threshold),
        return_xy=return_xy,
        residuals=residuals,
        return_area_weights_m2=np.full(count, float(weight_m2), dtype=np.float64),
        source_indices=np.asarray(observations.index, dtype=np.int64),
    )


def _return_support(
    xy: list[tuple[float, float]],
    residuals: list[float],
    *,
    roof_band: float = 0.08,
    threshold: float = 0.40,
    weight_m2: float = 0.04,
) -> ReturnSupportModel:
    return_xy = np.asarray(xy, dtype=np.float64)
    residual_array = np.asarray(residuals, dtype=np.float64)
    return ReturnSupportModel(
        roof_band_m=float(roof_band),
        super_threshold_m=float(threshold),
        return_xy=return_xy,
        residuals=residual_array,
        return_area_weights_m2=np.full(len(xy), float(weight_m2), dtype=np.float64),
        source_indices=np.arange(len(xy), dtype=np.int64),
    )


def _class_id(label: str) -> int:
    for class_id, class_label in CLASS_NAMES.items():
        if class_label == label:
            return int(class_id)
    raise AssertionError(f"Unknown image class label: {label}")


def _image_class_map(label: str = "pvmodule") -> np.ndarray:
    class_map = np.full((8, 8), int(RID2_BACKGROUND_CLASS_ID), dtype=np.uint8)
    class_map[4:6, 2:4] = _class_id(label)
    return class_map


def test_segmentation_cache_is_invalidated_when_the_corrected_rgb_changes(
    tmp_path,
) -> None:
    class FakeClient:
        mask_role = "roof"

        def __init__(self) -> None:
            self.calls = 0

        def cache_metadata(self):
            return {"checkpoint_sha256": "test"}

        def predict_hard_segmentation_many(self, items):
            self.calls += 1
            outputs = []
            for rgb, _mask in items:
                class_map = np.full(
                    rgb.shape[:2], int(RID2_BACKGROUND_CLASS_ID), dtype=np.uint8
                )
                outputs.append(
                    RoofSuperstructureSegmentation(
                        class_ids=class_map,
                        foreground=np.zeros_like(class_map, dtype=bool),
                        hard_mask=np.zeros_like(class_map, dtype=bool),
                        foreground_class_map=class_map,
                        hard_class_map=class_map,
                        diagnostics={},
                    )
                )
            return tuple(outputs)

    workspace = type(
        "Workspace",
        (),
        {"corrected_orthophoto_root": tmp_path / "corrected_orthophotos"},
    )()
    roof_mask = np.ones((4, 4), dtype=bool)
    first = RoofSuperstructureSegmentationInput(
        building_fid=1,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        vector_roof_mask=roof_mask,
    )
    changed = RoofSuperstructureSegmentationInput(
        building_fid=1,
        rgb=np.ones((4, 4, 3), dtype=np.uint8),
        vector_roof_mask=roof_mask,
    )
    client = FakeClient()

    load_or_compute_roof_superstructure_segmentation_many(
        workspace=workspace,
        items=(first,),
        client=client,
    )
    cached = load_or_compute_roof_superstructure_segmentation_many(
        workspace=workspace,
        items=(first,),
        client=client,
    )
    refreshed = load_or_compute_roof_superstructure_segmentation_many(
        workspace=workspace,
        items=(changed,),
        client=client,
    )

    assert client.calls == 2
    assert cached[1].source == "cache"
    assert refreshed[1].source == "computed"


def _z_at(plane: tuple[float, float, float], x: float, y: float) -> float:
    return float(plane[0]) * float(x) + float(plane[1]) * float(y) + float(plane[2])


def _rotated_box(
    center: tuple[float, float], width: float, height: float, angle_degrees: float
) -> Polygon:
    angle = math.radians(float(angle_degrees))
    along = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    across = np.asarray([-along[1], along[0]], dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    corners = [
        origin - 0.5 * width * along - 0.5 * height * across,
        origin + 0.5 * width * along - 0.5 * height * across,
        origin + 0.5 * width * along + 0.5 * height * across,
        origin - 0.5 * width * along + 0.5 * height * across,
    ]
    return Polygon([(float(x), float(y)) for x, y in corners])


def _has_edge_parallel_to(polygon: Polygon, axis: np.ndarray) -> bool:
    target = np.asarray(axis, dtype=np.float64)
    target = target / float(np.linalg.norm(target))
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    for left, right in zip(coords, np.roll(coords, -1, axis=0), strict=False):
        edge = right - left
        length = float(np.linalg.norm(edge))
        if length > 1e-9 and abs(float(np.dot(edge / length, target))) > 0.999:
            return True
    return False


def test_lidar_return_thresholds_are_fixed_geometric_bands() -> None:
    observations = _observations([-0.05, 0.02, 0.12, 0.40])

    assert residual_thresholds(observations) == (0.20, 0.40)


def test_lidar_calibration_preserves_planimetric_coordinates_on_sloped_roofs() -> None:
    roof = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    plane = (0.5, -0.25, 10.0)
    face = RoofFace(
        face_id="face_00",
        building_fid=1,
        polygon_xy=roof,
        points_xyz=tuple(),
        plane_coeffs=plane,
        vertex_rmse_m=0.0,
        area_m2=float(roof.area),
    )
    segment = RoofSegment(
        segment_id="segment_00",
        face_ids=(face.face_id,),
        polygon_xy=roof,
        plane_coeffs=plane,
        normal=(-0.5, 0.25, 1.0),
        area_m2=float(roof.area),
        is_base=True,
    )
    house = VectorHouse(
        building_fid=1,
        object_type="building",
        roof_faces=(face,),
        roof_segments=(segment,),
        roof_envelope=roof,
        bounds_xy=(0.0, 0.0, 4.0, 4.0),
        bounds_xyz=(0.0, 0.0, 10.0, 4.0, 4.0, 13.0),
        is_independent=True,
    )
    samples = [
        (
            0.7 + 0.35 * col,
            0.9 + 0.35 * row,
            _z_at(plane, 0.7 + 0.35 * col, 0.9 + 0.35 * row) + 0.12,
        )
        for row in range(2)
        for col in range(4)
    ]
    assigned = assign_points_to_segments(_raw_lidar_points(samples), house)

    calibrated = calibrate_residuals(assigned, house)

    assert np.allclose(
        calibrated["x_cal"].to_numpy(dtype=np.float64),
        assigned["x"].to_numpy(dtype=np.float64),
    )
    assert np.allclose(
        calibrated["y_cal"].to_numpy(dtype=np.float64),
        assigned["y"].to_numpy(dtype=np.float64),
    )
    assert np.allclose(
        calibrated["residual_calibrated"].to_numpy(dtype=np.float64), 0.0
    )


def test_segment_calibration_uses_affine_when_support_is_spatially_broad() -> None:
    house = _flat_house()
    samples = []
    for x in np.linspace(0.5, 3.5, 5):
        for y in np.linspace(0.5, 3.5, 5):
            samples.append(
                (float(x), float(y), 0.05 * float(x) + 0.04 * float(y) + 0.05)
            )
    assigned = assign_points_to_segments(_raw_lidar_points(samples), house)

    calibrated = calibrate_residuals(assigned, house)

    assert set(calibrated["calibration_method"]) == {"segment_mode_affine"}
    raw_std = float(np.std(assigned["residual_signed"].to_numpy(dtype=np.float64)))
    calibrated_std = float(
        np.std(calibrated["residual_calibrated"].to_numpy(dtype=np.float64))
    )
    assert calibrated_std < 0.5 * raw_std


def test_fit_superstructures_uses_lidar_residuals_for_geometry() -> None:
    house = _flat_house()
    observations = _observations([0.35, 0.38, 0.36, 0.34])
    solids = fit_superstructures(observations, house, _support_model(observations))

    assert len(solids) == 1
    solid = solids[0]
    assert solid.solid_id == "solid_000"
    assert solid.source == "lidar"
    assert solid.point_count == 4
    assert solid.height_offset_m > 0.30
    assert solid.fit_terms["support_weight_m2"] > 0.0
    assert solid.fit_terms["top_face_count"] == 1.0


def test_lidar_only_singleton_superstructure_is_rejected() -> None:
    house = _flat_house()
    observations = pd.DataFrame(
        [
            {
                "x_cal": 1.5,
                "y_cal": 1.5,
                "classification": CLASS_BUILDING,
                "is_base_segment": True,
                "segment_id": "segment_00",
                "residual_calibrated": 0.42,
            },
        ]
    )

    solids = fit_superstructures(
        observations,
        house,
        _support_model(observations, threshold=0.40, weight_m2=0.061),
    )

    assert solids == ()


def test_image_components_keep_corner_touching_classes_separate() -> None:
    class_map = np.full((4, 4), int(RID2_BACKGROUND_CLASS_ID), dtype=np.uint8)
    class_map[1, 1] = _class_id("pvmodule")
    class_map[2, 2] = _class_id("window")

    components = image_class_components(
        class_map=class_map,
        extent_lv95=(0.0, 0.4, 0.0, 0.4),
    )

    assert len(components) == 2
    assert {component.class_label for component in components} == {"pvmodule", "window"}
    assert all(
        component.geometry.area == pytest.approx(0.01) for component in components
    )


def test_image_components_keep_touching_classes_separate() -> None:
    class_map = np.full((8, 8), int(RID2_BACKGROUND_CLASS_ID), dtype=np.uint8)
    class_map[2:6, 2:5] = _class_id("dormer")
    class_map[3:5, 5:7] = _class_id("other")

    components = image_class_components(
        class_map=class_map,
        extent_lv95=(0.0, 0.8, 0.0, 0.8),
    )

    assert len(components) == 2
    assert {component.class_label for component in components} == {"dormer", "other"}
    assert sorted(component.geometry.area for component in components) == pytest.approx(
        [0.04, 0.12]
    )


def test_segmentation_keeps_single_foreground_pixel() -> None:
    class_ids = np.full((4, 4), int(RID2_BACKGROUND_CLASS_ID), dtype=np.uint8)
    class_ids[1, 2] = _class_id("other")

    segmentation = RoofSuperstructureMaskClient()._finalize_segmentation(
        class_ids,
        class_ids != int(RID2_BACKGROUND_CLASS_ID),
        {},
    )

    assert (
        np.count_nonzero(
            segmentation.foreground_class_map != int(RID2_BACKGROUND_CLASS_ID)
        )
        == 1
    )
    assert segmentation.foreground_class_map[1, 2] == _class_id("other")


def test_image_prior_candidate_can_fill_lidar_coverage_gap() -> None:
    house = _flat_house()
    observations = pd.DataFrame(
        columns=[
            "x_cal",
            "y_cal",
            "classification",
            "is_base_segment",
            "segment_id",
            "residual_calibrated",
        ]
    )

    solids = fit_superstructures(
        observations,
        house,
        _support_model(observations),
        image_class_map=_image_class_map("window"),
        image_extent_lv95=(0.0, 4.0, 0.0, 4.0),
    )

    assert len(solids) == 1
    assert solids[0].source == "image_segmentation"
    assert solids[0].class_label == "window"
    assert solids[0].fit_terms["observed_support_weight_m2"] == 0.0
    assert solids[0].fit_terms["image_overlap_component_fraction"] > 0.0


def test_superstructures_can_cross_continuous_segment_boundary() -> None:
    house = _continuous_two_segment_house()
    observations = pd.DataFrame(
        [
            {
                "x_cal": x,
                "y_cal": y,
                "classification": CLASS_BUILDING,
                "is_base_segment": True,
                "segment_id": segment_id,
                "residual_calibrated": 0.55,
                "z_cal": 0.55,
            }
            for x, y, segment_id in (
                (1.80, 1.40, "segment_00"),
                (1.80, 1.80, "segment_00"),
                (2.20, 1.40, "segment_01"),
                (2.20, 1.80, "segment_01"),
            )
        ]
    )

    solids = fit_superstructures(observations, house, _support_model(observations))

    assert len(solids) == 1
    assert min(x for x, _ in solids[0].footprint_xy) < 2.0
    assert max(x for x, _ in solids[0].footprint_xy) > 2.0
    assert solids[0].host_segment_ids == ("segment_00", "segment_01")


def test_superstructures_do_not_cross_discontinuous_boundary() -> None:
    house = _drop_house()
    observations = pd.DataFrame(
        [
            {
                "x_cal": x,
                "y_cal": y,
                "classification": CLASS_BUILDING,
                "is_base_segment": True,
                "segment_id": segment_id,
                "residual_calibrated": residual,
                "z_cal": 1.55,
            }
            for x, y, segment_id, residual in (
                (1.55, 1.40, "segment_00", 0.55),
                (1.55, 1.80, "segment_00", 0.55),
                (1.95, 1.40, "segment_00", 0.55),
                (1.95, 1.80, "segment_00", 0.55),
                (2.55, 1.40, "segment_01", 1.55),
                (2.55, 1.80, "segment_01", 1.55),
                (3.15, 1.60, "segment_01", 1.55),
            )
        ]
    )

    solids = fit_superstructures(observations, house, _support_model(observations))

    assert len(solids) == 2
    assert {solid.host_segment_ids for solid in solids} == {
        ("segment_00",),
        ("segment_01",),
    }
    for solid in solids:
        xs = [x for x, _y in solid.footprint_xy]
        assert max(xs) <= 2.0 + 1e-6 or min(xs) >= 2.0 - 1e-6


def test_image_component_matching_modeled_chimney_is_rejected() -> None:
    house = _house_with_modeled_details(box(1.1, 2.1, 1.9, 2.9))
    image_class_map = np.full((40, 40), int(RID2_BACKGROUND_CLASS_ID), dtype=np.uint8)
    image_class_map[10:20, 10:20] = _class_id("other")
    observations = pd.DataFrame(
        columns=[
            "x_cal",
            "y_cal",
            "classification",
            "is_base_segment",
            "segment_id",
            "residual_calibrated",
        ]
    )

    solids = fit_superstructures(
        observations,
        house,
        _support_model(observations),
        image_class_map=image_class_map,
        image_extent_lv95=(0.0, 4.0, 0.0, 4.0),
    )

    assert solids == ()


def test_top_face_is_clipped_when_it_enters_the_roof_scaffold() -> None:
    house = _drop_house()
    face = _FaceFit(
        polygon=box(1.50, 1.00, 2.50, 2.00),
        plane=(0.0, 0.0, 0.55),
        height_m=0.55,
        height_model="constant_z",
    )

    clipped_parts = _clip_face_to_roof_parts(face, house)

    assert len(clipped_parts) == 1
    clipped = clipped_parts[0]
    assert clipped is not None
    assert np.isclose(float(clipped.polygon.area), 0.50)
    assert min(x for x, _ in clipped.polygon.exterior.coords[:-1]) >= 2.0


def test_top_face_clip_preserves_disconnected_visible_parts() -> None:
    roof = Polygon(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 3.0),
            (3.0, 3.0),
            (3.0, 1.0),
            (0.0, 1.0),
        ]
    )
    roof_face = RoofFace(
        face_id="face_00",
        building_fid=1,
        polygon_xy=roof,
        points_xyz=tuple(),
        plane_coeffs=(1.0, 0.0, 0.0),
        vertex_rmse_m=0.0,
        area_m2=float(roof.area),
    )
    segment = RoofSegment(
        segment_id="segment_00",
        face_ids=(roof_face.face_id,),
        polygon_xy=roof,
        plane_coeffs=roof_face.plane_coeffs,
        normal=(0.0, 0.0, 1.0),
        area_m2=float(roof.area),
        is_base=True,
    )
    house = VectorHouse(
        building_fid=1,
        object_type="building",
        roof_faces=(roof_face,),
        roof_segments=(segment,),
        roof_envelope=roof,
        bounds_xy=(0.0, 0.0, 4.0, 4.0),
        bounds_xyz=(0.0, 0.0, 0.0, 4.0, 4.0, 4.0),
        is_independent=True,
    )
    face = _FaceFit(
        polygon=roof,
        plane=(0.0, 0.0, 2.0),
        height_m=2.0,
        height_model="constant_z",
    )

    parts = _clip_face_to_roof_parts(face, house)

    assert len(parts) == 2
    assert [round(float(part.polygon.area), 6) for part in parts] == [2.0, 2.0]


def test_original_fitter_produces_identical_solids_and_fit_terms():
    """Compare every fitted coordinate and term with the isolated original fitter."""
    import os
    from dataclasses import asdict
    from pathlib import Path

    import pytest

    reference_root = os.environ.get("EMBOSS_REFERENCE_ROOT")
    if not reference_root:
        pytest.skip("Set EMBOSS_REFERENCE_ROOT for direct original-method comparison.")
    from test_segmentation_parity import _reference_module

    reference = _reference_module(
        Path(reference_root) / "src/emboss/superstructure_fitting/core.py",
        "emboss.superstructure_fitting",
        (("from ..geometry import", "from building_data.geometry import"),),
    )
    for residuals in (
        [0.35, 0.38, 0.36, 0.34],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.1, 1.2, 1.3],
    ):
        house = _flat_house()
        observations = _observations(residuals)
        support = _support_model(observations)
        expected = reference.fit_superstructures(observations, house, support)
        actual = fit_superstructures(observations, house, support)
        assert [asdict(solid) for solid in actual] == [
            asdict(solid) for solid in expected
        ]


def test_correction_preparation_preserves_tile_grid_and_pins_assets(
    tmp_path, monkeypatch
):

    from building_data.orthophoto_correction import prepare
    from building_data.orthophoto_correction.models import (
        CorrectionConfig,
        RasterAsset,
        StripCandidate,
    )
    from building_data.orthophoto_correction.raster import (
        read_geotiff_array,
        read_raster_crop,
        write_geotiff,
    )

    cfg = CorrectionConfig(cache_root=tmp_path / "queries")
    bounds = (2600000.0, 1200000.0, 2600002.0, 1200002.0)
    expanded = prepare._patch_bounds(bounds, None, cfg)
    remote = {}
    for key, gsd in (("swissimage", 0.1), ("surface", 0.5), ("terrain", 0.5)):
        width = round((expanded[2] - expanded[0]) / gsd)
        pixels = np.indices((width, width)).sum(axis=0)
        array = (
            np.repeat((pixels % 256).astype(np.uint8)[..., None], 3, axis=2)
            if key == "swissimage"
            else pixels.astype(np.float32)
        )
        path = tmp_path / f"source-{key}.tif"
        write_geotiff(path, array, bounds_lv95=expanded)
        remote[key] = [RasterAsset(key, f"{key}-2020", key, str(path), gsd, 2020)]
    strip = StripCandidate(
        "strip-2020",
        2020,
        "2020-01-01",
        0.1,
        "SWISSIMAGE",
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    monkeypatch.setattr(
        prepare, "resolve_correction_assets", lambda *args, **kwargs: remote
    )
    monkeypatch.setattr(prepare, "query_lubis_strips", lambda *args, **kwargs: (strip,))
    monkeypatch.setattr(
        prepare, "filter_swissimage_strips", lambda strips, **kwargs: strips
    )
    local = prepare.prepare_assets(
        tile_key="swisssurface3d_2020_2600-1200",
        bounds_lv95=bounds,
        cache=tmp_path / "prepared",
        config=cfg,
        progress=lambda _: None,
    )
    for key, assets in local.items():
        expected, extent = read_raster_crop(
            remote[key],
            bounds_lv95=expanded,
            gsd_m=assets[0].gsd_m,
            resample_alg="bilinear",
        )
        np.testing.assert_array_equal(
            read_geotiff_array(__import__("pathlib").Path(assets[0].href)), expected
        )
        assert assets[0].metadata["extent_lv95"] == list(extent)
    monkeypatch.setattr(
        prepare,
        "resolve_correction_assets",
        lambda *args, **kwargs: pytest.fail(
            "cached selection must not query newer sources"
        ),
    )
    cached = prepare.prepare_assets(
        tile_key="swisssurface3d_2020_2600-1200",
        bounds_lv95=bounds,
        cache=tmp_path / "prepared",
        config=cfg,
        progress=lambda _: None,
    )
    assert cached == local
    # Move the cache: local manifest paths must remain usable.
    moved = tmp_path / "moved"
    (tmp_path / "prepared").rename(moved)
    moved_assets = prepare.prepare_assets(
        tile_key="swisssurface3d_2020_2600-1200",
        bounds_lv95=bounds,
        cache=moved,
        config=cfg,
        progress=lambda _: None,
    )
    assert __import__("pathlib").Path(moved_assets["swissimage"][0].href).is_file()
    assert moved_assets["swissimage"][0].href != local["swissimage"][0].href
    # Cross-boundary requests preserve both original pixel grids.
    extended = prepare._patch_bounds(
        bounds, (expanded[0] - 0.21, expanded[1], expanded[2] + 0.72, expanded[3]), cfg
    )
    assert extended[0] == expanded[0] - 0.5
    assert extended[2] == expanded[2] + 1.0
