import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from emboss.roof_surface import covered_footprint, exposed_roof, uppermost_pieces


def square(x0=0, y0=0, size=2, z=0):
    return np.array(
        [
            [x0, y0, z],
            [x0 + size, y0, z],
            [x0 + size, y0 + size, z],
            [x0, y0 + size, z],
        ],
        dtype=float,
    )


FACES = np.array([[0, 1, 2], [0, 2, 3]])


def area(triangles):
    return float(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        ).sum()
        / 2
    )


def test_layered_roofs_keep_usable_upper_and_remove_hidden_lower():
    vertices = np.vstack((square(size=4), square(1, 1, 2, 1)))
    faces = np.vstack((FACES, FACES + 4))
    result = exposed_roof(vertices, faces)
    lower = result.triangles[result.source_face_ids < 2]
    upper = result.triangles[result.source_face_ids >= 2]
    assert area(lower) == pytest.approx(12)
    assert area(upper) == pytest.approx(4)
    assert unary_union([Polygon(t[:, :2]) for t in lower]).intersection(
        box(1, 1, 3, 3)
    ).area == pytest.approx(0)
    assert np.array_equal(vertices, np.vstack((square(size=4), square(1, 1, 2, 1))))


def test_single_vertex_welding_does_not_change_exposed_geometry():
    base = square(size=4)
    detail = np.array([[0, 0, 0], [2, 0, 1], [2, 2, 1], [0, 2, 1]], float)
    separate_vertices = np.vstack((base, detail))
    separate_faces = np.vstack((FACES, FACES + 4))
    welded_faces = separate_faces.copy()
    welded_faces[welded_faces == 4] = 0
    original = exposed_roof(separate_vertices, separate_faces)
    welded = exposed_roof(separate_vertices, welded_faces)
    assert np.array_equal(original.triangles, welded.triangles)
    assert np.array_equal(original.source_face_ids, welded.source_face_ids)


def test_crossing_planes_split_at_height_equality():
    first = square(size=2)
    first[:, 2] = first[:, 0]
    second = square(size=2)
    second[:, 2] = 2 - second[:, 0]
    result = exposed_roof(np.vstack((first, second)), np.vstack((FACES, FACES + 4)))
    assert area(result.triangles) == pytest.approx(4 * np.sqrt(2))
    assert np.allclose(
        result.triangles[:, :, 2],
        np.maximum(result.triangles[:, :, 0], 2 - result.triangles[:, :, 0]),
    )
    assert np.any(np.isclose(result.triangles[:, :, 0], 1))


def test_large_world_coordinates_preserve_local_exposed_surface():
    vertices = np.vstack((square(size=4), square(1, 1, 2, 1)))
    vertices[:, 2] += vertices[:, 0] * 0.2 - vertices[:, 1] * 0.1
    faces = np.vstack((FACES, FACES + 4))
    origin = np.array([2_668_461.602, 1_206_408.452, 448.19])
    local = exposed_roof(vertices, faces)
    world = exposed_roof(vertices + origin, faces)
    assert area(world.triangles) == pytest.approx(area(local.triangles), abs=1e-7)
    for triangle, source_id in zip(
        world.triangles - origin, world.source_face_ids, strict=True
    ):
        expected_z = (
            triangle[:, 0] * 0.2 - triangle[:, 1] * 0.1 + (1 if source_id >= 2 else 0)
        )
        assert np.allclose(triangle[:, 2], expected_z, atol=1e-8)
    assert np.array_equal(exposed_roof(vertices, faces).triangles, local.triangles)


def test_polygon_core_preserves_holes_and_source_ties():
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]]
    )
    pieces = uppermost_pieces(
        [polygon, polygon], [(0, 0, 1), (0, 0, 1)], source_ids=[5, 2]
    )
    assert sum(piece.polygon_xy.area for piece in pieces) == pytest.approx(12)
    assert all(piece.source_index == 1 for piece in pieces)


def test_height_aware_coverage_preserves_upper_roof_and_crossing_parts():
    triangle = np.array([[[0, 0, -1], [2, 0, 1], [2, 2, 1]]], float)
    cover = covered_footprint(
        box(0, 0, 2, 2), [0, 0, 0], [0, 0, 1], triangle, clearance_m=0
    )
    assert cover.area == pytest.approx(1.5, abs=1e-7)
    upper = square(size=2, z=2)[FACES]
    assert covered_footprint(
        box(0, 0, 2, 2), [0, 0, 0], [0, 0, 1], upper
    ).area == pytest.approx(4)
    assert covered_footprint(box(0, 0, 2, 2), [0, 0, 2], [0, 0, 1], upper).is_empty


def test_elevated_vertical_wall_retains_clearance_line():
    wall = np.array([[[1, 0, 0], [1, 2, 0], [1, 2, 2]]], float)
    cover = covered_footprint(box(0, 0, 2, 2), [0, 0, 0], [0, 0, 1], wall)
    assert cover.area == 0
    assert cover.length == pytest.approx(2, abs=1e-6)
    assert cover.buffer(0.2).area > 0.7
    assert covered_footprint(box(0, 0, 2, 2), [0, 0, 3], [0, 0, 1], wall).is_empty


def test_height_aware_coverage_is_translation_invariant():
    from shapely.affinity import translate

    origin = np.array([2_668_461.602, 1_206_408.452, 448.19])
    triangles = np.array([[[0, 0, -1], [2, 0, 1], [2, 2, 1]]], float)
    polygon = box(0, 0, 2, 2)
    local = covered_footprint(polygon, [0, 0, 0], [-0.2, 0.1, 1], triangles)
    world = covered_footprint(
        translate(polygon, xoff=origin[0], yoff=origin[1]),
        origin,
        [-0.2, 0.1, 1],
        triangles + origin,
    )
    restored = translate(world, xoff=-origin[0], yoff=-origin[1])
    assert local.symmetric_difference(restored).area < 1e-8


def test_empty_and_invalid_input_contract():
    assert exposed_roof(
        np.empty((0, 3)), np.empty((0, 3), dtype=int)
    ).triangles.shape == (0, 3, 3)
    with pytest.raises(ValueError, match="index"):
        exposed_roof(square(), np.array([[0, 1, 9]]))
