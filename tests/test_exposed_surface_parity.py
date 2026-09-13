"""Independent upper-surface regressions; optional original/real-data parity.

EXPOSED_REAL_RESULT points to an acquired Emboss result directory. Set
EXPOSED_ORIGINAL_ROOT to an explicitly authorized original checkout or reference
copy to compare its shared numerical module. Neither input is modified.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

from emboss.roof_surface import exposed_roof


def stacked_roof(*, shared_vertex=False, reverse=False):
    """Same surfaces, with a dormer joined to the roof at one vertex or detached."""
    vertices = np.array([
        [0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0],
        [0, 0, -3], [8, 0, -3], [8, 8, -3], [0, 8, -3],
        [2, 2, 0], [6, 2, 0], [6, 5, 0], [2, 5, 0],
        [2, 2, 2], [6, 2, 2], [6, 5, 2], [2, 5, 2],
    ], dtype=float)
    roof = [[0, 1, 8], [1, 2, 8], [2, 3, 8], [3, 0, 8]] if shared_vertex else [[0, 1, 2], [0, 2, 3]]
    faces = roof + [[4, 6, 5], [4, 7, 6]]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        faces.extend([[a, a + 4, b + 4], [a, b + 4, b]])
    faces += [[12, 13, 14], [12, 14, 15]]
    for a, b in [(8, 9), (9, 10), (10, 11), (11, 8)]:
        faces.extend([[a, b, b + 4], [a, b + 4, a + 4]])
    triangles = np.asarray(faces, dtype=np.int64)
    if reverse:
        triangles = triangles[::-1, ::-1]
    return vertices, triangles


def projected(triangles):
    return unary_union([Polygon(triangle[:, :2]) for triangle in triangles])


def height_layers(triangles):
    layers = {}
    for triangle in triangles:
        assert np.ptp(triangle[:, 2]) < 1e-10
        layers.setdefault(round(float(triangle[0, 2]), 8), []).append(Polygon(triangle[:, :2]))
    return {z: unary_union(polygons) for z, polygons in layers.items()}


@pytest.mark.parametrize('shared_vertex,reverse', [(False, False), (True, False), (False, True), (True, True)])
def test_identical_surfaces_keep_dormer_top_and_remove_covered_base(shared_vertex, reverse):
    vertices, faces = stacked_roof(shared_vertex=shared_vertex, reverse=reverse)
    result = exposed_roof(vertices, faces)
    layers = height_layers(result.triangles)
    expected_top = box(2, 2, 6, 5)
    assert set(layers) == {0.0, 2.0}
    assert layers[2.0].symmetric_difference(expected_top).area < 1e-10
    assert layers[0.0].symmetric_difference(box(0, 0, 8, 8).difference(expected_top)).area < 1e-10
    assert projected(result.triangles).area == pytest.approx(64.0)
    assert sum(Polygon(t[:, :2]).area for t in result.triangles) == pytest.approx(64.0)


def original_core():
    root = os.environ.get('EXPOSED_ORIGINAL_ROOT')
    if not root:
        pytest.skip('Set EXPOSED_ORIGINAL_ROOT for original/package numerical comparison')
    path = Path(root) / 'src/emboss/roof_surface.py'
    spec = importlib.util.spec_from_file_location('_exposed_original_core', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Execute from source text to keep the read-only original tree free of pyc writes.
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def test_shared_core_original_package_arrays_identical():
    original = original_core()
    for welded in (False, True):
        vertices, faces = stacked_roof(shared_vertex=welded)
        vertices += np.array([2668450.0, 1206400.0, 450.0])
        expected = original.exposed_roof(vertices, faces)
        actual = exposed_roof(vertices, faces)
        np.testing.assert_array_equal(actual.triangles, expected.triangles)
        np.testing.assert_array_equal(actual.source_face_ids, expected.source_face_ids)


def real_result():
    root = os.environ.get('EXPOSED_REAL_RESULT')
    if not root:
        pytest.skip('Set EXPOSED_REAL_RESULT for the real building regression')
    return Path(root)


def read_triangles(path):
    lines = path.read_text().splitlines()
    end = lines.index('end_header')
    vertex_count = int(next(line.split()[-1] for line in lines[:end] if line.startswith('element vertex ')))
    face_count = int(next(line.split()[-1] for line in lines[:end] if line.startswith('element face ')))
    vertices = np.array([[float(v) for v in line.split()[:3]] for line in lines[end + 1:end + 1 + vertex_count]])
    faces = []
    for line in lines[end + 1 + vertex_count:end + 1 + vertex_count + face_count]:
        values = [int(v) for v in line.split()]
        ids = values[1:1 + values[0]]
        faces.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1))
    return vertices, np.array(faces, dtype=np.int64)


def test_real_dormer_is_exposed_with_no_underlying_surface():
    root = real_result()
    vertices, faces = read_triangles(root / 'mesh.ply')
    exposed = exposed_roof(vertices, faces)
    features = json.loads((root / 'roof_details.geojson').read_text())['features']
    tested = 0
    for feature in features:
        props = feature['properties']
        if props.get('class_label') != 'dormer':
            continue
        plane = np.asarray(props['top_plane'])
        footprint = shape(feature['geometry'])
        top_parts = []
        for triangle in exposed.triangles:
            overlap = Polygon(triangle[:, :2]).intersection(footprint)
            # GeoJSON coordinates are rounded separately from mesh doubles.
            # Ignore sub-square-millimeter boundary slivers between the exports.
            if overlap.area < 1e-6:
                continue
            origin = triangle[0]
            slope = np.linalg.solve(triangle[1:, :2] - origin[:2], triangle[1:, 2] - origin[2])
            xy = shapely.get_coordinates(overlap)
            surface_z = (xy - origin[:2]) @ slope + origin[2]
            dormer_z = xy @ plane[:2] + plane[2]
            assert float(np.min(surface_z - dormer_z)) >= -1e-6
            if np.max(np.abs(surface_z - dormer_z)) < 1e-6:
                top_parts.append(overlap)
        assert unary_union(top_parts).area >= footprint.area - 1e-5
        tested += 1
    assert tested >= 2


def test_real_original_package_arrays_identical():
    root = real_result()
    original = original_core()
    vertices, faces = read_triangles(root / 'mesh.ply')
    expected = original.exposed_roof(vertices, faces)
    actual = exposed_roof(vertices, faces)
    np.testing.assert_array_equal(actual.triangles, expected.triangles)
    np.testing.assert_array_equal(actual.source_face_ids, expected.source_face_ids)
