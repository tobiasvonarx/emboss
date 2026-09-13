"""Solid provenance stays stable across welding and face triangulation."""

import json
from pathlib import Path

import numpy as np

from emboss.surface_semantics import classify_faces


def artifacts(tmp_path):
    scaffold = tmp_path / "scaffold.geojson"
    scaffold.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "roof_segment",
                            "plane_coeffs": [0, 0, 0],
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
                        },
                    }
                ],
            }
        )
    )
    details = tmp_path / "details.geojson"
    features = []
    for kind, left, right, height in [("dormer", 1, 2, 2), ("pvmodule", 2, 3, 0.2)]:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "solid_id": kind,
                    "class_label": kind,
                    "top_plane": [0, 0, height],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[left, 1], [right, 1], [right, 2], [left, 2], [left, 1]]
                    ],
                },
            }
        )
    details.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return details, scaffold


def test_face_semantics_ignore_vertex_connections(tmp_path):
    details, scaffold = artifacts(tmp_path)
    triangles = np.array(
        [
            [[0, 0, 0], [4, 0, 0], [2, 1, 0]],  # scaffold shares PV/dormer contact
            [[1, 1, 2], [2, 1, 2], [1, 2, 2]],  # opposite diagonal from exporter
            [[2, 1, 2], [2, 2, 2], [1, 2, 2]],
            [[2, 1, 0], [2, 1, 2], [1, 1, 2]],  # dormer wall
            [[2, 1, 0.2], [3, 1, 0.2], [3, 2, 0.2]],
            [
                [2, 1, 0],
                [3, 1, 0],
                [3, 1, 0.2],
            ],  # PV wall; never remove neighbor dormer
        ],
        dtype=float,
    )
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    expected = ["scaffold", "dormer", "dormer", "dormer", "pvmodule", "pvmodule"]
    assert classify_faces(vertices, faces, details, scaffold).tolist() == expected
    welded, inverse = np.unique(vertices, axis=0, return_inverse=True)
    assert (
        classify_faces(welded, inverse[faces], details, scaffold).tolist() == expected
    )
    assert (
        classify_faces(vertices, faces[::-1, ::-1], details, scaffold).tolist()
        == expected[::-1]
    )
