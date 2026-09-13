"""Recover canonical solid provenance without inferring semantics from connectivity.

Replay the same exported solid surfaces and match their geometry to mesh faces.
The matching tolerates vertex welding, face order, winding, and retriangulation.
The source mesh and prediction artifacts are never modified.
"""

from pathlib import Path
import json

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from .mesh import Mesh, _append_superstructure_mesh, _load_roof_surfaces
from .scaffold_geometry import shape_eval_geometry

MATCH_TOLERANCE_M = 1e-5


def _surface_groups(mesh):
    groups = []
    vertices = np.asarray(mesh.vertices, dtype=float)
    for face in mesh.faces:
        for index in range(1, len(face) - 1):
            triangle = vertices[[face[0], face[index], face[index + 1]]]
            cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = np.linalg.norm(cross)
            if length < 1e-12:
                continue
            normal = cross / length
            group = next(
                (
                    group
                    for group in groups
                    if abs(np.dot(group["normal"], normal)) >= 1 - 1e-10
                    and np.max(np.abs((triangle - group["origin"]) @ group["normal"]))
                    <= MATCH_TOLERANCE_M
                ),
                None,
            )
            if group is None:
                group = {
                    "origin": triangle[0],
                    "normal": normal,
                    "axes": np.delete(np.arange(3), np.argmax(np.abs(normal))),
                    "polygons": [],
                }
                groups.append(group)
            xy = (triangle - group["origin"])[:, group["axes"]]
            group["polygons"].append(Polygon(xy))
    for group in groups:
        group["polygon"] = unary_union(group.pop("polygons")).buffer(MATCH_TOLERANCE_M)
    return groups


def classify_faces(
    vertices, faces, details_path: Path, scaffold_path: Path
) -> np.ndarray:
    """Return one semantic kind per triangle from the canonical Emboss artifacts.

    Scaffold roof planes identify eligible base surfaces; unmatched base faces
    remain body/occluder geometry, not inferred roof supports. Exported solids supply their
    semantic label (unlabelled solids are 'other'). Explicit non-PV solids take
    precedence over coincident PV geometry, so PV removal cannot erase a dormer.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError("Expected 3D vertices and triangular faces")
    collection = json.loads(Path(details_path).read_text())
    roof_surfaces = _load_roof_surfaces(Path(scaffold_path))
    if not roof_surfaces:
        raise ValueError("Scaffold has no roof planes for solid provenance")
    labels = np.full(len(faces), "scaffold_body", dtype=object)
    if not len(faces):
        return labels
    triangles = vertices[faces]
    scaffold = json.loads(Path(scaffold_path).read_text())
    roof_features = [
        feature
        for feature in scaffold.get("features", [])
        if feature.get("properties", {}).get("kind") == "roof_face"
    ]
    if not roof_features:
        roof_features = [
            feature
            for feature in scaffold.get("features", [])
            if feature.get("properties", {}).get("kind") == "roof_segment"
        ]
    for feature in roof_features:
        plane = feature["properties"].get("plane_coeffs")
        if plane is None:
            continue
        footprint = shape(feature["geometry"])
        origin_xy = np.array(footprint.representative_point().coords[0][:2])
        a, b, c = plane
        origin = np.r_[origin_xy, a * origin_xy[0] + b * origin_xy[1] + c]
        normal = np.array([-a, -b, 1.0])
        normal /= np.linalg.norm(normal)
        candidates = np.flatnonzero(
            np.max(np.abs((triangles - origin) @ normal), axis=1) <= MATCH_TOLERANCE_M
        )
        region = footprint.buffer(MATCH_TOLERANCE_M)
        for index in candidates:
            polygon = Polygon(triangles[index, :, :2])
            if polygon.area > 1e-12 and region.covers(polygon):
                labels[index] = "scaffold"
    features = sorted(
        collection.get("features", []),
        key=lambda feature: (
            feature.get("properties", {}).get("class_label") != "pvmodule"
        ),
    )
    for feature in features:
        if not feature.get("geometry"):
            continue
        properties = feature.get("properties") or {}
        solid = Mesh(vertices=[], faces=[])
        _append_superstructure_mesh(
            mesh=solid,
            vertex_index={},
            geometry=shape_eval_geometry(feature["geometry"]),
            properties=properties,
            roof_surfaces=roof_surfaces,
            fallback_z=float(vertices[:, 2].min()),
        )
        kind = properties.get("class_label") or "other"
        for group in _surface_groups(solid):
            residuals = (triangles - group["origin"]) @ group["normal"]
            candidates = np.flatnonzero(
                np.max(np.abs(residuals), axis=1) <= MATCH_TOLERANCE_M
            )
            for index in candidates:
                xy = (triangles[index] - group["origin"])[:, group["axes"]]
                polygon = Polygon(xy)
                if polygon.area > 1e-12 and group["polygon"].covers(polygon):
                    labels[index] = kind
    return labels
