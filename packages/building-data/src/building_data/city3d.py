"""Optional City3D process adapter; no dependency on application or thesis code."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import shapely
from filelock import FileLock

from .storage import write_json


def scaffold_surfaces(provider, row):
    directory = provider.cache / "scaffolds" / provider.revision / str(row.building_id)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "surfaces.json"
    with FileLock(str(directory / "prepare.lock")):
        if output.exists():
            return json.loads(output.read_text())
        binary = provider.binary
        if not binary or not Path(binary).is_file():
            raise ValueError(
                "Set CITY3D_BINARY to the installed City3D CLI_Emboss executable"
            )
        polygon = row.geometry
        if polygon.geom_type != "Polygon" or polygon.interiors:
            raise ValueError(
                "This City3D adapter requires a single footprint without courtyard holes"
            )
        x0, y0, x1, y1 = polygon.bounds
        points, _ = provider.points_for_bounds((x0 - 3, y0 - 3, x1 + 3, y1 + 3))
        ground = points[points.classification.eq(2)]
        roof = points[
            points.classification.eq(6)
            & shapely.intersects_xy(polygon, points.x, points.y)
        ]
        if len(roof) < 40 or ground.empty:
            raise ValueError(
                f"Insufficient City3D input: {len(roof)} building returns and {len(ground)} ground returns"
            )
        origin = np.array(
            [polygon.centroid.x, polygon.centroid.y, float(ground.z.median())]
        )
        xyz = roof[["x", "y", "z"]].to_numpy() - origin
        ply = directory / "points.ply"
        with ply.open("w") as stream:
            stream.write(
                f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
            )
            np.savetxt(stream, xyz, fmt="%.6f")
        footprint = directory / "footprint.obj"
        xy = np.asarray(polygon.exterior.coords[:-1])[:, :2] - origin[:2]
        with footprint.open("w") as stream:
            for x, y in xy:
                stream.write(f"v {x:.6f} {y:.6f} 0\n")
            stream.write("f " + " ".join(str(i + 1) for i in range(len(xy))) + "\n")
        mesh = directory / "scaffold.obj"
        command = [
            str(binary),
            "--point-cloud",
            str(ply),
            "--footprint",
            str(footprint),
            "--output",
            str(mesh),
            "--ground-z",
            "0",
        ]
        with (directory / "city3d.log").open("w") as log:
            process = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
            )
        if process.returncode or not mesh.exists():
            raise RuntimeError(
                f"City3D could not reconstruct {row.toid}; see {directory / 'city3d.log'}"
            )
        vertices = []
        surfaces = {"Roof": [], "Wall": [], "Floor": []}
        for line in mesh.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append(np.array(list(map(float, fields[1:4]))) + origin)
            elif fields[0] == "f":
                indices = [int(s.split("/")[0]) for s in fields[1:]]
                coords = np.array([vertices[i - 1 if i > 0 else i] for i in indices])
                normal = np.sum(
                    np.cross(coords - origin, np.roll(coords - origin, -1, axis=0)),
                    axis=0,
                )
                norm = np.linalg.norm(normal)
                if norm < 1e-8:
                    continue
                if abs(normal[2]) / norm < 0.3:
                    kind = "Wall"
                else:
                    kind = "Floor" if np.max(coords[:, 2]) - origin[2] < 0.1 else "Roof"
                surfaces[kind].append(coords.tolist())
        if not surfaces["Roof"]:
            raise RuntimeError("City3D output contains no usable roof planes")
        write_json(
            directory / "input.json",
            {
                "origin_xyz": origin.tolist(),
                "toid": row.toid,
                "source_revision": provider.revision,
                "building_returns": len(roof),
                "command": command,
            },
        )
        write_json(output, surfaces)
        return surfaces
