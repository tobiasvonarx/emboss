"""Source target identity, scope and topology, independent of roof reconstruction.

An optional Target layer travels with the surfaces GeoPackage. It records the
selected source footprints even when the roof solver changes their boundaries.
Providers without this layer retain the existing geometry-derived topology.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyogrio
from shapely.geometry import mapping


def read_targets(path: Path) -> dict[int, dict]:
    if not Path(path).is_file():
        return {}
    try:
        layers = {name for name, _ in pyogrio.list_layers(path)}
    except pyogrio.errors.DataSourceError:
        # Non-GDAL providers may supply their own vector loader.
        return {}
    if "Target" not in layers:
        return {}
    frame = pyogrio.read_dataframe(path, layer="Target", fid_as_index=True)
    return {
        int(fid): {
            "scope": row.scope,
            "source_ids": json.loads(row.source_ids),
            "source_footprint": mapping(row.geometry),
            "source_area_m2": float(row.geometry.area),
            "touching_building_fids": json.loads(row.touching_ids),
            "coverage": json.loads(row.coverage),
        }
        for fid, row in frame.iterrows()
    }


def source_topology(index, path: Path):
    """Prefer source neighbours over topology inferred from successful roofs."""
    for fid, target in read_targets(path).items():
        mask = index.building_fid.eq(fid)
        neighbours = target["touching_building_fids"]
        index.loc[mask, "is_independent"] = not neighbours
        index.loc[mask, "touching_house_count"] = len(neighbours)
        index.loc[mask, "touching_building_fids"] = ";".join(map(str, neighbours))
    return index
