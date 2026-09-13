"""Readers for fetched Emboss acquisition workspaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .models import OrthophotoCrop


@dataclass(frozen=True)
class EmbossWorkspace:
    workspace_root: Path
    tile_key: str
    surfaces_vector_path: Path
    las_path_or_zip: Path
    tmp_dir: Path
    tile_bounds_xy: tuple[float, float, float, float]
    manifest: dict[str, Any]

    @property
    def corrected_orthophoto_root(self) -> Path:
        return self.workspace_root / "corrected_orthophotos"


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def normalize_lv95_extent_for_emboss(values: tuple[float, ...] | list[float]) -> tuple[float, float, float, float]:
    """Convert stored bounds order to Emboss raster extent order.

    Corrected orthophoto metadata stores LV95 extents as
    (min_x, min_y, max_x, max_y). Emboss raster helpers use
    (min_x, max_x, min_y, max_y).
    """

    if len(values) != 4:
        raise ValueError(f"Expected four LV95 extent values, got {len(values)}")
    min_x, min_y, max_x, max_y = (float(value) for value in values)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(f"Invalid LV95 extent bounds: {tuple(values)}")
    return (min_x, max_x, min_y, max_y)


def load_workspace(path: str | Path) -> EmbossWorkspace:
    """Load a fetched workspace from `workspace.json` or the tile directory."""

    input_path = Path(path)
    manifest_path = input_path / "workspace.json" if input_path.is_dir() else input_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"No workspace manifest found at {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = _resolve_path(payload.get("workspace_root", manifest_path.parent), base=manifest_path.parent)
    raw_asset_paths = dict(payload.get("raw_asset_paths", {}))
    las_value = raw_asset_paths.get("surfacePointCloud") or raw_asset_paths.get("surface") or ""
    if not las_value:
        candidate = next(root.glob("tmp/*.las"), None)
        if candidate is None:
            raise RuntimeError("Workspace manifest does not contain a surfacePointCloud asset")
        las_path = candidate
    else:
        las_path = _resolve_path(las_value, base=root)
    tile_record = dict(payload.get("tile_record", {}))
    tile_bounds = (
        float(tile_record.get("tile_min_x", 0.0)),
        float(tile_record.get("tile_min_y", 0.0)),
        float(tile_record.get("tile_max_x", 0.0)),
        float(tile_record.get("tile_max_y", 0.0)),
    )
    return EmbossWorkspace(
        workspace_root=root,
        tile_key=str(payload.get("tile_key", root.name)),
        surfaces_vector_path=_resolve_path(payload.get("surfaces_vector_path", root / "surfaces.gpkg"), base=root),
        las_path_or_zip=las_path,
        tmp_dir=_resolve_path(payload.get("tmp_dir", root / "tmp"), base=root),
        tile_bounds_xy=tile_bounds,
        manifest=payload,
    )


def load_orthophoto_crop(workspace: EmbossWorkspace, building_fid: int) -> OrthophotoCrop:
    """Load the already corrected/nadir orthophoto crop for one house."""

    crop_dir = workspace.corrected_orthophoto_root / f"fid_{int(building_fid)}"
    metadata_path = crop_dir / "orthophoto_correction.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing corrected orthophoto metadata for fid={building_fid}: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = dict(metadata.get("outputs", {}))
    rgb_path = Path(outputs.get("rgb_corrected") or crop_dir / "rgb_corrected.tif")
    if not rgb_path.is_absolute():
        rgb_path = crop_dir / rgb_path
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing corrected orthophoto RGB at {rgb_path}")
    rgb = tifffile.imread(rgb_path)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    extent = normalize_lv95_extent_for_emboss(metadata["extent_lv95"])
    return OrthophotoCrop(rgb=np.asarray(rgb), extent_lv95=extent, metadata=metadata, path=rgb_path)
