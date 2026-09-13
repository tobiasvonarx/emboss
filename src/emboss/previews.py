"""Display saved scientific rasters and try correction sources in a separate cache."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import tifffile
from building_data.geometry import load_vector_house
from building_data.orthophoto_correction.geometry import crop_bounds_lv95
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.pipeline import correct_house_orthophoto
from building_data.orthophoto_correction.prepare import prepare_assets
from building_data.orthophoto_correction.selection import RAW_CANDIDATE_ID
from building_data.storage import checksum, contained_path, write_json
from filelock import FileLock
from PIL import Image
from pyproj import Transformer

from .api import Client, Reconstruction, _imagery_fingerprint
from .export import write_vector_house_geojson

# Presentation colors only. Scientific class IDs and raster pixels stay unchanged.
CLASS_PALETTE = (
    (0, "Solar panels", "#4169e1"),
    (1, "Dormers", "#ef8354"),
    (2, "Roof windows", "#27b9bd"),
    (3, "Balconies", "#aa64c7"),
    (4, "Other roof details", "#e7bd35"),
)


def _artifact(result: Reconstruction, name: str) -> Path:
    bundle = json.loads(result.manifest_path.read_text())
    return contained_path(result.root, bundle["artifacts"][name])


def _metadata_path(result: Reconstruction) -> Path:
    return result.root / "inputs/orthophoto_correction.json"


def footprint_overlay(result: Reconstruction, metadata: dict) -> dict | None:
    """Project the provider's saved scaffold envelope onto the image pixel grid.

    GeoJSON rings retain courtyard holes and disconnected components. This is a
    display layer only; neither source geometry nor scientific rasters are edited.
    """
    if "transform" not in metadata:
        return None  # Older results may lack georeferencing.
    collection = json.loads(_artifact(result, "vector_house").read_text())
    source_crs = collection["crs"]["properties"]["name"]
    project = Transformer.from_crs(source_crs, metadata["crs"], always_xy=True)
    x0, dx, rx, y0, ry, dy = metadata["transform"]
    inverse = np.linalg.inv(np.array([[dx, rx], [ry, dy]], dtype=float))
    rings = []
    for feature in collection["features"]:
        if feature.get("properties", {}).get("kind") != "roof_envelope":
            continue
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            raise ValueError("Expected a polygonal scaffold footprint")
        for polygon in polygons:
            for ring in polygon:
                xy = np.asarray(ring, dtype=float)[:, :2]
                x, y = project.transform(xy[:, 0], xy[:, 1])
                pixels = (inverse @ (np.array([x, y]) - [[x0], [y0]])).T
                rings.append(pixels.tolist())
    return {"width": metadata["width"], "height": metadata["height"], "rings": rings}


def imagery_info(result: Reconstruction) -> dict:
    """Return source choices belonging to this result's saved imagery snapshot."""
    metadata = json.loads(_metadata_path(result).read_text())
    selection = metadata.get("candidate_selection", {})
    selected = selection.get("selected_id")
    override = metadata.get("requested_strip_override", selection.get("override_id"))
    scores = {row["candidate_id"]: row for row in selection.get("candidate_scores", [])}
    candidates = []
    seen = set()
    for row in [{"id": RAW_CANDIDATE_ID}, *metadata.get("strip_candidates", [])]:
        candidate_id = row["id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        raw = candidate_id == RAW_CANDIDATE_ID
        date = row.get("flight_date")
        candidates.append(
            {
                "id": candidate_id,
                "flight_date": date,
                "year": row.get("flight_year", metadata.get("source_year")),
                "selected": candidate_id == selected,
                "kind": "raw" if raw else "corrected",
                "roof_iou": scores.get(candidate_id, {}).get("roof_iou"),
            }
        )
    classes = [
        {"id": class_id, "label": label, "color": color}
        for class_id, label, color in CLASS_PALETTE
    ]
    return {
        "selected_id": selected,
        "override_id": override,
        "source_year": metadata.get("source_year"),
        "candidates": candidates,
        "classes": classes,
        "footprint": footprint_overlay(result, metadata),
    }


def _png(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def orthophoto_png(result: Reconstruction) -> bytes:
    """Encode the exact saved RGB grid without resizing or resampling."""
    rgb = tifffile.imread(_artifact(result, "orthophoto"))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an unsigned 8-bit RGB orthophoto")
    return _png(rgb)


def segmentation_png(result: Reconstruction) -> bytes:
    """Color the saved class grid, with fully transparent background pixels."""
    labels = tifffile.imread(_artifact(result, "segmentation"))
    if labels.ndim != 2 or not np.isin(labels, np.arange(6)).all():
        raise ValueError("Expected a two-dimensional RID2 class map with IDs 0–5")
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for class_id, _label, color in CLASS_PALETTE:
        rgba[labels == class_id] = [
            *(int(color[i : i + 2], 16) for i in (1, 3, 5)),
            180,
        ]
    return _png(rgba)


def _correction_config(client, candidate_id=None):
    return CorrectionConfig(
        cache_root=client.store.root / "cache/imagery",
        mobile_sam_checkpoint=client.store.root / "cache/models/mobile_sam.pt",
        strip_id_override=candidate_id,
    )


def _correct_preview(client, item, config, progress):
    house = load_vector_house(item.surfaces_path, item.building_fid)
    center = house.roof_envelope.centroid
    source = next(
        (
            source
            for source in item.sources
            if source["bounds"][0] <= center.x < source["bounds"][2]
            and source["bounds"][1] <= center.y < source["bounds"][3]
        ),
        item.sources[0],
    )
    with laspy.open(source["path"]) as reader:
        bounds = (
            *map(float, reader.header.mins[:2]),
            *map(float, reader.header.maxs[:2]),
        )
    assets = prepare_assets(
        tile_key=source["tile"],
        bounds_lv95=bounds,
        cache=config.cache_root,
        config=config,
        progress=progress,
        required_bounds_lv95=crop_bounds_lv95(
            house.bounds_xy, padding_m=8.0, min_side_m=config.min_crop_side_m
        ),
    )
    workspace = SimpleNamespace(
        tile_key=source["tile"],
        tile_bounds_xy=bounds,
        correction_raster_assets=assets,
    )
    progress("Previewing source correction")
    with client._inference_lock:
        corrected = correct_house_orthophoto(
            {"house": house}, workspace, padding_m=8.0, config=config
        )
    return corrected, house


def _preview_source(
    client: Client, house_id: str, candidate_id: str, *, progress=None, snapshot=None
) -> dict:
    """Run the existing correction method without replacing imagery or results.

    Only prepared input caches and this separate PNG cache may be written. Choosing
    a source permanently remains the reconstruction API's explicit operation.
    """
    result = snapshot if snapshot is not None else client.load_result(house_id)
    info = imagery_info(result)
    if candidate_id not in {row["id"] for row in info["candidates"]}:
        raise ValueError("The requested source is not available for this result")
    progress = progress or (lambda _message: None)
    item = result.house_input
    config = _correction_config(client, candidate_id)
    identity = {
        "schema": "emboss-source-preview-v1",
        "imagery": _imagery_fingerprint(item, config, checksum(item.surfaces_path)),
        "metadata_sha256": checksum(_metadata_path(result)),
        "orthophoto_sha256": checksum(_artifact(result, "orthophoto")),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = client.store.root / "cache/source-previews" / key
    cache.mkdir(parents=True, exist_ok=True)
    png_path, receipt_path = cache / "orthophoto.png", cache / "preview.json"
    with FileLock(str(cache) + ".lock"):
        try:
            receipt = json.loads(receipt_path.read_text())
            if checksum(png_path) == receipt["png_sha256"]:
                return {
                    "png_path": png_path,
                    "selected_id": candidate_id,
                    "cached": True,
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if candidate_id == info["selected_id"]:
            png = orthophoto_png(result)
        else:
            corrected, _house = _correct_preview(client, item, config, progress)
            footprint = info["footprint"]
            if footprint and corrected.rgb_corrected.shape[:2] != (
                footprint["height"],
                footprint["width"],
            ):
                raise ValueError(
                    "The scaffold extent changed. Rebuild before comparing imagery."
                )
            png = _png(corrected.rgb_corrected)
        temporary = cache / (uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_bytes(png)
            temporary.replace(png_path)
        finally:
            temporary.unlink(missing_ok=True)
        write_json(
            receipt_path,
            {**identity, "png_sha256": hashlib.sha256(png).hexdigest()},
        )
        return {"png_path": png_path, "selected_id": candidate_id, "cached": False}


def preview_candidate(client: Client, house_id: str, candidate_id: str) -> bytes:
    return _preview_source(client, house_id, candidate_id)["png_path"].read_bytes()


@dataclass(frozen=True)
class _PreparedImagery:
    """An image-only snapshot; no segmentation or reconstruction artifacts."""

    root: Path
    manifest_path: Path
    house_input: object


def _saved_imagery(client, house_id):
    try:
        result = client.load_result(house_id)
        return result, imagery_info(result)
    except (FileNotFoundError, ValueError):
        return None, None


def _house_imagery(client, house_id, *, prepare=False, progress=None):
    item = client.store.house(house_id)
    saved, saved_info = _saved_imagery(client, house_id)
    if saved_info and saved_info["override_id"] is None:
        return saved, saved_info
    config = _correction_config(client)
    identity = {
        "schema": "emboss-image-gallery-v1",
        "imagery": _imagery_fingerprint(item, config, checksum(item.surfaces_path)),
        "lidar_sha256": [checksum(Path(source["path"])) for source in item.sources],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = client.store.root / "cache/image-galleries" / key
    snapshot = _PreparedImagery(root, root / "imagery.json", item)
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock"):
        try:
            manifest = json.loads(snapshot.manifest_path.read_text())
            if (
                manifest["identity"] == identity
                and set(manifest["artifact_sha256"])
                == {"orthophoto", "metadata", "vector_house"}
                and all(
                    checksum(_artifact(snapshot, name)) == expected
                    for name, expected in manifest["artifact_sha256"].items()
                )
            ):
                return snapshot, saved_info
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if not prepare:
            raise FileNotFoundError(
                "Prepare the building imagery before loading previews"
            )
        progress = progress or (lambda _message: None)
        progress("Preparing aerial-image choices")
        corrected, house = _correct_preview(client, item, config, progress)
        root.mkdir(parents=True, exist_ok=True)
        (root / "inputs").mkdir(exist_ok=True)
        tifffile.imwrite(root / "inputs/orthophoto.tif", corrected.rgb_corrected)
        write_json(_metadata_path(snapshot), corrected.metadata)
        write_vector_house_geojson(root / "vector_house.geojson", house, crs=item.crs)
        artifacts = {
            "orthophoto": "inputs/orthophoto.tif",
            "metadata": "inputs/orthophoto_correction.json",
            "vector_house": "vector_house.geojson",
        }
        write_json(
            snapshot.manifest_path,
            {
                "schema": "emboss-image-gallery-v1",
                "identity": identity,
                "artifacts": artifacts,
                "artifact_sha256": {
                    name: checksum(root / path) for name, path in artifacts.items()
                },
            },
        )
        progress("Aerial-image choices ready")
        return snapshot, saved_info


def prepare_house_imagery(client: Client, house_id: str, *, progress=None) -> dict:
    """Prepare default corrected imagery without choosing inputs or running Emboss."""
    snapshot, saved_info = _house_imagery(
        client, house_id, prepare=True, progress=progress
    )
    info = imagery_info(snapshot)
    recommended = info["selected_id"]
    current = saved_info["selected_id"] if saved_info else None
    selected = (
        current if current in {row["id"] for row in info["candidates"]} else recommended
    )
    return {
        **info,
        "house_id": house_id,
        "selected_id": selected,
        "current_id": current,
        "recommended_id": recommended,
        "has_reconstruction": saved_info is not None,
        "override_id": saved_info["override_id"] if saved_info else None,
        "candidates": [
            {
                **row,
                "selected": row["id"] == selected,
                "recommended": row["id"] == recommended,
            }
            for row in info["candidates"]
        ],
    }


def house_preview_candidate(client: Client, house_id: str, candidate_id: str) -> bytes:
    snapshot, _saved_info = _house_imagery(client, house_id)
    saved, saved_info = _saved_imagery(client, house_id)
    if saved_info and candidate_id == saved_info["selected_id"]:
        return orthophoto_png(saved)
    return _preview_source(client, house_id, candidate_id, snapshot=snapshot)[
        "png_path"
    ].read_bytes()
