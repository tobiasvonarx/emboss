"""Public reconstruction API. Storage, acquisition and method inputs are explicit."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import laspy
import tifffile
from building_data.geometry import (
    build_house_index,
    house_index_rows,
    load_vector_house,
    load_vector_house_scaffold,
)
from building_data.orthophoto_correction.geometry import crop_bounds_lv95
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.pipeline import correct_house_orthophoto
from building_data.orthophoto_correction.prepare import prepare_assets
from building_data.orthophoto_correction.raster import write_correction_outputs
from building_data.raster import rasterize_polygon
from building_data.storage import checksum, contained_path, write_json
from building_data.store import AcquisitionStore, HouseInput
from filelock import FileLock

from .export import write_vector_house_geojson
from .lidar import prepare_lidar_observations_from_points
from .mesh import write_prediction_mesh, write_swissbuildings3d_mesh
from .pipeline import run_emboss_house
from .roof_superstructures import (
    BACKGROUND_CLASS_ID,
    RoofSuperstructureMaskClient,
    RoofSuperstructureOptions,
)


def _imagery_fingerprint(
    item: HouseInput, config: CorrectionConfig, vector_sha256: str
) -> str:
    from building_data.orthophoto_correction.mobile_sam import (
        MODEL_REPOSITORY,
        MODEL_REVISION,
        MODEL_SHA256,
    )

    configuration = {
        key: value
        for key, value in asdict(config).items()
        if key not in ("cache_root", "mobile_sam_checkpoint", "local_strip_catalog")
    }
    identity = {
        "schema": "emboss-imagery-v2",
        "building_fid": item.building_fid,
        "vector_sha256": vector_sha256,
        "bounds_xy": list(item.bounds_xy),
        "sources": [
            {key: value for key, value in source.items() if key != "path"}
            for source in item.sources
        ],
        "correction": configuration,
        "crop_padding_m": 8.0,
        "mobile_sam": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "sha256": MODEL_SHA256,
        },
        "local_strip_catalog_sha256": checksum(config.local_strip_catalog)
        if config.local_strip_catalog and config.local_strip_catalog.is_file()
        else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _imagery_state(image_dir: Path, fingerprint: str) -> dict | None:
    rgb_path = image_dir / "rgb_corrected.tif"
    metadata_path = image_dir / "orthophoto_correction.json"
    if not rgb_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (ValueError, OSError):
        return None
    if (
        not isinstance(metadata, dict)
        or metadata.get("imagery_fingerprint") != fingerprint
    ):
        return None
    return {
        "fingerprint": fingerprint,
        "rgb_sha256": checksum(rgb_path),
        "metadata_sha256": checksum(metadata_path),
    }


@dataclass(frozen=True)
class Reconstruction:
    root: Path
    manifest_path: Path
    house_input: HouseInput

    @property
    def mesh_path(self):
        return self.root / "mesh.ply"

    @property
    def roof_details_path(self):
        return self.root / "roof_details.geojson"

    @property
    def scaffold_path(self):
        return self.root / "scaffold.geojson"

    @property
    def orthophoto_path(self):
        return self.root / "inputs/orthophoto.tif"

    def as_dict(self) -> dict:
        bundle = json.loads(self.manifest_path.read_text())
        result = json.loads((self.root / "result.json").read_text())
        return {
            "house_id": self.house_input.id,
            "solid_count": result["solid_count"],
            "artifacts": bundle["artifacts"],
            "provenance": bundle["provenance"],
        }


class Client:
    def __init__(self, data_dir: str | Path, workers: int = 2, device: str = "auto"):
        self.store = AcquisitionStore(Path(data_dir), workers=workers)
        self.workers = max(1, int(workers))
        self.segmentation = RoofSuperstructureMaskClient(
            RoofSuperstructureOptions(device=device)
        )
        self._inference_lock = threading.Lock()

    def load_result(self, house_id: str) -> Reconstruction:
        house = self.store.house(house_id)
        root = self.store.root / "results" / house.id
        bundle = json.loads((root / "bundle.json").read_text())
        if not isinstance(bundle, dict) or bundle.get("schema") != "emboss-bundle-v1":
            raise ValueError("Unsupported Emboss bundle schema")
        if bundle.get("house_id", house_id) != house_id:
            raise ValueError("Reconstruction bundle belongs to a different house")
        artifacts = bundle.get("artifacts")
        hashes = bundle.get("artifact_sha256", {})
        required = {
            "result",
            "prediction",
            "mesh",
            "roof_details",
            "orthophoto",
            "vector_house",
            "segmentation",
        }
        if not isinstance(artifacts, dict) or not required.issubset(artifacts):
            raise ValueError("Incomplete reconstruction artifact manifest")
        if not isinstance(hashes, dict):
            raise ValueError("Invalid reconstruction artifact checksums")  # noqa: TRY004 - invalid saved JSON
        for name, relative in artifacts.items():
            if not isinstance(relative, str):
                raise ValueError("Invalid reconstruction artifact path")  # noqa: TRY004 - invalid saved JSON
            path = contained_path(root, relative)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Incomplete reconstruction artifact: {relative}"
                )
            expected = hashes.get(name)
            if expected and checksum(path) != expected:
                raise ValueError(
                    f"Reconstruction artifact checksum mismatch: {relative}"
                )
        return Reconstruction(root, root / "bundle.json", house)

    def reconstruct(
        self,
        house_id: str,
        *,
        force: bool = False,
        strip_id_override: str | None = None,
        reset_strip_override: bool = False,
        progress=None,
    ) -> Reconstruction:
        """Reconstruct a house, preserving its selected strip unless explicitly reset.

        Pass reset_strip_override=True to restore automatic candidate selection.
        """
        progress = progress or (lambda message: None)
        item = self.store.house(house_id)
        root = self.store.root / "results" / item.id
        root.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root) + ".lock"):
            image_dir = item.root / "imagery"
            image_meta = image_dir / "orthophoto_correction.json"
            try:
                saved_metadata = json.loads(image_meta.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                saved_metadata = {}
            if not isinstance(saved_metadata, dict):
                saved_metadata = {}
            if reset_strip_override:
                strip_id_override = None
            elif strip_id_override is None:
                strip_id_override = saved_metadata.get("requested_strip_override")
            config = CorrectionConfig(
                cache_root=self.store.root / "cache/imagery",
                mobile_sam_checkpoint=self.store.root / "cache/models/mobile_sam.pt",
                strip_id_override=strip_id_override,
            )
            progress("Checking inputs and segmentation checkpoint")
            model_path = self.segmentation.checkpoint_path
            original_house = load_vector_house(item.surfaces_path, item.building_fid)
            item = self.store.ensure_house_coverage(
                item, original_house, progress=progress
            )
            vector_sha256 = checksum(item.surfaces_path)
            lidar_sha256 = [checksum(path) for path in item.lidar_paths]
            imagery_fingerprint = _imagery_fingerprint(item, config, vector_sha256)
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "input": json.loads((item.root / "house.json").read_text()),
                        "vector_sha256": vector_sha256,
                        "lidar_sha256": lidar_sha256,
                        "model_sha256": checksum(model_path),
                        "imagery_fingerprint": imagery_fingerprint,
                        "strip_override": strip_id_override,
                        "method": "emboss-result-v6",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if not force and (root / "bundle.json").exists():
                try:
                    bundle = json.loads((root / "bundle.json").read_text())
                except json.JSONDecodeError:
                    bundle = {}
                imagery_state = _imagery_state(image_dir, imagery_fingerprint)
                if (
                    isinstance(bundle, dict)
                    and bundle.get("fingerprint") == fingerprint
                    and imagery_state is not None
                    and bundle.get("imagery") == imagery_state
                ):
                    try:
                        cached_result = self.load_result(house_id)
                    except (FileNotFoundError, ValueError):
                        progress(
                            "Rebuilding incomplete or changed reconstruction artifacts"
                        )
                    else:
                        progress("Reusing completed reconstruction")
                        return cached_result
            work = root.with_name(root.name + ".tmp-" + uuid.uuid4().hex)
            work.mkdir()
            try:
                progress("Preparing roof geometry and imagery")
                topology = next(
                    row
                    for row in house_index_rows(build_house_index(item.surfaces_path))
                    if row.building_fid == item.building_fid
                )
                original_house = replace(
                    original_house,
                    is_independent=topology.is_independent,
                    touching_building_fids=topology.touching_building_fids,
                )
                scaffold = work / "scaffold.geojson"
                write_vector_house_geojson(scaffold, original_house, crs=item.crs)
                # Preserve the original scaffold serialization/reload before fitting.
                house = load_vector_house_scaffold(scaffold, item.building_fid)
                with FileLock(str(item.root / "imagery.lock")):
                    if force or _imagery_state(image_dir, imagery_fingerprint) is None:
                        center = house.roof_envelope.centroid
                        source = next(
                            (
                                s
                                for s in item.sources
                                if s["bounds"][0] <= center.x < s["bounds"][2]
                                and s["bounds"][1] <= center.y < s["bounds"][3]
                            ),
                            item.sources[0],
                        )
                        with laspy.open(source["path"]) as reader:
                            bounds = (
                                *map(float, reader.header.mins[:2]),
                                *map(float, reader.header.maxs[:2]),
                            )
                        crop_bounds = crop_bounds_lv95(
                            house.bounds_xy,
                            padding_m=8.0,
                            min_side_m=config.min_crop_side_m,
                        )
                        assets = prepare_assets(
                            tile_key=source["tile"],
                            bounds_lv95=bounds,
                            cache=config.cache_root,
                            config=config,
                            progress=progress,
                            required_bounds_lv95=crop_bounds,
                        )
                        workspace = SimpleNamespace(
                            tile_key=source["tile"],
                            tile_bounds_xy=bounds,
                            correction_raster_assets=assets,
                        )
                        corrected = correct_house_orthophoto(
                            {"house": original_house},
                            workspace,
                            padding_m=8.0,
                            config=config,
                        )
                        paths = write_correction_outputs(corrected, image_dir)
                        metadata = {
                            **corrected.metadata,
                            "requested_strip_override": strip_id_override,
                            "imagery_fingerprint": imagery_fingerprint,
                            "outputs": {k: Path(v).name for k, v in paths.items()},
                        }
                        write_json(image_meta, metadata)
                    metadata = json.loads(image_meta.read_text())
                    imagery_state = _imagery_state(image_dir, imagery_fingerprint)
                    rgb = tifffile.imread(image_dir / "rgb_corrected.tif")
                    x0, y0, x1, y1 = metadata["extent_lv95"]
                    extent = (x0, x1, y0, y1)
                    (work / "inputs").mkdir()
                    shutil.copy2(
                        image_dir / "rgb_corrected.tif", work / "inputs/orthophoto.tif"
                    )
                    shutil.copy2(image_meta, work / "inputs/orthophoto_correction.json")
                mask = rasterize_polygon(
                    house.roof_envelope,
                    extent=extent,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                )
                progress("Segmenting roof details")
                with self._inference_lock:
                    segmentation = self.segmentation.predict_hard_segmentation(
                        rgb, vector_roof_mask=mask
                    )
                progress("Calibrating LiDAR and fitting roof details")
                a, b, c, d = house.bounds_xy
                points, _sources = self.store.provider.points_for_bounds(
                    (a - 1, b - 1, c + 1, d + 1),
                    item.survey_year,
                    progress=progress,
                    pinned_sources=item.sources,
                )
                observations = prepare_lidar_observations_from_points(points, house)
                result = run_emboss_house(
                    output_dir=work,
                    workspace_label=item.id,
                    house=house,
                    observations=observations,
                    rgb=rgb,
                    image_superstructure_mask=segmentation.foreground_class_map
                    != BACKGROUND_CLASS_ID,
                    image_superstructure_class_map=segmentation.foreground_class_map,
                    image_segmentation_source="computed",
                    vector_roof_mask=mask,
                    extent_lv95=extent,
                    crs=item.crs,
                )
                progress("Exporting the building mesh")
                base_mesh = work / "inputs/swissbuildings3d_mesh.ply"
                write_swissbuildings3d_mesh(
                    surfaces_vector_path=item.surfaces_path,
                    building_fid=item.building_fid,
                    output_path=base_mesh,
                )
                shutil.copy2(
                    work / "superstructures.geojson", work / "roof_details.geojson"
                )
                mesh_stats = write_prediction_mesh(
                    roof_scaffold=scaffold,
                    superstructures_path=work / "roof_details.geojson",
                    output_path=work / "mesh.ply",
                    base_mesh_path=base_mesh,
                    crs=item.crs,
                )
                shutil.copy2(
                    work / "rasters/superstructure_mask.tif", work / "mask.tif"
                )
                native = json.loads((work / "result.json").read_text())
                write_json(
                    work / "prediction.json",
                    {
                        "schema_version": "emboss-eval-prediction-v2",
                        "method": "emboss_swissbuildings3d",
                        "case_id": item.id,
                        "consumes": {
                            "roof_scaffold": "scaffold.geojson",
                            "lidar": "bundle:provenance.input.sources",
                            "orthophoto": "inputs/orthophoto.tif",
                        },
                        "limitations": [],
                        "artifacts": {
                            "mesh": "mesh.ply",
                            "roof_details": "roof_details.geojson",
                            "mask": "mask.tif",
                            "height_above_roof": "rasters/height_above_roof.tif",
                            "diagnostics": "diagnostics/fit_diagnostics.csv",
                        },
                        "metadata": {
                            "scaffold_provider": "swissbuildings3d",
                            "native_result": native,
                            "mesh": mesh_stats,
                            "input_mode": "prepared_inputs",
                            "source_workspace": item.id,
                        },
                    },
                )
                artifacts = {
                    "result": "result.json",
                    "prediction": "prediction.json",
                    "mesh": "mesh.ply",
                    "roof_details": "roof_details.geojson",
                    "orthophoto": "inputs/orthophoto.tif",
                    "vector_house": "vector_house.geojson",
                    "scaffold": "scaffold.geojson",
                    "segmentation": "rasters/image_superstructure_class_map.tif",
                }
                write_json(
                    work / "bundle.json",
                    {
                        "schema": "emboss-bundle-v1",
                        "house_id": item.id,
                        "fingerprint": fingerprint,
                        "imagery": imagery_state,
                        "artifacts": artifacts,
                        "artifact_sha256": {
                            name: checksum(work / relative)
                            for name, relative in artifacts.items()
                        },
                        "provenance": {
                            "input": item.as_dict(),
                            "checkpoint_sha256": checksum(model_path),
                            "vector_sha256": vector_sha256,
                            "lidar_sha256": lidar_sha256,
                            "segmentation": segmentation.diagnostics,
                            "crs": item.crs,
                            "vertical_crs": "EPSG:5728",
                            "bounds_xy": [x0, y0, x1, y1],
                            "strip_override": strip_id_override,
                        },
                    },
                )
                backup = root.with_name(root.name + ".previous")
                if backup.exists():
                    shutil.rmtree(backup)
                if root.exists():
                    root.rename(backup)
                try:
                    work.rename(root)
                except Exception:
                    if backup.exists():
                        backup.rename(root)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
                progress(f"Completed: {len(result.solids)} roof detail(s)")
            finally:
                if work.exists():
                    shutil.rmtree(work)
        return self.load_result(house_id)

    def reconstruct_many(
        self, house_ids: list[str], *, progress=None
    ) -> list[Reconstruction]:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(
                executor.map(
                    lambda house_id: self.reconstruct(house_id, progress=progress),
                    house_ids,
                )
            )
