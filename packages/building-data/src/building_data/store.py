"""Explicit per-application acquisition storage; no source-checkout state."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from filelock import FileLock
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform

from .geometry import build_house_index, load_vector_house
from .providers import Bounds, BuildingProvider
from .storage import contained_path, write_json
from .swiss import SwissProvider

TO_LV95 = Transformer.from_crs(4326, 2056, always_xy=True)
TO_WGS84 = Transformer.from_crs(2056, 4326, always_xy=True)


@dataclass(frozen=True)
class HouseInput:
    id: str
    building_fid: int
    root: Path
    surfaces_path: Path
    lidar_paths: tuple[Path, ...]
    bounds_xy: Bounds
    sources: tuple[dict, ...]
    crs: str = "EPSG:2056"
    survey_year: int | None = None

    def as_dict(self) -> dict:
        to_wgs84 = Transformer.from_crs(self.crs, 4326, always_xy=True)
        lon, lat = to_wgs84.transform(
            (self.bounds_xy[0] + self.bounds_xy[2]) / 2,
            (self.bounds_xy[1] + self.bounds_xy[3]) / 2,
        )
        return {
            "id": self.id,
            "building_fid": self.building_fid,
            "bounds_xy": list(self.bounds_xy),
            "crs": self.crs,
            "longitude": lon,
            "latitude": lat,
            "survey_year": self.survey_year,
            "sources": [
                {k: v for k, v in source.items() if k != "path"}
                for source in self.sources
            ],
        }


def selection_geometry(selection: dict):
    """Validate a selection and return its WGS84 longitude/latitude geometry."""
    if not isinstance(selection, dict):
        raise ValueError("Selection must be an object")
    mode = selection.get("mode", "house")
    if mode == "house":
        try:
            longitude = float(selection["longitude"])
            latitude = float(selection["latitude"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "House selection requires numeric longitude and latitude"
            ) from error
        if not np.isfinite([longitude, latitude]).all():
            raise ValueError("House coordinates must be finite")
        geo = Point(longitude, latitude)
    elif mode == "area":
        if selection.get("geometry") is not None:
            value = selection["geometry"]
            if isinstance(value, dict) and value.get("type") == "Feature":
                value = value.get("geometry")
            if not isinstance(value, dict) or value.get("type") not in (
                "Polygon",
                "MultiPolygon",
            ):
                raise ValueError(
                    "Area geometry must be a GeoJSON Polygon or MultiPolygon"
                )
            try:
                geo = shape(value)
            except (
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
                IndexError,
            ) as error:
                raise ValueError("Invalid area geometry coordinates") from error
        else:
            values = selection.get("bbox")
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError("Area bbox must be [west,south,east,north]")
            try:
                coords = tuple(float(value) for value in values)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Area bbox coordinates must be numbers") from error
            if (
                not np.isfinite(coords).all()
                or coords[2] <= coords[0]
                or coords[3] <= coords[1]
            ):
                raise ValueError("Area bbox must be [west,south,east,north]")
            geo = box(*coords)
    else:
        raise ValueError("Selection mode must be house or area")
    if geo.is_empty or not np.isfinite(geo.bounds).all() or not geo.is_valid:
        raise ValueError("Invalid selection geometry")
    if not box(-180, -90, 180, 90).covers(geo):
        raise ValueError("Selection coordinates must be valid longitude and latitude")
    return mode, geo


class AcquisitionStore:
    def __init__(
        self, root: Path, workers: int = 2, provider: BuildingProvider | None = None
    ):
        self.root = Path(root).expanduser().resolve()
        self.workers = max(1, int(workers))
        self.root.mkdir(parents=True, exist_ok=True)
        self.provider = provider or SwissProvider(self.root / "cache", self.workers)

    def house(self, house_id: str) -> HouseInput:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", house_id):
            raise ValueError("Invalid house ID")
        root = contained_path(self.root, f"houses/{house_id}")
        data = json.loads((root / "house.json").read_text())
        if not isinstance(data, dict) or data.get("schema") != "building-input-v1":
            raise ValueError("Unsupported building input schema")
        if data.get("id") != house_id:
            raise ValueError("House manifest ID does not match the requested house")
        if data.get("units") != "m":
            raise ValueError("Building inputs must use metric coordinates")
        sources = tuple(
            {**s, "path": str(contained_path(self.root, s["path"]))}
            for s in data["sources"]
        )
        return HouseInput(
            house_id,
            int(data["building_fid"]),
            root,
            contained_path(self.root, data["surfaces_path"]),
            tuple(Path(s["path"]) for s in sources),
            tuple(data["bounds_xy"]),
            sources,
            data["crs"],
            data.get("survey_year"),
        )

    def ensure_house_coverage(
        self, item: HouseInput, house, progress=None
    ) -> HouseInput:
        """Refresh legacy input extents that omitted disconnected roof sections."""
        if tuple(item.bounds_xy) == tuple(house.bounds_xy):
            return item
        with FileLock(str(item.root / "prepare.lock")):
            manifest = item.root / "house.json"
            saved = json.loads(manifest.read_text())
            if tuple(saved["bounds_xy"]) != tuple(house.bounds_xy):
                minx, miny, maxx, maxy = house.bounds_xy
                _paths, sources = self.provider.lidar(
                    (minx - 2, miny - 2, maxx + 2, maxy + 2),
                    progress=progress or (lambda _message: None),
                )
                saved["bounds_xy"] = list(house.bounds_xy)
                saved["sources"] = [
                    {**source, "path": str(Path(source["path"]).relative_to(self.root))}
                    for source in sources
                ]
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                anchor = next(
                    (
                        source
                        for source in sources
                        if source["bounds"][0] <= cx < source["bounds"][2]
                        and source["bounds"][1] <= cy < source["bounds"][3]
                    ),
                    sources[0],
                )
                saved["survey_year"] = anchor["year"]
                write_json(manifest, saved)
        return self.house(item.id)

    def list_houses(self) -> list[dict]:
        return [
            self.house(p.parent.name).as_dict()
            for p in sorted((self.root / "houses").glob("*/house.json"))
        ]

    def acquire(
        self, selection: dict, progress: Callable[[str], None] | None = None
    ) -> dict:
        progress = progress or (lambda message: None)
        mode, wgs84_geometry = selection_geometry(selection)
        self.provider.validate_selection(wgs84_geometry)
        to_provider = Transformer.from_crs(4326, self.provider.crs, always_xy=True)
        geometry = transform(to_provider.transform, wgs84_geometry)
        identity = {
            "selection": selection,
            "provider": self.provider.name,
            "revision": self.provider.revision,
        }
        acquisition_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:16]
        root = self.root / "acquisitions" / acquisition_id
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / "acquire.lock")):
            surfaces = root / "surfaces.gpkg"
            # Context around the selection preserves complete buildings and touching-house topology.
            if not surfaces.exists():
                self.provider.buildings(geometry.buffer(50).bounds, surfaces, progress)
            index = build_house_index(surfaces)
            candidates = []
            failures = []
            seen = set()
            for row in index.itertuples(index=False):
                fid = int(row.building_fid)
                if fid in seen:
                    continue
                seen.add(fid)
                if getattr(row, "roof_face_count", None) == 0:
                    continue
                try:
                    house = load_vector_house(surfaces, fid, index=index)
                    if not house.roof_faces:
                        continue
                    if mode == "house":
                        distance = house.roof_envelope.distance(geometry)
                        if distance <= 30:
                            candidates.append((distance, fid, house))
                    elif house.roof_envelope.intersects(geometry):
                        candidates.append((0, fid, house))
                except Exception as error:
                    failures.append({"building_fid": fid, "error": str(error)})
            candidates.sort(key=lambda item: (item[0], item[1]))
            if mode == "house":
                candidates = candidates[:1]
            if not candidates and not failures:
                raise ValueError("No roof-bearing building found in this selection")
            progress(f"Found {len(candidates)} building(s)")
            # Acquire every tile in an area, even tiles without buildings.
            if mode == "area":
                self.provider.lidar(geometry.bounds, progress=progress)
            house_ids = []

            def prepare(item):
                _, fid, house = item
                house_id = self.provider.house_id(fid)
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", house_id):
                    raise ValueError("Provider returned an invalid house ID")
                directory = self.root / "houses" / house_id
                directory.mkdir(parents=True, exist_ok=True)
                with FileLock(str(directory / "prepare.lock")):
                    manifest = directory / "house.json"
                    if manifest.exists():
                        saved = json.loads(manifest.read_text())
                        if saved["bounds_xy"] == list(house.bounds_xy):
                            return house_id
                    minx, miny, maxx, maxy = house.bounds_xy
                    _paths, sources = self.provider.lidar(
                        (minx - 2, miny - 2, maxx + 2, maxy + 2), progress=progress
                    )
                    relative_sources = [
                        {**s, "path": str(Path(s["path"]).relative_to(self.root))}
                        for s in sources
                    ]
                    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                    anchor = next(
                        (
                            s
                            for s in sources
                            if s["bounds"][0] <= cx < s["bounds"][2]
                            and s["bounds"][1] <= cy < s["bounds"][3]
                        ),
                        sources[0],
                    )
                    write_json(
                        manifest,
                        {
                            "schema": "building-input-v1",
                            "id": house_id,
                            "building_fid": fid,
                            "provider": self.provider.name,
                            "provider_revision": self.provider.revision,
                            "crs": self.provider.crs,
                            "vertical_crs": self.provider.vertical_crs,
                            "units": "m",
                            "surfaces_path": str(surfaces.relative_to(self.root)),
                            "bounds_xy": list(house.bounds_xy),
                            "sources": relative_sources,
                            "survey_year": anchor["year"],
                        },
                    )
                return house_id

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(prepare, item): item[1] for item in candidates
                }
                for done, future in enumerate(as_completed(futures), 1):
                    try:
                        house_ids.append(future.result())
                    except Exception as error:
                        failures.append(
                            {"building_fid": futures[future], "error": str(error)}
                        )
                    progress(f"Prepared {done}/{len(candidates)} buildings")
            result = {
                "id": acquisition_id,
                **identity,
                "houses": sorted(house_ids),
                "failures": sorted(failures, key=lambda item: item["building_fid"]),
            }
            write_json(root / "acquisition.json", result)
            return result

    def export_scene(self, house_id: str, scene_dir: Path) -> Path:
        """Export a generic scene bundle without loading reconstruction or segmentation."""
        item = self.house(house_id)
        target = Path(scene_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target) + ".lock"):
            if any(path.is_symlink() for path in (target, *target.parents)):
                raise ValueError("Scene destination cannot contain symlinks")
            if (target / "scene.json").is_symlink():
                raise ValueError("Scene manifest cannot be a symlink")
            if (target / "scene.json").exists():
                manifest = json.loads((target / "scene.json").read_text())
                # Only add missing terrain. Existing DEMs and all user assets stay
                # untouched, including unreferenced user-supplied dem.tif files.
                if not manifest.get("dem") and not (target / "dem.tif").exists():
                    if (target / "dem.tif").is_symlink():
                        raise ValueError("Scene terrain cannot be a symlink")
                    temporary = target.parent / (".terrain-" + uuid.uuid4().hex)
                    temporary.mkdir()
                    try:
                        x0, y0, x1, y1 = item.bounds_xy
                        terrain = self.provider.terrain(
                            (x0 - 25, y0 - 25, x1 + 25, y1 + 25),
                            temporary / "dem.tif",
                            pinned_sources=item.sources,
                        )
                        if terrain is not None:
                            if (
                                Path(terrain) != temporary / "dem.tif"
                                or Path(terrain).is_symlink()
                            ):
                                raise ValueError(
                                    "Terrain provider must write the requested regular raster"
                                )
                            # link refuses to replace a concurrently supplied DEM.
                            (target / "dem.tif").hardlink_to(terrain)
                            manifest["dem"] = "dem.tif"
                            write_json(target / "scene.json", manifest)
                    finally:
                        shutil.rmtree(temporary)
                return target / "scene.json"
            temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                house = load_vector_house(item.surfaces_path, item.building_fid)
                x0, y0, x1, y1 = item.bounds_xy
                frame, _sources = self.provider.points_for_bounds(
                    (x0 - 2, y0 - 2, x1 + 2, y1 + 2),
                    item.survey_year,
                    pinned_sources=item.sources,
                )
                header = laspy.LasHeader(point_format=3, version="1.2")
                header.scales = np.array([0.001, 0.001, 0.001])
                header.offsets = np.floor(frame[["x", "y", "z"]].min().to_numpy())
                data = laspy.LasData(header)
                data.x = frame.x.to_numpy()
                data.y = frame.y.to_numpy()
                data.z = frame.z.to_numpy()
                data.classification = frame.classification.to_numpy().astype(np.uint8)
                data.write(temporary / "points.las")
                write_json(
                    temporary / "outline.geojson",
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": mapping(house.roof_envelope),
                                "properties": {
                                    "kind": "roof_envelope",
                                    "z_min": house.bounds_xyz[2],
                                    "z_max": house.bounds_xyz[5],
                                },
                            }
                        ],
                    },
                )
                manifest = {
                    "schema": "label3d-scene-v1",
                    "name": house_id,
                    "collection": "acquired",
                    "crs": item.crs,
                    "units": "m",
                    "points": "points.las",
                    "building_outline": "outline.geojson",
                    "focus_extent": list(item.bounds_xy),
                    "base_height": house.bounds_xyz[2],
                }
                terrain = self.provider.terrain(
                    (x0 - 25, y0 - 25, x1 + 25, y1 + 25),
                    temporary / "dem.tif",
                    pinned_sources=item.sources,
                )
                if terrain is not None:
                    if Path(terrain).resolve() != (temporary / "dem.tif").resolve():
                        raise ValueError(
                            "Terrain provider must write the requested output"
                        )
                    if not Path(terrain).is_file() or Path(terrain).is_symlink():
                        raise ValueError(
                            "Terrain provider did not write a regular raster"
                        )
                    manifest["dem"] = "dem.tif"
                write_json(temporary / "scene.json", manifest)
                if target.exists():
                    if any(target.iterdir()):
                        raise ValueError("Scene destination is not empty")
                    target.rmdir()
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return target / "scene.json"
