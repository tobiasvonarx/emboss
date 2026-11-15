"""Local Cambridge sources: MasterMap footprints, EA points and supplied imagery.

Install once with ``building-data import-cambridge``. Both applications read
the same immutable source directory; their jobs and building results stay local.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import pyogrio
import shapely
from filelock import FileLock
from osgeo import gdal, ogr, osr
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform

from .points import read_las_points
from .storage import write_json

gdal.UseExceptions()
ogr.UseExceptions()
DEFAULT_ROOT = Path.home() / ".local/share/building-data/cambridge"
RETAINED_CLASSES = (2, 3, 4, 5, 6)


def source_root() -> Path:
    return Path(os.getenv("CAMBRIDGE_DATA", str(DEFAULT_ROOT))).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(imagery: Path, topography: Path, lidar: Path, root: Path) -> None:
    """Materialize source files, georeferencing and a spatial point index once."""
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "install.lock")):
        if (root / "dataset.json").exists():
            raise ValueError(f"Dataset already installed at {root}")
        images = root / "imagery"
        images.mkdir(exist_ok=True)
        with zipfile.ZipFile(imagery) as archive:
            for entry in archive.infolist():
                if Path(entry.filename).suffix.lower() in {".jpg", ".jgw", ".xml"}:
                    with (
                        archive.open(entry) as src,
                        (images / Path(entry.filename).name).open("wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)
        vrt_paths = []
        dates = {}
        import xml.etree.ElementTree as ET

        for jpg in sorted(images.glob("*.jpg")):
            metadata = ET.parse(jpg.with_suffix(".xml")).getroot()
            if metadata.find(".//{*}Rectangle").attrib["srsName"] != "osgb:BNG":
                raise ValueError(f"Unsupported image coordinates: {jpg.name}")
            if metadata.find("{*}correctionType").text != "Orthorectified":
                raise ValueError(
                    f"Expected supplied orthorectified imagery: {jpg.name}"
                )
            dates[jpg.stem] = metadata.find("{*}dateFlown").text
            vrt = jpg.with_suffix(".vrt")
            ds = gdal.Translate(
                str(vrt), str(jpg), format="VRT", outputSRS="EPSG:27700"
            )
            if not np.allclose(
                [ds.GetGeoTransform()[1], ds.GetGeoTransform()[5]], [0.25, -0.25]
            ):
                raise ValueError("Expected 25 cm imagery")
            ds = None
            vrt_paths.append(str(vrt))
        if not vrt_paths:
            raise ValueError("No RGB imagery found")
        mosaic = gdal.BuildVRT(str(root / "imagery.vrt"), vrt_paths)
        gt = mosaic.GetGeoTransform()
        image_bounds = [
            gt[0],
            gt[3] + gt[5] * mosaic.RasterYSize,
            gt[0] + gt[1] * mosaic.RasterXSize,
            gt[3],
        ]
        mosaic = None
        with zipfile.ZipFile(topography) as archive:
            names = [n for n in archive.namelist() if n.endswith(".gpkg")]
            if len(names) != 1:
                raise ValueError("Expected one MasterMap GeoPackage")
            with (
                archive.open(names[0]) as src,
                (root / "mastermap.gpkg").open("wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            contents = archive.read(
                next(n for n in archive.namelist() if n.startswith("contents_"))
            ).decode()
        extent = re.search(
            r"Extent:\s*([\d.]+),\s*([\d.]+)\s*-\s*([\d.]+),\s*([\d.]+)", contents
        )
        if not extent:
            raise ValueError("MasterMap extraction extent is missing")
        version = re.search(r"Version:\s*(.+)", contents).group(1).strip()
        source = pyogrio.read_dataframe(
            root / "mastermap.gpkg",
            layer="Topographicarea",
            where="featurecode = 10021",
        )
        if source.crs.to_epsg() != 27700 or not source.geometry.is_valid.all():
            raise ValueError("Expected valid BNG MasterMap buildings")
        if source.fid.duplicated().any():
            raise ValueError("Duplicate MasterMap TOIDs")
        source = source.rename(columns={"fid": "toid"})
        source["building_id"] = source.toid.str.removeprefix("osgb").astype("int64")
        source.to_file(root / "footprints.gpkg", layer="buildings", driver="GPKG")
        raw = root / "source.laz"
        if lidar.suffix.lower() == ".zip":
            with zipfile.ZipFile(lidar) as archive:
                names = [
                    n
                    for n in archive.namelist()
                    if n.lower().endswith((".las", ".laz"))
                ]
                if len(names) != 1:
                    raise ValueError("Expected one EA point cloud")
                with archive.open(names[0]) as src, raw.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(lidar, raw)
        tiles = root / "points"
        tiles.mkdir(exist_ok=True)
        writers = {}
        try:
            with laspy.open(raw) as reader:
                if reader.header.parse_crs().to_epsg() != 27700:
                    raise ValueError("Expected EPSG:27700 LiDAR")
                lidar_bounds = [
                    *np.floor(reader.header.mins[:2]),
                    *np.ceil(reader.header.maxs[:2]),
                ]
                for points in reader.chunk_iterator(1_000_000):
                    cells = np.floor(
                        np.column_stack((points.x, points.y)) / 1000
                    ).astype(int)
                    keep = np.isin(points.classification, RETAINED_CLASSES)
                    for cx, cy in np.unique(cells[keep], axis=0):
                        key = (int(cx), int(cy))
                        if key not in writers:
                            writers[key] = laspy.open(
                                tiles / f"{cx}_{cy}.laz", mode="w", header=reader.header
                            )
                        writers[key].write_points(
                            points[keep & (cells[:, 0] == cx) & (cells[:, 1] == cy)]
                        )
        finally:
            for writer in writers.values():
                writer.close()
        coverage = (
            box(*lidar_bounds)
            .intersection(box(*image_bounds))
            .intersection(box(*map(float, extent.groups())))
        )
        revision = hashlib.sha256(
            (sha256(imagery) + sha256(topography) + sha256(raw)).encode()
        ).hexdigest()[:16]
        write_json(
            root / "dataset.json",
            {
                "schema": "cambridge-source-v1",
                "revision": revision,
                "coverage": mapping(coverage),
                "lidar_bounds": lidar_bounds,
                "crs": "EPSG:27700",
                "vertical_crs": "source heights; vertical CRS not declared in LAS",
                "survey_year": 2023,
                "survey_date": "2023-02-13",
                "imagery_dates": dates,
                "topography_version": version,
                "source_sha256": sha256(raw),
            },
        )
        print(f"Installed Cambridge data at {root}")


class CambridgeProvider:
    name = "cambridge"
    crs = "EPSG:27700"

    def __init__(self, cache: Path, root: Path | None = None):
        self.root = root or source_root()
        self.cache = cache / "cambridge"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.metadata = json.loads((self.root / "dataset.json").read_text())
        self.binary = os.getenv("CITY3D_BINARY") or shutil.which("CLI_Emboss")
        binary_hash = (
            sha256(Path(self.binary))
            if self.binary and Path(self.binary).is_file()
            else "not-configured"
        )
        self.revision = hashlib.sha256(
            (self.metadata["revision"] + ":source-targets-v2:" + binary_hash).encode()
        ).hexdigest()[:16]
        self.vertical_crs = self.metadata["vertical_crs"]
        self.coverage = shapely.geometry.shape(self.metadata["coverage"])
        # Use the same selected datum operation as the vector/map coordinates.
        # Native GDAL and pyproj can otherwise choose different grid resources.
        project = Transformer.from_crs(self.crs, 3857, always_xy=True)
        project.transform(self.coverage.centroid.x, self.coverage.centroid.y)
        self.map_operation = project.get_last_used_operation().definition
        self.map_revision = hashlib.sha256(self.map_operation.encode()).hexdigest()[:12]

    def validate_selection(self, geometry):
        project = Transformer.from_crs(4326, self.crs, always_xy=True)
        if not self.coverage.covers(transform(project.transform, geometry)):
            raise ValueError(
                "Select a building or area inside the Cambridge coverage outline"
            )

    def house_id(self, building_fid):
        return f"uk-cam-{self.revision}-{int(building_fid)}"

    def footprints(self, bounds):
        return pyogrio.read_dataframe(
            self.root / "footprints.gpkg", layer="buildings", bbox=bounds
        )

    def selected_parts(self, source_ids):
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or len(source_ids) > 256
            or any(not isinstance(value, str) or not re.fullmatch(r"osgb\d+", value) for value in source_ids)
        ):
            raise ValueError("Select between 1 and 256 MasterMap source parts")
        ids = sorted(set(source_ids))
        frame = pyogrio.read_dataframe(
            self.root / "footprints.gpkg",
            layer="buildings",
            where="toid IN (" + ",".join(f"'{value}'" for value in ids) + ")",
        ).sort_values("toid")
        if set(frame.toid) != set(ids):
            raise ValueError("A selected MasterMap part is not in the installed source")
        if any(not self.coverage.covers(geometry) for geometry in frame.geometry):
            raise ValueError("A complete selected part extends beyond Cambridge source coverage")
        return frame

    def target_geometry(self, selection):
        """Resolve explicit part IDs on the server, never trust a client outline."""
        if "source_ids" not in selection:
            return None
        if selection.get("mode", "house") != "house":
            raise ValueError("Explicit source parts require building-parts mode")
        project = Transformer.from_crs(self.crs, 4326, always_xy=True)
        return transform(project.transform, shapely.union_all(self.selected_parts(selection["source_ids"]).geometry))

    @staticmethod
    def target_id(parts):
        ids = sorted(parts.toid)
        if len(ids) == 1:
            return int(parts.building_id.iloc[0])
        # Composite targets have a deterministic identity independent of click order.
        return int(hashlib.sha256("|".join(ids).encode()).hexdigest()[:15], 16)

    def source_neighbours(self, geometry, selected_ids):
        from .geometry import HOUSE_TOUCH_TOLERANCE_M

        nearby = self.footprints(geometry.buffer(HOUSE_TOUCH_TOLERANCE_M).bounds)
        return sorted(
            int(row.building_id)
            for row in nearby.itertuples()
            if row.toid not in selected_ids
            and row.geometry.distance(geometry) <= HOUSE_TOUCH_TOLERANCE_M
        )

    def preview_target(self, source_ids):
        parts = self.selected_parts(source_ids)
        geometry = shapely.union_all(parts.geometry)
        neighbours = self.source_neighbours(geometry, set(parts.toid))
        project = Transformer.from_crs(self.crs, 4326, always_xy=True)
        return {
            "scope": "selected_source_parts",
            "source_ids": list(parts.toid),
            "area_m2": float(geometry.area),
            "touching_source_ids": [f"osgb{value}" for value in neighbours],
            "geometry": mapping(transform(project.transform, geometry)),
            "unsupported_source_ids": [
                row.toid for row in parts.itertuples()
                if row.geometry.geom_type != "Polygon" or row.geometry.interiors
            ],
        }

    def map_tile(self, z, x, y):
        """Render the supplied imagery on Leaflet's grid; cache only derived tiles."""
        if not 12 <= z <= 20 or not (0 <= x < 2**z and 0 <= y < 2**z):
            raise ValueError("Invalid imagery tile")
        tile = (
            self.cache
            / "map"
            / self.metadata["revision"]
            / self.map_revision
            / str(z)
            / str(x)
            / f"{y}.png"
        )
        tile.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(tile) + ".lock"):
            if not tile.exists():
                half_world = 20037508.342789244
                span = 2 * half_world / 2**z
                bounds = (
                    -half_world + x * span,
                    half_world - (y + 1) * span,
                    -half_world + (x + 1) * span,
                    half_world - y * span,
                )
                ds = gdal.Warp(
                    "",
                    str(self.root / "imagery.vrt"),
                    format="MEM",
                    dstSRS="EPSG:3857",
                    coordinateOperation=self.map_operation,
                    outputBounds=bounds,
                    width=256,
                    height=256,
                    dstAlpha=True,
                    resampleAlg="bilinear",
                )
                array = ds.ReadAsArray().transpose(1, 2, 0)
                stream = io.BytesIO()
                Image.fromarray(array).save(stream, format="PNG")
                tile.write_bytes(stream.getvalue())
        return tile.read_bytes()

    def lidar(self, bounds, reference_year=None, progress=print, *, pinned_sources=()):
        if reference_year not in (None, 2023):
            raise ValueError("Cambridge contains the 2023 survey only")
        if not box(*self.metadata["lidar_bounds"]).covers(box(*bounds)):
            raise ValueError(
                "The complete requested LiDAR context extends beyond TL4055"
            )
        sources = []
        for cx in range(
            int(np.floor(bounds[0] / 1000)), int(np.ceil(bounds[2] / 1000))
        ):
            for cy in range(
                int(np.floor(bounds[1] / 1000)), int(np.ceil(bounds[3] / 1000))
            ):
                name = f"{cx}_{cy}.laz"
                target = self.cache / "points" / self.metadata["revision"] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(str(target) + ".lock"):
                    if not target.exists():
                        try:
                            target.hardlink_to(self.root / "points" / name)
                        except OSError:
                            shutil.copyfile(self.root / "points" / name, target)
                sources.append(
                    {
                        "tile": f"ea-2023-{cx}-{cy}",
                        "path": str(target),
                        "year": 2023,
                        "bounds": [
                            cx * 1000,
                            cy * 1000,
                            (cx + 1) * 1000,
                            (cy + 1) * 1000,
                        ],
                        "crs": self.crs,
                        "provider": self.name,
                        "revision": self.revision,
                        "source_sha256": self.metadata["source_sha256"],
                    }
                )
        if any(s.get("revision") != self.revision for s in pinned_sources):
            raise ValueError("Pinned Cambridge source revision changed")
        return tuple(Path(s["path"]) for s in sources), tuple(sources)

    def points_for_bounds(
        self, bounds, reference_year=None, progress=print, *, pinned_sources=()
    ):
        paths, sources = self.lidar(
            bounds, reference_year, progress, pinned_sources=pinned_sources
        )
        frames = [
            read_las_points(p, bounds_xy=bounds, class_ids=RETAINED_CLASSES)
            for p in paths
        ]
        return pd.concat(frames, ignore_index=True), sources

    def terrain(self, bounds, output, *, pinned_sources=()):
        from scipy.interpolate import NearestNDInterpolator

        frame, _ = self.points_for_bounds(bounds, pinned_sources=pinned_sources)
        ground = frame[frame.classification.eq(2)]
        if ground.empty:
            raise ValueError("No ground returns in Cambridge terrain crop")
        x0, y0, x1, y1 = bounds
        width, height = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
        xx, yy = np.meshgrid(x0 + np.arange(width) + 0.5, y1 - np.arange(height) - 0.5)
        values = NearestNDInterpolator(ground[["x", "y"]], ground.z)(xx, yy)
        ds = gdal.GetDriverByName("GTiff").Create(
            str(output), width, height, 1, gdal.GDT_Float32
        )
        ds.SetGeoTransform((x0, 1, 0, y1, 0, -1))
        ds.SetProjection(_srs().ExportToWkt())
        ds.GetRasterBand(1).WriteArray(values)
        ds = None
        return output

    def build_selection(self, geometry, mode, output, progress=print, *, source_ids=None):
        if source_ids is not None:
            parts = self.selected_parts(source_ids)
            return self._build_targets([parts], output, progress)
        footprints = self.footprints(
            geometry.buffer(30 if mode == "house" else 0).bounds
        )
        if mode == "house":
            distances = footprints.distance(geometry)
            footprints = footprints.loc[
                distances[distances <= 30].sort_values().index[:1]
            ]
        else:
            footprints = footprints[footprints.intersects(geometry)]
        return self._build_footprints(footprints, output, progress)

    def buildings(self, bounds, output, progress=print):
        return self._build_footprints(self.footprints(bounds), output, progress)

    def _build_footprints(self, footprints, output, progress):
        return self._build_targets(
            [footprints.loc[[index]] for index in footprints.index], output, progress
        )

    def _build_targets(self, targets, output, progress):
        from .city3d import scaffold_surfaces

        failures = []
        ready = []
        for parts in targets:
            fid = self.target_id(parts)
            selected_ids = sorted(parts.toid)
            geometry = shapely.union_all(parts.geometry)
            if not self.coverage.covers(geometry):
                failures.append(
                    {
                        "building_fid": fid,
                        "error": "The complete building extends beyond Cambridge source coverage",
                    }
                )
                continue
            try:
                surfaces = {kind: [] for kind in ("Roof", "Wall", "Floor")}
                for row in parts.itertuples():
                    progress(f"Preparing selected source part {row.toid}")
                    part_surfaces = scaffold_surfaces(self, row)
                    for kind, rings in surfaces.items():
                        rings.extend(part_surfaces[kind])
                projected_roof = shapely.union_all([
                    shapely.make_valid(Polygon(np.asarray(ring)[:, :2]))
                    for ring in surfaces["Roof"]
                ])
                coverage = {
                    "source_area_m2": float(geometry.area),
                    "roof_projection_area_m2": float(projected_roof.area),
                    "missing_area_m2": float(geometry.difference(projected_roof).area),
                    "extra_area_m2": float(projected_roof.difference(geometry).area),
                    "intersection_over_union": float(geometry.intersection(projected_roof).area / geometry.union(projected_roof).area),
                }
                ready.append((fid, selected_ids, surfaces, geometry,
                              self.source_neighbours(geometry, set(selected_ids)), coverage))
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append(
                    {"building_fid": fid, "error": str(error), **(
                        {"source_ids": selected_ids, "failed_source_id": row.toid}
                        if len(selected_ids) > 1 else {}
                    )}
                )
        temporary = output.with_suffix(".tmp.gpkg")
        temporary.unlink(missing_ok=True)
        ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(temporary))
        for kind in ("Roof", "Wall", "Floor"):
            layer = ds.CreateLayer(kind, _srs(), ogr.wkbMultiPolygon25D)
            layer.CreateField(ogr.FieldDefn("toid", ogr.OFTString))
            for fid, selected_ids, surfaces, _geometry, _neighbours, _coverage in ready:
                if not surfaces[kind]:
                    continue
                multi = ogr.Geometry(ogr.wkbMultiPolygon25D)
                for coordinates in surfaces[kind]:
                    polygon = ogr.CreateGeometryFromJson(
                        json.dumps(mapping(Polygon(coordinates)))
                    )
                    multi.AddGeometry(polygon)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetFID(fid)
                feature.SetField("toid", ";".join(selected_ids))
                feature.SetGeometry(multi)
                layer.CreateFeature(feature)
        layer = ds.CreateLayer("Target", _srs(), ogr.wkbUnknown)
        for field in ("scope", "source_ids", "touching_ids", "coverage"):
            layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))
        for fid, selected_ids, _surfaces, geometry, neighbours, coverage in ready:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetFID(fid)
            feature.SetField("scope", "selected_source_parts")
            feature.SetField("source_ids", json.dumps(selected_ids))
            feature.SetField("touching_ids", json.dumps(neighbours))
            feature.SetField("coverage", json.dumps(coverage))
            feature.SetGeometry(ogr.CreateGeometryFromJson(json.dumps(mapping(geometry))))
            layer.CreateFeature(feature)
        ds = None
        temporary.replace(output)
        write_json(output.with_suffix(".failures.json"), failures)
        return output

    def orthophoto(self, house, config):
        from .orthophoto_correction.models import CorrectionResult, DisplacementField

        if config.strip_id_override not in (None, "raw"):
            raise ValueError("Cambridge uses the supplied orthorectified imagery")
        x0, y0, x1, y1 = house.bounds_xy
        bounds = (
            np.floor((x0 - 8) * 4) / 4,
            np.floor((y0 - 8) * 4) / 4,
            np.ceil((x1 + 8) * 4) / 4,
            np.ceil((y1 + 8) * 4) / 4,
        )
        src = gdal.Open(str(self.root / "imagery.vrt"))
        gt = src.GetGeoTransform()
        image_box = box(
            gt[0],
            gt[3] + gt[5] * src.RasterYSize,
            gt[0] + gt[1] * src.RasterXSize,
            gt[3],
        )
        if not image_box.covers(box(*bounds)):
            raise ValueError("The full roof image crop extends beyond supplied imagery")
        cropped = gdal.Translate(
            "", src, format="MEM", projWin=[bounds[0], bounds[3], bounds[2], bounds[1]]
        )
        rgb = cropped.ReadAsArray().transpose(1, 2, 0)
        valid = np.ones(rgb.shape[:2], dtype=bool)
        metadata = {
            "method": "supplied_orthorectified",
            "crs": self.crs,
            "extent_lv95": list(bounds),
            "transform": cropped.GetGeoTransform(),
            "width": cropped.RasterXSize,
            "height": cropped.RasterYSize,
            "source_year": 2023,
            "source_revision": self.revision,
            "imagery_dates": self.metadata["imagery_dates"],
            "candidate_selection": {"selected_id": "raw", "candidate_scores": []},
        }
        zeros = np.zeros(rgb.shape[:2], dtype=np.float32)
        inactive = np.zeros_like(valid)
        displacement = DisplacementField(zeros, zeros, zeros, zeros, valid, inactive)
        return CorrectionResult(
            rgb,
            valid,
            inactive,
            inactive,
            displacement,
            bounds,
            cropped.GetGeoTransform(),
            metadata,
            rgb_raw=rgb,
        )


def _srs():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(27700)
    return srs
