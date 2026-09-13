"""Swisstopo source discovery for Emboss acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen


STAC_API_ROOT = "https://data.geo.admin.ch/api/stac/v1"
SEARCH_DELTA_DEG = 0.00005
SEARCH_LIMIT = 100


@dataclass(frozen=True)
class SourceSpec:
    collection: str
    extensions: tuple[str, ...]


SOURCE_SPECS = {
    "surfacePointCloud": SourceSpec(
        collection="ch.swisstopo.swisssurface3d",
        extensions=(".copc.laz", ".las.zip", ".las", ".laz"),
    ),
    "buildings": SourceSpec(
        collection="ch.swisstopo.swissbuildings3d_2",
        extensions=(".gpkg.zip", ".gdb.zip", ".dxf.zip", ".zip"),
    ),
}


@dataclass(frozen=True)
class SourceCandidate:
    dataset_key: str
    collection: str
    item_id: str
    asset_key: str
    asset_href: str
    file_name: str
    bbox: tuple[float, float, float, float] | None
    year: int | None
    variant: str
    media_type: str

    # These aliases keep the acquisition record independent of the web viewer's
    # schema while retaining the small interface used by the existing backend.
    @property
    def itemId(self) -> str:  # noqa: N802
        return self.item_id

    @property
    def assetHref(self) -> str:  # noqa: N802
        return self.asset_href

    @property
    def fileName(self) -> str:  # noqa: N802
        return self.file_name


def _get_json(url: str, *, params: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"Accept": "application/geo+json, application/json", "User-Agent": "emboss-acquisition/1"},
    )
    with urlopen(request, timeout=60.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _item_year(item: dict[str, Any]) -> int | None:
    datetime_value = str(item.get("properties", {}).get("datetime", "") or "")
    if match := re.search(r"\b(20\d{2})\b", datetime_value):
        return int(match.group(1))
    if match := re.search(r"_(20\d{2})(?:\D|$)", str(item.get("id", ""))):
        return int(match.group(1))
    return None


def _bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values = item.get("bbox")
    if not isinstance(values, list) or len(values) < 4:
        return None
    bounds = tuple(float(value) for value in values[:4])
    if not all(math.isfinite(value) for value in bounds):
        return None
    return bounds


def _matches_extension(href: str, extensions: tuple[str, ...]) -> bool:
    path = href.split("?", maxsplit=1)[0].lower()
    return any(path.endswith(extension) for extension in extensions)


def search_source_candidates(
    dataset_key: str,
    *,
    longitude: float,
    latitude: float,
) -> tuple[SourceCandidate, ...]:
    """Return every usable STAC asset intersecting a WGS84 location."""

    try:
        spec = SOURCE_SPECS[str(dataset_key)]
    except KeyError as exc:
        raise ValueError(f"Unknown acquisition source `{dataset_key}`.") from exc

    lon = float(longitude)
    lat = float(latitude)
    delta = float(SEARCH_DELTA_DEG)
    payload = _get_json(
        f"{STAC_API_ROOT}/collections/{spec.collection}/items",
        params={
            "bbox": f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}",
            "limit": SEARCH_LIMIT,
        },
    )
    candidates: list[SourceCandidate] = []
    for item in payload.get("features", []):
        item_bbox = _bbox(item)
        year = _item_year(item)
        for asset_key, asset in dict(item.get("assets", {})).items():
            href = str(asset.get("href", "") or "")
            if not href or not _matches_extension(href, spec.extensions):
                continue
            candidates.append(
                SourceCandidate(
                    dataset_key=str(dataset_key),
                    collection=spec.collection,
                    item_id=str(item.get("id", "unknown")),
                    asset_key=str(asset_key),
                    asset_href=href,
                    file_name=Path(href.split("?", maxsplit=1)[0]).name,
                    bbox=item_bbox,
                    year=year,
                    variant=str(asset.get("geoadmin:variant", "") or ""),
                    media_type=str(asset.get("type", "") or ""),
                )
            )
    return tuple(candidates)


def _contains(candidate: SourceCandidate, longitude: float, latitude: float) -> bool:
    if candidate.bbox is None:
        return False
    min_lon, min_lat, max_lon, max_lat = candidate.bbox
    return min_lon <= longitude <= max_lon and min_lat <= latitude <= max_lat


def _bbox_area(candidate: SourceCandidate) -> float:
    if candidate.bbox is None:
        return math.inf
    min_lon, min_lat, max_lon, max_lat = candidate.bbox
    return max(0.0, max_lon - min_lon) * max(0.0, max_lat - min_lat)


def _format_rank(candidate: SourceCandidate) -> int:
    name = candidate.file_name.lower()
    if candidate.dataset_key == "surfacePointCloud":
        return 0 if name.endswith(".copc.laz") else 1
    for rank, extension in enumerate((".gpkg.zip", ".gdb.zip", ".dxf.zip", ".zip")):
        if name.endswith(extension):
            return rank
    return 99


def candidate_rank(
    candidate: SourceCandidate,
    *,
    longitude: float,
    latitude: float,
) -> tuple[Any, ...]:
    """Rank analytic source assets without preferring nationwide archives."""

    contains_rank = 0 if _contains(candidate, float(longitude), float(latitude)) else 1
    year_rank = -(candidate.year or 0)
    if candidate.dataset_key == "buildings":
        tiled_rank = 0 if candidate.variant == "tiled" else 1
        return (contains_rank, tiled_rank, _bbox_area(candidate), _format_rank(candidate), year_rank, candidate.item_id)
    return (contains_rank, _format_rank(candidate), year_rank, _bbox_area(candidate), candidate.item_id)


def resolve_source_candidate(dataset_key: str, *, longitude: float, latitude: float) -> SourceCandidate:
    candidates = search_source_candidates(dataset_key, longitude=float(longitude), latitude=float(latitude))
    if not candidates:
        raise RuntimeError(f"No STAC candidates found for `{dataset_key}` at {latitude}, {longitude}.")
    return min(
        candidates,
        key=lambda candidate: candidate_rank(
            candidate,
            longitude=float(longitude),
            latitude=float(latitude),
        ),
    )
