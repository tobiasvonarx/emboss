"""The same acquisition HTTP API and map picker for all three applications."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .jobs import JobQueue
from .store import AcquisitionStore, selection_geometry


class Selection(BaseModel):
    provider: str | None = None
    mode: str = "house"
    longitude: float | None = None
    latitude: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    geometry: dict | None = None


def mount_acquisition(app: FastAPI, store: AcquisitionStore) -> None:
    static = Path(__file__).with_name("web")
    jobs = JobQueue(store.root / "jobs/acquisition")
    app.state.acquisition_store = store
    app.mount(
        "/acquisition-assets", StaticFiles(directory=static), name="acquisition-assets"
    )

    @app.get("/acquire", include_in_schema=False)
    def picker():
        return FileResponse(static / "index.html")

    @app.get("/api/houses")
    def houses():
        return store.list_houses()

    @app.get("/api/datasets")
    def datasets():
        from .datasets import describe

        return [describe(provider) for provider in store.providers.values()]

    @app.get("/api/datasets/{name}/footprints")
    def footprints(name: str, bbox: str):
        from pyproj import Transformer
        from shapely.geometry import box
        from shapely.ops import transform

        try:
            provider = store.provider_for(name)
            values = tuple(map(float, bbox.split(",")))
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                raise ValueError("Expected four finite bounding-box coordinates")
            if values[2] <= values[0] or values[3] <= values[1]:
                raise ValueError("Invalid bounding box")
            if not (
                -180 <= values[0] < values[2] <= 180
                and -90 <= values[1] < values[3] <= 90
            ):
                raise ValueError("Bounding box must use longitude and latitude")
            if not hasattr(provider, "footprints"):
                return {"type": "FeatureCollection", "features": []}
            project = Transformer.from_crs(4326, provider.crs, always_xy=True)
            query = transform(project.transform, box(*values)).intersection(
                provider.coverage
            )
            if query.is_empty or query.area > 2_000_000:
                return {"type": "FeatureCollection", "features": []}
            frame = provider.footprints(query.bounds)
            frame = frame[frame.intersects(query)]
            return json.loads(frame[["toid", "geometry"]].to_crs(4326).to_json())
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/datasets/{name}/tiles/{z}/{x}/{y}.png")
    def imagery_tile(name: str, z: int, x: int, y: int):
        try:
            provider = store.provider_for(name)
            if not hasattr(provider, "map_tile"):
                raise ValueError("This dataset has no local imagery tiles")
            return Response(provider.map_tile(z, x, y), media_type="image/png")
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/houses/{house_id}")
    def house(house_id: str):
        try:
            return store.house(house_id).as_dict()
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/acquisition", status_code=202)
    def acquire(selection: Selection):
        payload = selection.model_dump(exclude_none=True)
        try:
            _, geometry = selection_geometry(payload)
            store.provider_for(payload.get("provider")).validate_selection(geometry)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
        return jobs.submit(
            lambda progress: store.acquire(payload, progress),
            description="Acquire buildings",
        )

    @app.get("/api/acquisition/{job_id}")
    def status(job_id: str):
        try:
            return jobs.get(job_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/search")
    async def search(q: str, provider: str | None = None):
        if not 2 <= len(q) <= 200:
            return []
        try:
            selected = store.provider_for(provider)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if selected.name != "switzerland":
            # Local Cambridge names/TOIDs avoid a second external geocoder.
            return []

        def fetch():
            url = (
                "https://api3.geo.admin.ch/rest/services/api/SearchServer?"
                + urlencode(
                    {
                        "searchText": q,
                        "type": "locations",
                        "origins": "address",
                        "limit": 8,
                        "sr": 4326,
                    }
                )
            )
            with urlopen(
                Request(url, headers={"User-Agent": "building-data/0.1"}), timeout=30
            ) as response:
                data = json.load(response)
            return [
                {
                    "label": entry["attrs"]["label"],
                    "longitude": entry["attrs"]["lon"],
                    "latitude": entry["attrs"]["lat"],
                }
                for entry in data.get("results", [])
            ]

        try:
            return await asyncio.to_thread(fetch)
        except Exception as error:
            raise HTTPException(
                502, "Address search is unavailable; select on the map."
            ) from error
