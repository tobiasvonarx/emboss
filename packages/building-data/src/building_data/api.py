"""The same acquisition HTTP API and map picker for all three applications."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .jobs import JobQueue
from .store import AcquisitionStore, selection_geometry


class Selection(BaseModel):
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
            store.provider.validate_selection(geometry)
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
    async def search(q: str):
        if not 2 <= len(q) <= 200:
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
