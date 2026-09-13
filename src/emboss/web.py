"""Thin local HTTP interface over the public Emboss API."""

from __future__ import annotations

import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from building_data.api import mount_acquisition
from building_data.jobs import JobQueue
from building_data.orthophoto_correction.pipeline import CorrectionUnavailable
from building_data.runtime import data_directory, worker_count
from building_data.storage import contained_path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .api import Client
from .imagery_web import mount_imagery
from .library import BuildingBusyError, BuildingLibrary
from .mesh import read_ascii_ply_mesh
from .previews import (
    imagery_info,
    orthophoto_png,
    preview_candidate,
    segmentation_png,
)


class ReconstructionRequest(BaseModel):
    houses: list[str] = Field(min_length=1)
    force: bool = False
    strip_id_override: str | None = None
    reset_strip_override: bool = False
    image_choices: dict[str, str | None] = Field(default_factory=dict)


def create_app() -> FastAPI:
    client = Client(
        data_directory(), workers=worker_count(), device=os.getenv("DEVICE", "auto")
    )
    app = FastAPI(title="Emboss", version="0.1.0")
    app.state.client = client
    library = BuildingLibrary(client.store)
    app.state.library = library
    mount_acquisition(app, client.store)
    jobs = JobQueue(client.store.root / "jobs/reconstruction")
    mount_imagery(app, client, library=library)
    static = Path(__file__).with_name("web_static")
    app.mount("/assets", StaticFiles(directory=static), name="assets")

    def reserve(house_ids):
        try:
            return library.reserve(house_ids)
        except BuildingBusyError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    def submit(queue, house_ids, function, description):
        # Reserve synchronously, before the queue can leave work waiting to run.
        release = reserve(house_ids)

        def guarded(progress):
            try:
                return function(progress)
            finally:
                release()

        try:
            return queue.submit(guarded, description=description)
        except Exception:
            release()
            raise

    def library_action(action, *args):
        try:
            return action(*args)
        except BuildingBusyError as error:
            raise HTTPException(409, str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.delete("/api/houses/{house_id}")
    def remove_building(house_id: str):
        return library_action(library.remove, house_id)

    @app.get("/api/removed-buildings")
    def removed_buildings():
        return library_action(library.removed)

    @app.post("/api/removed-buildings/{undo_token}/restore")
    def restore_building(undo_token: str):
        return library_action(library.restore, undo_token)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "application": "emboss", "version": "0.1.0"}

    @app.get("/api/results")
    def results():
        output = []
        for house in client.store.list_houses():
            try:
                output.append({**house, **client.load_result(house["id"]).as_dict()})
            except (FileNotFoundError, ValueError):
                pass
        return output

    @app.post("/api/reconstructions", status_code=202)
    def reconstruct(request: ReconstructionRequest):
        ids = list(dict.fromkeys(request.houses))
        if not set(request.image_choices).issubset(ids):
            raise HTTPException(
                422, "Image choices must belong to the requested buildings"
            )
        if any(
            value is not None and not value.strip()
            for value in request.image_choices.values()
        ):
            raise HTTPException(
                422, "Image choices must be a source ID or null for automatic selection"
            )
        try:
            for house_id in ids:
                client.store.house(house_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

        def run(progress):
            results = []
            failures = []
            with ThreadPoolExecutor(max_workers=client.workers) as executor:
                futures = {
                    executor.submit(
                        client.reconstruct,
                        house_id,
                        force=request.force,
                        strip_id_override=request.image_choices.get(
                            house_id, request.strip_id_override
                        ),
                        reset_strip_override=(
                            request.image_choices[house_id] is None
                            if house_id in request.image_choices
                            else request.reset_strip_override
                        ),
                        progress=lambda message, h=house_id: progress(
                            f"{h}: {message}"
                        ),
                    ): house_id
                    for house_id in ids
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result().as_dict())
                    except Exception as error:  # noqa: BLE001 - preserve other houses when a worker fails
                        failures.append(
                            {"house_id": futures[future], "error": str(error)}
                        )
            if not results:
                raise RuntimeError("; ".join(f["error"] for f in failures))
            return {"results": results, "failures": failures}

        return submit(jobs, ids, run, f"Reconstruct {len(ids)} building(s)")

    @app.get("/api/reconstructions/{job_id}")
    def status(job_id: str):
        try:
            return jobs.get(job_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    def result(house_id):
        try:
            return client.load_result(house_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/results/{house_id}")
    def summary(house_id: str):
        return result(house_id).as_dict()

    @app.get("/api/results/{house_id}/mesh")
    def mesh(house_id: str):
        value = result(house_id)
        shape = read_ascii_ply_mesh(value.mesh_path)
        return {
            "vertices": shape.vertices,
            "faces": shape.faces,
            "details": json.loads(value.roof_details_path.read_text()),
        }

    @app.get("/api/results/{house_id}/imagery")
    def imagery(house_id: str):
        return imagery_info(result(house_id))

    def png(data):
        return Response(
            data, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/results/{house_id}/orthophoto.png")
    def orthophoto(house_id: str):
        return png(orthophoto_png(result(house_id)))

    @app.get("/api/results/{house_id}/segmentation.png")
    def segmentation(house_id: str):
        return png(segmentation_png(result(house_id)))

    @app.get("/api/results/{house_id}/imagery/preview")
    def preview(house_id: str, candidate_id: str):
        result(house_id)
        release = reserve([house_id])
        try:
            return png(preview_candidate(client, house_id, candidate_id))
        except (ValueError, FileNotFoundError, CorrectionUnavailable) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            release()

    @app.get("/api/results/{house_id}/artifacts/{artifact}")
    def artifact(house_id: str, artifact: str):
        value = result(house_id)
        files = json.loads(value.manifest_path.read_text())["artifacts"]
        if artifact not in files:
            raise HTTPException(404, "Unknown artifact")
        path = contained_path(value.root, files[artifact])
        return FileResponse(path, filename=path.name)

    @app.get("/api/results/{house_id}/download")
    def download(house_id: str):
        value = result(house_id)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(value.root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(value.root))
        return Response(
            stream.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{value.house_input.id}.zip"'
            },
        )

    return app
