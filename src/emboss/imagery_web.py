"""Reusable imagery review endpoints and gallery assets for Emboss applications.

Mount alongside building-data acquisition; all paths and caches belong to the
supplied Client. No separate Emboss server or external workspace is required.
"""

from pathlib import Path

from building_data.jobs import JobQueue
from building_data.orthophoto_correction.pipeline import CorrectionUnavailable
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .library import BuildingBusyError, BuildingLibrary
from .previews import house_preview_candidate, prepare_house_imagery


def mount_imagery(app, client, *, library=None):
    """Mount /api imagery routes and /emboss-imagery/{gallery.js,style.css}.

    Pass the host's BuildingLibrary when it offers building removal so imagery
    preparation and preview requests participate in the same activity guard.
    """
    library = library or BuildingLibrary(client.store)
    imagery_jobs = JobQueue(client.store.root / "jobs/imagery", workers=client.workers)
    app.mount(
        "/emboss-imagery",
        StaticFiles(directory=Path(__file__).with_name("imagery_static")),
        name="emboss-imagery",
    )

    def reserve(house_ids):
        try:
            return library.reserve(house_ids)
        except BuildingBusyError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    def submit(queue, house_ids, function, description):
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

    def png(data):
        return Response(
            data, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/houses/{house_id}/imagery", status_code=202)
    def prepare_imagery(house_id: str):
        try:
            client.store.house(house_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error
        return submit(
            imagery_jobs,
            [house_id],
            lambda progress: prepare_house_imagery(client, house_id, progress=progress),
            f"Prepare imagery for {house_id}",
        )

    @app.get("/api/imagery-jobs/{job_id}")
    def imagery_status(job_id: str):
        try:
            return imagery_jobs.get(job_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/houses/{house_id}/imagery/preview")
    def house_preview(house_id: str, candidate_id: str):
        try:
            client.store.house(house_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error
        release = reserve([house_id])
        try:
            return png(house_preview_candidate(client, house_id, candidate_id))
        except FileNotFoundError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, CorrectionUnavailable) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            release()
