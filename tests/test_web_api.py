"""HTTP job and artifact contract tests with real storage and no model inference."""

from __future__ import annotations

import emboss.imagery_web as imagery_web

import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import tifffile
from building_data.jobs import JobQueue
from fastapi.testclient import TestClient
from PIL import Image

from emboss import web
from emboss.api import Client, Reconstruction


@pytest.fixture
def web_fixture(tmp_path, monkeypatch):
    client = Client(tmp_path, workers=2)
    calls = []
    release = threading.Event()
    for house_id in ("good-house", "failed-house"):
        house = tmp_path / "houses" / house_id
        house.mkdir(parents=True)
        (house / "house.json").write_text(
            json.dumps(
                {
                    "schema": "building-input-v1",
                    "id": house_id,
                    "units": "m",
                    "building_fid": 1,
                    "surfaces_path": "surfaces.gpkg",
                    "sources": [],
                    "bounds_xy": [0.0, 0.0, 1.0, 1.0],
                    "crs": "EPSG:2056",
                }
            )
        )
    (tmp_path / "surfaces.gpkg").write_bytes(b"unused by HTTP contract fixture")

    def reconstruct(
        house_id,
        *,
        force=False,
        strip_id_override=None,
        reset_strip_override=False,
        progress=None,
    ):
        calls.append((house_id, force, strip_id_override, reset_strip_override))
        progress("Fixture processing started")
        assert release.wait(5), "Test did not release asynchronous reconstruction"
        if house_id == "failed-house":
            raise RuntimeError("Fixture has no usable LiDAR returns")
        root = tmp_path / "results" / house_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "mesh.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 1\n1 0 1\n0 1 1\n3 0 1 2\n"
        )
        (root / "roof_details.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": []})
        )
        (root / "result.json").write_text(json.dumps({"solid_count": 1}))
        for filename in (
            "prediction.json",
            "orthophoto.tif",
            "vector_house.geojson",
            "segmentation.tif",
        ):
            (root / filename).write_bytes(b"unused by HTTP contract fixture")
        (root / "bundle.json").write_text(
            json.dumps(
                {
                    "schema": "emboss-bundle-v1",
                    "artifacts": {
                        "mesh": "mesh.ply",
                        "roof_details": "roof_details.geojson",
                        "result": "result.json",
                        "prediction": "prediction.json",
                        "orthophoto": "orthophoto.tif",
                        "vector_house": "vector_house.geojson",
                        "segmentation": "segmentation.tif",
                    },
                    "provenance": {"fixture": True},
                }
            )
        )
        return Reconstruction(root, root / "bundle.json", client.store.house(house_id))

    monkeypatch.setattr(client, "reconstruct", reconstruct)
    monkeypatch.setattr(web, "Client", lambda *args, **kwargs: client)
    queues = []
    original_queue = web.JobQueue

    def queue(*args, **kwargs):
        result = original_queue(*args, **kwargs)
        queues.append(result)
        return result

    monkeypatch.setattr(web, "JobQueue", queue)
    monkeypatch.setattr(imagery_web, "JobQueue", queue)
    with TestClient(web.create_app()) as http:
        yield http, client, calls, release
    release.set()
    for queue in queues:
        queue.executor.shutdown(wait=True)


def completed_job(http, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = http.get(f"/api/reconstructions/{job_id}")
        assert response.status_code == 200
        result = response.json()
        if result["status"] in ("completed", "failed"):
            return result
        time.sleep(0.01)
    pytest.fail("Asynchronous job did not complete")


def imagery_job(http, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = http.get(f"/api/imagery-jobs/{job_id}")
        assert response.status_code == 200
        state = response.json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.01)
    pytest.fail("Imagery job did not complete")


def test_remove_list_and_restore_http_contract(web_fixture):
    http, client, _calls, _release = web_fixture
    before = (client.store.house("good-house").root / "house.json").read_bytes()
    removed = http.delete("/api/houses/good-house")
    assert removed.status_code == 200
    receipt = removed.json()
    assert receipt["house_id"] == "good-house"
    assert http.get("/api/removed-buildings").json() == [receipt]
    assert http.delete("/api/houses/good-house").status_code == 404
    restored = http.post(f"/api/removed-buildings/{receipt['undo_token']}/restore")
    assert restored.status_code == 200 and restored.json() == {"house_id": "good-house"}
    assert (client.store.house("good-house").root / "house.json").read_bytes() == before
    assert http.get("/api/removed-buildings").json() == []
    assert http.post("/api/removed-buildings/invalid/restore").status_code == 422


@pytest.mark.parametrize("kind", ["reconstruction", "imagery"])
def test_queued_work_reserves_building_before_worker_starts(
    web_fixture, monkeypatch, kind
):
    http, _client, _calls, release = web_fixture
    queued = []

    def hold(_queue, function, *, description):
        queued.append(function)
        return {"id": "a" * 32, "status": "queued"}

    monkeypatch.setattr(JobQueue, "submit", hold)
    if kind == "reconstruction":
        response = http.post("/api/reconstructions", json={"houses": ["good-house"]})
    else:
        monkeypatch.setattr(
            imagery_web,
            "prepare_house_imagery",
            lambda *_args, **_kwargs: {"house_id": "good-house"},
        )
        response = http.post("/api/houses/good-house/imagery")
    assert response.status_code == 202 and len(queued) == 1
    assert http.delete("/api/houses/good-house").status_code == 409
    release.set()
    queued[0](lambda _message: None)
    assert http.delete("/api/houses/good-house").status_code == 200


def test_failed_queue_submission_releases_building_reservation(
    web_fixture, monkeypatch
):
    http, _client, _calls, _release = web_fixture

    def broken(*_args, **_kwargs):
        raise RuntimeError("Queue unavailable")

    monkeypatch.setattr(JobQueue, "submit", broken)
    with pytest.raises(RuntimeError, match="Queue unavailable"):
        http.post("/api/houses/good-house/imagery")
    assert http.delete("/api/houses/good-house").status_code == 200


def test_failed_worker_releases_building_reservation(web_fixture, monkeypatch):
    http, _client, _calls, _release = web_fixture

    def broken(*_args, **_kwargs):
        raise RuntimeError("No image source")

    monkeypatch.setattr(imagery_web, "prepare_house_imagery", broken)
    response = http.post("/api/houses/good-house/imagery")
    assert imagery_job(http, response.json()["id"])["status"] == "failed"
    assert http.delete("/api/houses/good-house").status_code == 200


def test_active_candidate_preview_blocks_removal_until_response_ready(
    web_fixture, monkeypatch
):
    http, _client, _calls, _release = web_fixture
    started, finish = threading.Event(), threading.Event()

    def preview(*_args):
        started.set()
        assert finish.wait(5)
        return b"preview-png"

    monkeypatch.setattr(imagery_web, "house_preview_candidate", preview)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            http.get,
            "/api/houses/good-house/imagery/preview",
            params={"candidate_id": "raw"},
        )
        try:
            assert started.wait(5)
            assert http.delete("/api/houses/good-house").status_code == 409
        finally:
            finish.set()
        assert future.result().status_code == 200
    assert http.delete("/api/houses/good-house").status_code == 200


def test_acquired_house_gallery_is_async_before_any_reconstruction(
    web_fixture, monkeypatch
):
    http, _client, reconstruction_calls, _release = web_fixture
    started, finish = threading.Event(), threading.Event()
    info = {
        "house_id": "good-house",
        "selected_id": "raw",
        "recommended_id": "raw",
        "has_reconstruction": False,
        "candidates": [{"id": "raw", "kind": "raw"}],
    }

    def prepare(client, house_id, *, progress):
        assert house_id == "good-house"
        progress("Checking aerial images")
        started.set()
        assert finish.wait(5)
        return info

    monkeypatch.setattr(imagery_web, "prepare_house_imagery", prepare)
    response = http.post("/api/houses/good-house/imagery")
    assert response.status_code == 202
    assert started.wait(5)
    state = http.get("/api/imagery-jobs/" + response.json()["id"]).json()
    assert state["status"] == "running"
    assert not reconstruction_calls
    finish.set()
    state = imagery_job(http, response.json()["id"])
    assert state["result"] == info and state["status"] == "completed"
    assert state["messages"] == ["Checking aerial images"]
    monkeypatch.setattr(
        imagery_web, "house_preview_candidate", lambda *_: b"test-png-bytes"
    )
    png = http.get(
        "/api/houses/good-house/imagery/preview", params={"candidate_id": "raw"}
    )
    assert png.status_code == 200 and png.content == b"test-png-bytes"
    assert png.headers["content-type"] == "image/png"
    assert not reconstruction_calls


def test_gallery_reports_failed_preparation_and_invalid_requests(
    web_fixture, monkeypatch
):
    http, _client, calls, _release = web_fixture

    def fail(*args, **kwargs):
        raise RuntimeError("No aerial images available")

    monkeypatch.setattr(imagery_web, "prepare_house_imagery", fail)
    response = http.post("/api/houses/good-house/imagery")
    state = imagery_job(http, response.json()["id"])
    assert state["status"] == "failed" and "No aerial images" in state["error"]
    assert http.post("/api/houses/missing/imagery").status_code == 404
    assert http.get("/api/imagery-jobs/invalid").status_code == 404
    for error, status in [
        (FileNotFoundError("Prepare first"), 409),
        (ValueError("Unknown source"), 422),
    ]:

        def preview(*args, error=error):
            raise error

        monkeypatch.setattr(imagery_web, "house_preview_candidate", preview)
        assert (
            http.get(
                "/api/houses/good-house/imagery/preview", params={"candidate_id": "bad"}
            ).status_code
            == status
        )
    assert not calls


def test_batch_uses_each_buildings_chosen_image_and_explicit_automatic_reset(
    web_fixture,
):
    http, _client, calls, release = web_fixture
    response = http.post(
        "/api/reconstructions",
        json={
            "houses": ["good-house", "failed-house"],
            "strip_id_override": "global-strip",
            "reset_strip_override": True,
            "image_choices": {"good-house": "chosen-strip", "failed-house": None},
        },
    )
    assert response.status_code == 202
    release.set()
    assert completed_job(http, response.json()["id"])["status"] == "completed"
    assert set(calls) == {
        ("good-house", False, "chosen-strip", False),
        ("failed-house", False, None, True),
    }


@pytest.mark.parametrize("choices", [{"missing-house": "strip"}, {"good-house": " "}])
def test_invalid_image_choices_do_not_start_reconstruction(web_fixture, choices):
    http, _client, calls, _release = web_fixture
    response = http.post(
        "/api/reconstructions",
        json={"houses": ["good-house"], "image_choices": choices},
    )
    assert response.status_code == 422
    assert not calls


def test_async_batch_preserves_success_reports_failure_and_deduplicates(web_fixture):
    http, _client, calls, release = web_fixture
    response = http.post(
        "/api/reconstructions",
        json={
            "houses": ["good-house", "failed-house", "good-house"],
            "force": True,
            "strip_id_override": "strip-a",
            "reset_strip_override": True,
        },
    )
    assert response.status_code == 202
    assert response.json()["status"] in ("queued", "running")
    release.set()
    state = completed_job(http, response.json()["id"])
    assert state["status"] == "completed"
    assert [result["house_id"] for result in state["result"]["results"]] == [
        "good-house"
    ]
    assert state["result"]["failures"] == [
        {"house_id": "failed-house", "error": "Fixture has no usable LiDAR returns"}
    ]
    assert sorted(calls) == [
        ("failed-house", True, "strip-a", True),
        ("good-house", True, "strip-a", True),
    ]
    assert any("good-house:" in message for message in state["messages"])
    assert any("failed-house:" in message for message in state["messages"])
    summaries = http.get("/api/results").json()
    assert len(summaries) == 1 and summaries[0]["house_id"] == "good-house"


def test_all_failed_batch_is_observable_as_failed_job(web_fixture):
    http, _client, _calls, release = web_fixture
    response = http.post("/api/reconstructions", json={"houses": ["failed-house"]})
    release.set()
    state = completed_job(http, response.json()["id"])
    assert state["status"] == "failed"
    assert "no usable LiDAR" in state["error"]
    assert state["result"] is None


def test_mesh_artifact_and_portable_download_contract(web_fixture):
    http, client, _calls, release = web_fixture
    release.set()
    response = http.post("/api/reconstructions", json={"houses": ["good-house"]})
    assert completed_job(http, response.json()["id"])["status"] == "completed"
    assert http.get("/api/health").json()["application"] == "emboss"
    summary = http.get("/api/results/good-house")
    assert summary.status_code == 200 and summary.json()["solid_count"] == 1
    mesh = http.get("/api/results/good-house/mesh").json()
    assert mesh["vertices"] == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    assert mesh["faces"] == [[0, 1, 2]]
    artifact = http.get("/api/results/good-house/artifacts/mesh")
    assert artifact.status_code == 200
    assert artifact.content == (client.load_result("good-house").mesh_path).read_bytes()
    assert http.get("/api/results/good-house/artifacts/not-listed").status_code == 404
    root = client.load_result("good-house").root
    (root / "do-not-export").symlink_to(client.store.root / "surfaces.gpkg")
    download = http.get("/api/results/good-house/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert "good-house.zip" in download.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert "mesh.ply" in archive.namelist()
        assert "bundle.json" in archive.namelist()
        assert "do-not-export" not in archive.namelist()
        assert all(
            not Path(name).is_absolute() and ".." not in Path(name).parts
            for name in archive.namelist()
        )
        assert archive.read("mesh.ply") == artifact.content


@pytest.mark.parametrize("house_id", ["missing-house", "../outside", "bad id", ""])
def test_invalid_or_missing_house_ids_do_not_schedule_jobs(web_fixture, house_id):
    http, _client, calls, _release = web_fixture
    response = http.post("/api/reconstructions", json={"houses": [house_id]})
    assert response.status_code == 404
    assert calls == []


def test_incomplete_result_does_not_break_result_listing(web_fixture):
    http, client, _, release = web_fixture
    release.set()
    response = http.post("/api/reconstructions", json={"houses": ["good-house"]})
    assert completed_job(http, response.json()["id"])["status"] == "completed"
    client.load_result("good-house").mesh_path.unlink()
    assert http.get("/api/results").json() == []
    manifest = client.store.root / "results/good-house/bundle.json"
    manifest.write_text("{")
    assert http.get("/api/results").json() == []


def test_saved_imagery_preview_endpoints(web_fixture):
    http, client, _, release = web_fixture
    release.set()
    job = http.post("/api/reconstructions", json={"houses": ["good-house"]}).json()
    assert completed_job(http, job["id"])["status"] == "completed"
    root = client.load_result("good-house").root
    (root / "inputs").mkdir()
    (root / "inputs/orthophoto_correction.json").write_text(
        json.dumps(
            {
                "source_year": 2023,
                "strip_candidates": [{"id": "strip-a", "flight_date": "2023-05-29"}],
                "candidate_selection": {"selected_id": "strip-a"},
            }
        )
    )
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    labels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    tifffile.imwrite(root / "orthophoto.tif", rgb)
    tifffile.imwrite(root / "segmentation.tif", labels)
    source_bytes = (root / "orthophoto.tif").read_bytes()
    base = "/api/results/good-house"
    info = http.get(base + "/imagery").json()
    assert info["selected_id"] == "strip-a"
    assert {row["id"] for row in info["candidates"]} == {"raw", "strip-a"}
    photo = http.get(base + "/orthophoto.png")
    assert photo.headers["content-type"] == "image/png"
    assert photo.headers["cache-control"] == "no-store"
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(photo.content))), rgb)
    overlay = http.get(base + "/segmentation.png")
    pixels = np.array(Image.open(io.BytesIO(overlay.content)))
    assert pixels.shape == (2, 3, 4)
    assert pixels[labels == 5, 3].tolist() == [0]
    assert (pixels[labels < 5, 3] > 0).all()
    assert (
        http.get(base + "/imagery/preview?candidate_id=strip-a").content
        == photo.content
    )
    assert http.get(base + "/imagery/preview?candidate_id=unknown").status_code == 422
    assert http.get("/api/results/missing/orthophoto.png").status_code == 404
    assert (root / "orthophoto.tif").read_bytes() == source_bytes


def test_missing_results_invalid_jobs_and_empty_batch(web_fixture):
    http, _client, _calls, _release = web_fixture
    assert http.get("/api/results/missing-house").status_code == 404
    assert http.get("/api/results/good-house").status_code == 404
    assert http.get("/api/reconstructions/not-a-job").status_code == 404
    assert http.get("/api/reconstructions/" + "a" * 32).status_code == 404
    assert http.post("/api/reconstructions", json={"houses": []}).status_code == 422
    assert http.get("/").status_code == 200
    assert http.get("/acquire").status_code == 200


def test_imagery_mount_works_in_an_independent_host(web_fixture, monkeypatch):
    """Downstream hosts need only a local Client, not the Emboss application."""
    from fastapi import FastAPI

    _, client, _, _ = web_fixture
    app = FastAPI()
    monkeypatch.setattr(
        imagery_web,
        "prepare_house_imagery",
        lambda *_args, **_kwargs: {
            "selected_id": "strip-a",
            "candidates": [{"id": "strip-a"}],
        },
    )
    monkeypatch.setattr(
        imagery_web, "house_preview_candidate", lambda *_args: b"png-preview"
    )
    imagery_web.mount_imagery(app, client)
    with TestClient(app) as http:
        response = http.post("/api/houses/good-house/imagery")
        assert response.status_code == 202
        job_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = http.get(f"/api/imagery-jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.01)
        assert job["result"]["selected_id"] == "strip-a"
        assert (
            http.get(
                "/api/houses/good-house/imagery/preview?candidate_id=strip-a"
            ).content
            == b"png-preview"
        )
        assert http.post("/api/houses/missing/imagery").status_code == 404
        assert (
            "export class ImageGallery" in http.get("/emboss-imagery/gallery.js").text
        )
        assert ".footprint-overlay" in http.get("/emboss-imagery/style.css").text
        assert http.get("/api/reconstructions").status_code == 404
