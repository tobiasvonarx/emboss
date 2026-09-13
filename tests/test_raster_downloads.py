"""Complete downloads are checked without changing GDAL sampling semantics."""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
import pytest
from building_data import storage
from building_data.orthophoto_correction.models import RasterAsset
from building_data.orthophoto_correction.raster import read_raster_crop, write_geotiff


class Response(BytesIO):
    def __init__(self, data, *, status=200, length=None, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data) if length is None else length)}
        if content_range is not None:
            self.headers["Content-Range"] = content_range


def test_download_retries_truncation_and_checks_hash(tmp_path, monkeypatch):
    payload = b"complete raster bytes"
    responses = [Response(b"partial", length=len(payload)), Response(payload)]
    calls = []

    def fetch(request, **kwargs):
        calls.append(request)
        return responses.pop(0)

    monkeypatch.setattr(storage, "urlopen", fetch)
    path = storage.download(
        "https://example.test/source.tif",
        tmp_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        retry_delay_s=0,
    )
    assert path.read_bytes() == payload
    assert len(calls) == 2
    assert storage.download("https://example.test/source.tif", tmp_path) == path
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        storage.download("https://example.test/source.tif", tmp_path)


def test_download_rejects_partial_206_but_accepts_complete_206(tmp_path, monkeypatch):
    responses = [
        Response(b"abc", status=206, content_range="bytes 0-2/6"),
        Response(b"abcdef", status=206, content_range="bytes 0-5/6"),
    ]
    monkeypatch.setattr(storage, "urlopen", lambda *args, **kwargs: responses.pop(0))
    assert (
        storage.download(
            "https://example.test/a.tif", tmp_path, retry_delay_s=0
        ).read_bytes()
        == b"abcdef"
    )


def test_download_retries_503_and_bounds_checksum_failures(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("https://example.test/a.tif", 503, "unavailable", {}, None)
        return Response(b"wrong")

    monkeypatch.setattr(storage, "urlopen", fetch)
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        storage.download(
            "https://example.test/a.tif",
            tmp_path,
            sha256="0" * 64,
            attempts=3,
            retry_delay_s=0,
        )
    assert len(calls) == 3
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("a.tif"))


def test_stac_multihash_sha256():
    digest = "a" * 64
    assert storage.stac_sha256("1220" + digest) == digest
    assert storage.stac_sha256("sha256:" + digest) == digest
    assert storage.stac_sha256(digest) == digest
    assert storage.stac_sha256("1114" + "a" * 40) is None


@contextmanager
def raster_server(payload):
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            requested = self.headers.get("Range")
            start, end = 0, len(payload) - 1
            if requested:
                first, last = requested.removeprefix("bytes=").split("-")
                start = int(first)
                end = min(int(last) if last else end, end)
            self.send_response(206 if requested else 200)
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(payload[start : end + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/source.tif"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cached_source_io_is_bitwise_equal_to_remote_warp_without_grid_change(tmp_path):
    source = tmp_path / "source.tif"
    pixels = np.random.default_rng(44).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    write_geotiff(
        source, pixels, bounds_lv95=(2600000.0, 1200000.0, 2600003.2, 1200003.2)
    )
    bounds = (2600000.13, 1200000.17, 2600002.74, 1200002.84)
    with raster_server(source.read_bytes()) as url:
        asset = RasterAsset("fixture", "fixture", "rgb", url, 0.1, 2024)
        expected, expected_extent = read_raster_crop(
            [asset], bounds_lv95=bounds, gsd_m=0.1
        )
        downloaded = storage.download(
            url, tmp_path / "cache", sha256=storage.checksum(source)
        )
        actual, actual_extent = read_raster_crop(
            [asset], bounds_lv95=bounds, gsd_m=0.1, source_paths={url: downloaded}
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual_extent == expected_extent
        assert asset.href == url
        # The rejected alternative really changes the pixel grid on this case.
        from dataclasses import replace

        _, wrong_extent = read_raster_crop(
            [replace(asset, href=str(downloaded))], bounds_lv95=bounds, gsd_m=0.1
        )
        assert wrong_extent != expected_extent


def test_atomic_json_encodes_nonfinite_diagnostics_without_mutating_computation(
    tmp_path,
):
    import json

    original = {
        "candidate_scores": [{"mean_strip_distance_m": float("inf"), "roof_iou": 0.5}],
        "numpy_values": np.array([1.0, np.nan]),
    }
    target = tmp_path / "metadata.json"
    storage.write_json(target, original)
    saved = json.loads(target.read_text())
    assert saved["candidate_scores"] == [
        {"mean_strip_distance_m": None, "roof_iou": 0.5}
    ]
    assert saved["numpy_values"] == [1.0, None]
    assert np.isinf(original["candidate_scores"][0]["mean_strip_distance_m"])
    assert not list(tmp_path.glob("*.tmp"))
