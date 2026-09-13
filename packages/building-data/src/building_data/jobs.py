"""Small persistent local job queue with explicit per-item progress."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .storage import write_json


class JobQueue:
    def __init__(self, root: Path, workers: int = 1):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self.lock = threading.Lock()
        for path in self.root.glob("*.json"):
            state = json.loads(path.read_text())
            if state["status"] in ("queued", "running"):
                state.update(
                    status="interrupted",
                    error="Application stopped before this job completed. Start it again to reuse cached inputs.",
                )
                write_json(path, state)

    def get(self, job_id: str) -> dict:
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid job ID")
        return json.loads((self.root / f"{job_id}.json").read_text())

    def submit(self, function, *, description: str) -> dict:
        job_id = uuid.uuid4().hex
        state = {
            "id": job_id,
            "status": "queued",
            "description": description,
            "messages": [],
            "created_at": datetime.now(UTC).isoformat(),
            "result": None,
            "error": None,
        }
        path = self.root / f"{job_id}.json"
        write_json(path, state)

        def progress(message):
            with self.lock:
                state["messages"] = (state["messages"] + [str(message)])[-200:]
                write_json(path, state)

        def run():
            try:
                state["status"] = "running"
                write_json(path, state)
                result = function(progress)
                state.update(status="completed", result=result)
            except Exception as error:
                state.update(status="failed", error=str(error))
            with self.lock:
                write_json(path, state)

        self.executor.submit(run)
        return dict(state)
