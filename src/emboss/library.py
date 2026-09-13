"""Reversible removal of building-owned inputs and results."""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime

from building_data.storage import write_json
from filelock import FileLock, Timeout


class BuildingBusyError(RuntimeError):
    """A building is active or a restore would replace existing data."""


class BuildingLibrary:
    def __init__(self, store):
        self.store = store
        self.root = store.root
        self._mutex = threading.RLock()
        self._active = Counter()
        self._exclusive = set()

    @staticmethod
    def _house_id(value):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
            raise ValueError("Invalid building ID")
        return value

    def _path(self, relative):
        path = self.root / relative
        if not path.is_relative_to(self.root):
            raise ValueError("Path is outside the library")
        for parent in (path, *path.parents):
            if parent == self.root.parent:
                break
            if parent.is_symlink():
                raise ValueError(
                    "Symbolic links are not supported in building archives"
                )
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Path is outside the library")
        return path

    def _tree(self, path):
        self._path(path.relative_to(self.root))
        if path.exists() and not path.is_dir():
            raise ValueError("Expected a building directory")
        if any(child.is_symlink() for child in path.rglob("*")):
            raise ValueError("Symbolic links are not supported in building archives")

    def reserve(self, house_ids):
        """Reserve before queue submission; the returned release is idempotent."""
        ids = tuple(dict.fromkeys(house_ids))
        with self._mutex:
            for house_id in ids:
                self._house_id(house_id)
                if house_id in self._exclusive:
                    raise BuildingBusyError("The building is being removed or restored")
                self.store.house(house_id)
            self._active.update(ids)
        released = False

        def release():
            nonlocal released
            with self._mutex:
                if not released:
                    self._active.subtract(ids)
                    self._active += Counter()
                    released = True

        return release

    @contextmanager
    def activity(self, house_ids):
        release = self.reserve(house_ids)
        try:
            yield
        finally:
            release()

    @contextmanager
    def _changing(self, house_id):
        self._house_id(house_id)
        with self._mutex:
            if self._active[house_id] or house_id in self._exclusive:
                raise BuildingBusyError(
                    "Wait for this building's active work to finish"
                )
            self._exclusive.add(house_id)
        try:
            yield
        finally:
            with self._mutex:
                self._exclusive.remove(house_id)

    @contextmanager
    def _result_lock(self, house_id):
        path = self._path(f"results/{house_id}.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(path, timeout=0)
        try:
            lock.acquire()
        except Timeout as error:
            raise BuildingBusyError(
                "The building is currently being reconstructed"
            ) from error
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _move_all(pairs):
        moved = []
        try:
            for source, destination in pairs:
                source.rename(destination)
                moved.append((source, destination))
        except OSError:
            for source, destination in reversed(moved):
                destination.rename(source)
            raise

    def remove(self, house_id):
        with self._changing(house_id), self._result_lock(house_id):
            house = self._path(f"houses/{house_id}")
            result = self._path(f"results/{house_id}")
            self._tree(house)
            self._tree(result)
            item = self.store.house(house_id)
            token = uuid.uuid4().hex
            archive = self._path(f"trash/buildings/{token}")
            archive.mkdir(parents=True)
            receipt = {
                "schema": "emboss-removed-building-v1",
                "house_id": house_id,
                "building_fid": item.building_fid,
                "undo_token": token,
                "removed_at": datetime.now(UTC).isoformat(),
                "has_result": result.exists(),
            }
            write_json(archive / "removed.json", receipt)
            pairs = [(house, archive / "house")]
            if result.exists():
                pairs.append((result, archive / "result"))
            try:
                with (
                    FileLock(self._path(f"houses/{house_id}/prepare.lock"), timeout=0),
                    FileLock(self._path(f"houses/{house_id}/imagery.lock"), timeout=0),
                ):
                    self._move_all(pairs)
            except (OSError, Timeout) as error:
                # Rollback leaves the original directories in place. If rollback
                # itself failed, retain the receipt and remaining archived files.
                if (
                    not (archive / "house").exists()
                    and not (archive / "result").exists()
                ):
                    (archive / "removed.json").unlink()
                    archive.rmdir()
                if isinstance(error, Timeout):
                    raise BuildingBusyError(
                        "The building's inputs or imagery are being prepared"
                    ) from error
                raise
            return self._summary(receipt)

    def _archive(self, token):
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("Invalid undo token")
        archive = self._path(f"trash/buildings/{token}")
        self._tree(archive)
        receipt = json.loads((archive / "removed.json").read_text())
        if (
            receipt.get("schema") != "emboss-removed-building-v1"
            or receipt.get("undo_token") != token
        ):
            raise ValueError("Invalid building archive")
        self._house_id(receipt["house_id"])
        if not (archive / "house").is_dir():
            raise FileNotFoundError("This building has already been restored")
        if receipt["has_result"] and not (archive / "result").is_dir():
            raise ValueError("The building archive is incomplete")
        return archive, receipt

    @staticmethod
    def _summary(receipt):
        return {
            name: receipt[name]
            for name in ("house_id", "building_fid", "undo_token", "removed_at")
        }

    def removed(self):
        root = self._path("trash/buildings")
        rows = []
        for path in root.glob("*/removed.json"):
            try:
                _archive, receipt = self._archive(path.parent.name)
                rows.append(self._summary(receipt))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda row: row["removed_at"], reverse=True)

    def restore(self, token):
        archive, receipt = self._archive(token)
        house_id = receipt["house_id"]
        with self._changing(house_id), self._result_lock(house_id):
            archive, receipt = self._archive(token)
            house = self._path(f"houses/{house_id}")
            result = self._path(f"results/{house_id}")
            if house.exists() or result.exists():
                raise BuildingBusyError(
                    "A building with this ID already exists; restore would overwrite it"
                )
            house.parent.mkdir(parents=True, exist_ok=True)
            result.parent.mkdir(parents=True, exist_ok=True)
            pairs = [(archive / "house", house)]
            if receipt["has_result"]:
                pairs.append((archive / "result", result))
            self._move_all(pairs)
            return {"house_id": house_id}
