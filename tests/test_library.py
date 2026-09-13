"""Reversible building archives, rollback and queued-work protection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from building_data.store import AcquisitionStore
from filelock import FileLock

from emboss.library import BuildingBusyError, BuildingLibrary


@pytest.fixture
def library(tmp_path):
    house = tmp_path / "houses/building-1"
    house.mkdir(parents=True)
    (house / "house.json").write_text(
        json.dumps(
            {
                "schema": "building-input-v1",
                "id": "building-1",
                "building_fid": 42,
                "units": "m",
                "crs": "EPSG:2056",
                "bounds_xy": [0, 0, 10, 10],
                "surfaces_path": "acquisitions/shared/surfaces.gpkg",
                "sources": [],
            }
        )
    )
    (house / "imagery").mkdir()
    (house / "imagery/rgb.tif").write_bytes(b"house imagery")
    result = tmp_path / "results/building-1"
    result.mkdir(parents=True)
    (result / "mesh.ply").write_bytes(b"saved scientific mesh")
    for name in [
        "acquisitions/shared/surfaces.gpkg",
        "cache/lidar.las",
        "cache/model.pt",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"shared input")
    return BuildingLibrary(AcquisitionStore(tmp_path))


def contents(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".lock"
    }


def test_remove_and_restore_preserve_owned_and_shared_files_after_reload(library):
    before = contents(library.root)
    removed = library.remove("building-1")
    assert removed["building_fid"] == 42
    assert len(removed["undo_token"]) == 32
    assert not (library.root / "houses/building-1").exists()
    assert not (library.root / "results/building-1").exists()
    assert library.store.list_houses() == []
    for path, content in before.items():
        if path.parts[0] in {"cache", "acquisitions"}:
            assert (library.root / path).read_bytes() == content
    reloaded = BuildingLibrary(library.store)
    assert reloaded.removed() == [removed]
    assert reloaded.restore(removed["undo_token"]) == {"house_id": "building-1"}
    assert reloaded.removed() == []
    assert library.store.house("building-1").building_fid == 42
    for path, content in before.items():
        assert (library.root / path).read_bytes() == content
    with pytest.raises(FileNotFoundError):
        reloaded.restore(removed["undo_token"])


def test_unreconstructed_building_can_be_removed_and_restored(library):
    result = library.root / "results/building-1"
    (result / "mesh.ply").unlink()
    result.rmdir()
    token = library.remove("building-1")["undo_token"]
    library.restore(token)
    assert library.store.house("building-1").building_fid == 42
    assert not result.exists()


@pytest.mark.parametrize("directory", ["houses", "results"])
def test_restore_never_overwrites_reacquired_or_new_results(library, directory):
    token = library.remove("building-1")["undo_token"]
    destination = library.root / directory / "building-1"
    destination.mkdir(parents=True)
    marker = destination / "new-data"
    marker.write_bytes(b"preserve reacquired data")
    before = contents(library.root)
    with pytest.raises(BuildingBusyError, match="overwrite"):
        library.restore(token)
    assert contents(library.root) == before


@pytest.mark.parametrize("operation", ["remove", "restore"])
def test_partial_move_failure_rolls_back_without_losing_files(
    library, monkeypatch, operation
):
    if operation == "remove":
        source = library.root / "results/building-1"
        action = lambda: library.remove("building-1")
    else:
        token = library.remove("building-1")["undo_token"]
        source = library.root / "trash/buildings" / token / "result"
        action = lambda: library.restore(token)
    before = contents(library.root)
    rename = Path.rename

    def fail_second(path, target):
        if path == source:
            raise OSError("Injected second-directory move failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="Injected"):
        action()
    assert contents(library.root) == before
    if operation == "remove":
        assert list((library.root / "trash/buildings").iterdir()) == []


def test_reservations_cover_queued_work_and_allow_parallel_preview_reads(library):
    release_one = library.reserve(["building-1"])
    release_two = library.reserve(["building-1"])
    with pytest.raises(BuildingBusyError):
        library.remove("building-1")
    release_one()
    release_one()  # Idempotent release cannot cancel another preview's reservation.
    with pytest.raises(BuildingBusyError):
        library.remove("building-1")
    release_two()
    assert library.remove("building-1")["house_id"] == "building-1"


def test_invalid_batch_reservation_does_not_leak_busy_state(library):
    with pytest.raises(FileNotFoundError):
        library.reserve(["building-1", "missing"])
    library.remove("building-1")


@pytest.mark.parametrize(
    "relative",
    [
        "results/building-1.lock",
        "houses/building-1/prepare.lock",
        "houses/building-1/imagery.lock",
    ],
)
def test_direct_processing_file_locks_block_removal(library, relative):
    with FileLock(library.root / relative), pytest.raises(BuildingBusyError):
        library.remove("building-1")
    assert library.store.house("building-1").building_fid == 42
    assert library.removed() == []


@pytest.mark.parametrize("value", ["../outside", "a/b", "..", ""])
def test_invalid_ids_and_tokens_are_rejected(library, value):
    with pytest.raises(ValueError):
        library.remove(value)
    with pytest.raises(ValueError):
        library.restore(value)


def test_unknown_building_and_archive_are_not_found(library):
    with pytest.raises(FileNotFoundError):
        library.remove("missing")
    with pytest.raises(FileNotFoundError):
        library.restore("0" * 32)


@pytest.mark.parametrize("location", ["house", "result", "archive"])
def test_symbolic_links_are_rejected_without_touching_targets(
    library, tmp_path, location
):
    outside = tmp_path / "unrelated.txt"
    outside.write_text("unchanged")
    if location == "archive":
        token = library.remove("building-1")["undo_token"]
        folder = library.root / "trash/buildings" / token / "house"
        action = lambda: library.restore(token)
    else:
        folder = (
            library.root
            / ("houses" if location == "house" else "results")
            / "building-1"
        )
        action = lambda: library.remove("building-1")
    (folder / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="Symbolic"):
        action()
    assert outside.read_text() == "unchanged"
