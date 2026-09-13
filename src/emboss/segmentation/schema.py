"""RID2 dataset schema for roof-superstructure segmentation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]
RID2SubsetName = Literal["grid", "roof_centered"]
RID2TaskName = Literal["superstructures", "segments"]

RID2_ZENODO_RECORD = "14062580"
RID2_DOI = "10.5281/zenodo.14062580"
RID2_IMAGE_SIZE_PX = 512
RID2_GROUND_RESOLUTION_M = 0.08
RID2_CRS = "EPSG:28992"
RID2_BACKGROUND_CLASS_ID = 5

RID2_README_URL = "https://zenodo.org/records/14062580/files/README.pdf?download=1"
RID2_LABELING_GUIDE_URL = (
    "https://zenodo.org/records/14062580/files/RID2_labeling_guide.pdf?download=1"
)
RID2_ARCHIVE_URL = "https://zenodo.org/records/14062580/files/roof_information_dataset_2.zip?download=1"

RID2_FILE_MD5 = {
    "README.pdf": "14cbd9a18ba022cc53f10d0b5151393f",
    "RID2_labeling_guide.pdf": "fd27f0439fe1a94c128ef0a3d4cad53d",
    "roof_information_dataset_2.zip": "c967c3932b742728716c977acb925fc7",
}


@dataclass(frozen=True)
class RID2Class:
    """One compact RID2 semantic mask class."""

    class_id: int
    name: str
    description: str


RID2_SUPERSTRUCTURE_CLASSES = (
    RID2Class(0, "pvmodule", "Existing solar thermal or photovoltaic module."),
    RID2Class(1, "dormer", "Dormer or dormer-like raised roof structure."),
    RID2Class(2, "window", "Roof window or skylight."),
    RID2Class(3, "balcony", "Balcony on or attached to a roof surface."),
    RID2Class(
        4,
        "other",
        "Other roof superstructure such as chimney, AC, ladder, dish, or wall.",
    ),
    RID2Class(5, "background", "No annotated roof superstructure."),
)

RID2_SEGMENT_CLASSES = (
    RID2Class(0, "N", "North-facing roof segment."),
    RID2Class(1, "E", "East-facing roof segment."),
    RID2Class(2, "S", "South-facing roof segment."),
    RID2Class(3, "W", "West-facing roof segment."),
    RID2Class(4, "flat", "Flat roof segment."),
    RID2Class(5, "background", "No annotated roof segment."),
)

RID2_SUPERSTRUCTURE_CLASS_IDS = tuple(
    item.class_id for item in RID2_SUPERSTRUCTURE_CLASSES
)
RID2_SEGMENT_CLASS_IDS = tuple(item.class_id for item in RID2_SEGMENT_CLASSES)
RID2_HARD_SUPERSTRUCTURE_CLASS_IDS = (1, 2, 3, 4)

# Compatibility aliases for older Emboss import sites. The semantics are now RID2.
RID_BACKGROUND_CLASS_ID = RID2_BACKGROUND_CLASS_ID
RID_RELEVANT_HARD_STRUCTURE_IDS = RID2_HARD_SUPERSTRUCTURE_CLASS_IDS
RID_SUPERSTRUCTURE_CLASSES = RID2_SUPERSTRUCTURE_CLASSES
RID_TARGET_CLASS_IDS = RID2_SUPERSTRUCTURE_CLASS_IDS


@dataclass(frozen=True)
class RID2SampleRecord:
    """Portable manifest row for one RID2 image tile.

    Paths are stored relative to the extracted RID2 dataset root, so manifests can
    be moved together with the dataset directory.
    """

    sample_id: str
    split: SplitName
    subset: RID2SubsetName
    image_path: str
    geo_image_path: str | None
    segment_mask_path: str | None
    superstructure_mask_path: str | None
    width: int = RID2_IMAGE_SIZE_PX
    height: int = RID2_IMAGE_SIZE_PX
    ground_resolution_m: float = RID2_GROUND_RESOLUTION_M
    crs: str = RID2_CRS

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RID2SampleRecord:
        return cls(
            sample_id=str(payload["sample_id"]),
            split=payload["split"],  # type: ignore[arg-type]
            subset=payload["subset"],  # type: ignore[arg-type]
            image_path=str(payload["image_path"]),
            geo_image_path=(
                None
                if payload.get("geo_image_path") is None
                else str(payload["geo_image_path"])
            ),
            segment_mask_path=(
                None
                if payload.get("segment_mask_path") is None
                else str(payload["segment_mask_path"])
            ),
            superstructure_mask_path=(
                None
                if payload.get("superstructure_mask_path") is None
                else str(payload["superstructure_mask_path"])
            ),
            width=int(payload.get("width", RID2_IMAGE_SIZE_PX)),
            height=int(payload.get("height", RID2_IMAGE_SIZE_PX)),
            ground_resolution_m=float(
                payload.get("ground_resolution_m", RID2_GROUND_RESOLUTION_M)
            ),
            crs=str(payload.get("crs", RID2_CRS)),
        )

    @classmethod
    def from_json(cls, line: str) -> RID2SampleRecord:
        return cls.from_dict(json.loads(line))

    def resolve(self, dataset_root: str | Path, field_name: str) -> Path | None:
        value = getattr(self, field_name)
        if value is None:
            return None
        return Path(dataset_root) / str(value)


def class_names(task: RID2TaskName = "superstructures") -> tuple[str, ...]:
    """Return compact class names for a RID2 task."""

    classes = (
        RID2_SUPERSTRUCTURE_CLASSES
        if task == "superstructures"
        else RID2_SEGMENT_CLASSES
    )
    return tuple(item.name for item in classes)


def classes_for_task(task: RID2TaskName = "superstructures") -> tuple[RID2Class, ...]:
    """Return compact class definitions for a RID2 task."""

    if task == "superstructures":
        return RID2_SUPERSTRUCTURE_CLASSES
    if task == "segments":
        return RID2_SEGMENT_CLASSES
    raise ValueError(f"Unsupported RID2 task: {task}")


def validate_mask_values(values: set[int], task: RID2TaskName) -> None:
    """Validate that a mask only uses compact RID2 class ids."""

    allowed = {item.class_id for item in classes_for_task(task)}
    unexpected = sorted(values - allowed)
    if unexpected:
        raise ValueError(
            f"RID2 {task} mask contains unexpected class ids {unexpected}; "
            f"expected only {sorted(allowed)}."
        )
