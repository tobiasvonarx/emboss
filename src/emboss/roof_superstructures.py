"""Emboss adapter for the thesis roof-superstructure segmentation model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile

from emboss.segmentation.schema import (
    RID_BACKGROUND_CLASS_ID,
    RID_SUPERSTRUCTURE_CLASSES,
    RID_TARGET_CLASS_IDS,
)

from .image_superstructures import image_class_names
from .segmentation.checkpoint import CHECKPOINT_ENV_VAR, default_checkpoint_path
from .workspace import EmbossWorkspace

if TYPE_CHECKING:
    import torch

BACKGROUND_CLASS_ID = RID_BACKGROUND_CLASS_ID
DEFAULT_HARD_STRUCTURE_CLASS_IDS = tuple(
    class_id for class_id in RID_TARGET_CLASS_IDS if class_id != BACKGROUND_CLASS_ID
)
CLASS_NAMES = {item.class_id: item.name for item in RID_SUPERSTRUCTURE_CLASSES}


@dataclass(frozen=True)
class RoofSuperstructureOptions:
    """Inference options for the local roof-superstructure model adapter."""

    checkpoint_path: str | Path | None = None
    device: str = "auto"
    include_class_ids: tuple[int, ...] = DEFAULT_HARD_STRUCTURE_CLASS_IDS
    clip_to_vector_roof: bool = True


@dataclass(frozen=True)
class RoofSuperstructureSegmentation:
    """Semantic roof-superstructure inference in raw RID class ids."""

    class_ids: np.ndarray
    foreground: np.ndarray
    hard_mask: np.ndarray
    foreground_class_map: np.ndarray
    hard_class_map: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CachedRoofSuperstructureSegmentation:
    foreground_mask: np.ndarray
    class_map: np.ndarray
    mask_path: Path
    class_map_path: Path
    source: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RoofSuperstructureSegmentationInput:
    building_fid: int
    rgb: np.ndarray
    vector_roof_mask: np.ndarray


@dataclass(frozen=True)
class RoofMaskPrediction:
    mask: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    mask_role: str = "roof"


def roof_superstructure_class_cache_path(
    workspace: EmbossWorkspace,
    building_fid: int,
) -> Path:
    return (
        workspace.corrected_orthophoto_root
        / f"fid_{int(building_fid)}"
        / "emboss"
        / "image_superstructures_class_map.tif"
    )


def roof_superstructure_mask_cache_path(
    workspace: EmbossWorkspace,
    building_fid: int,
) -> Path:
    return (
        workspace.corrected_orthophoto_root
        / f"fid_{int(building_fid)}"
        / "emboss"
        / "image_superstructures_mask.tif"
    )


def read_cached_hard_class_map(
    path: str | Path,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    class_map = tifffile.imread(Path(path)).astype(np.uint8)
    if class_map.ndim == 3:
        class_map = class_map[..., 0]
    if tuple(class_map.shape) != tuple(int(value) for value in expected_shape):
        raise ValueError(
            f"Cached roof-superstructure class map at {path} has shape {tuple(class_map.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    return np.asarray(class_map, dtype=np.uint8)


def write_cached_hard_class_map(path: str | Path, class_map: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, np.asarray(class_map, dtype=np.uint8))
    return output_path


def _metadata_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".json")


def _read_cache_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = _metadata_path(path)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_cache_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    _metadata_path(path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _metadata_matches(path: str | Path, expected: dict[str, Any]) -> bool:
    cached = _read_cache_metadata(path)
    return bool(cached) and all(
        cached.get(key) == value for key, value in expected.items()
    )


def _segmentation_input_sha256(rgb: np.ndarray, vector_roof_mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (
        np.asarray(rgb, dtype=np.uint8),
        np.asarray(vector_roof_mask, dtype=bool),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _resolve_device(device: str) -> torch.device:
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_config_kwargs_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    from emboss.segmentation.config import ModelConfig

    configured = checkpoint.get("model_config") or {}
    allowed = {field.name for field in fields(ModelConfig)}
    return {key: value for key, value in dict(configured).items() if key in allowed}


def _checkpoint_rid_class_ids(
    checkpoint: dict[str, Any], class_count: int
) -> tuple[int, ...]:
    del checkpoint
    rid_class_ids = RID_TARGET_CLASS_IDS
    if int(class_count) != len(rid_class_ids):
        raise ValueError(
            "Roof-superstructure inference expects the active compact RID2 "
            f"{len(rid_class_ids)}-channel checkpoint; checkpoint declares {class_count} classes."
        )
    return rid_class_ids


def _map_train_ids_to_rid_class_ids(
    train_class_ids: np.ndarray,
    rid_class_ids: tuple[int, ...],
) -> np.ndarray:
    train_ids = np.asarray(train_class_ids, dtype=np.int64)
    mapping = np.asarray(rid_class_ids, dtype=np.uint8)
    if train_ids.size and int(train_ids.max()) >= mapping.shape[0]:
        raise ValueError(
            f"Predicted train id {int(train_ids.max())} is outside checkpoint class mapping "
            f"of length {mapping.shape[0]}."
        )
    return mapping[train_ids]


def _hard_class_map(class_ids: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    class_id_array = np.asarray(class_ids, dtype=np.uint8)
    mask = np.asarray(hard_mask, dtype=bool)
    class_map = np.full(class_id_array.shape, BACKGROUND_CLASS_ID, dtype=np.uint8)
    class_map[mask] = class_id_array[mask]
    return class_map


def _foreground_class_map(
    class_ids: np.ndarray, foreground_mask: np.ndarray
) -> np.ndarray:
    class_id_array = np.asarray(class_ids, dtype=np.uint8)
    mask = np.asarray(foreground_mask, dtype=bool) & (
        class_id_array != BACKGROUND_CLASS_ID
    )
    class_map = np.full(class_id_array.shape, BACKGROUND_CLASS_ID, dtype=np.uint8)
    class_map[mask] = class_id_array[mask]
    return class_map


class RoofSuperstructureMaskClient:
    """Local semantic roof-superstructure provider.

    The model predicts RID-compatible roof object classes. The binary mask
    returned through the Emboss provider interface keeps all non-background
    roof-superstructure classes; downstream lifting decides class-specific
    extrusion heights.
    """

    provider_name = "roof_superstructures"
    cache_key = "roof_superstructures"
    mask_role = "superstructure"

    def __init__(self, options: RoofSuperstructureOptions | None = None) -> None:
        self.options = options or RoofSuperstructureOptions()
        self._model: Any | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._device: Any | None = None

    @property
    def checkpoint_path(self) -> Path:
        configured = self.options.checkpoint_path
        if configured is None:
            return default_checkpoint_path()
        path = Path(configured).expanduser()
        return path.resolve()

    def cache_metadata(self) -> dict[str, Any]:
        checkpoint_path = self.checkpoint_path
        metadata: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "include_class_ids": [
                int(value) for value in self.options.include_class_ids
            ],
            "clip_to_vector_roof": bool(self.options.clip_to_vector_roof),
            "component_filter": "none",
        }
        if checkpoint_path.exists():
            metadata["checkpoint_mtime_ns"] = int(checkpoint_path.stat().st_mtime_ns)
        return metadata

    def _load_model(self) -> tuple[Any, dict[str, Any], Any]:
        if (
            self._model is not None
            and self._checkpoint is not None
            and self._device is not None
        ):
            return self._model, self._checkpoint, self._device

        import torch

        from emboss.segmentation.config import ModelConfig
        from emboss.segmentation.net import build_model

        checkpoint_path = self.checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Roof superstructure checkpoint not found at {checkpoint_path}. "
                f"Set {CHECKPOINT_ENV_VAR} to a compatible single-model checkpoint."
            )

        device = _resolve_device(self.options.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"{checkpoint_path} does not contain a full model_state_dict."
            )
        model = build_model(
            ModelConfig(**_model_config_kwargs_from_checkpoint(checkpoint))
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()

        self._model = model
        self._checkpoint = checkpoint
        self._device = device
        return model, checkpoint, device

    def _predict_arrays_many(
        self,
        rgbs: tuple[np.ndarray, ...],
    ) -> tuple[tuple[np.ndarray, np.ndarray, dict[str, Any]], ...]:
        import torch

        from emboss.segmentation.normalization import RGB_MEAN, RGB_STD

        if not rgbs:
            return ()
        model, checkpoint, device = self._load_model()
        training_config = dict(checkpoint.get("training_config", {}))
        tensors = []
        shapes = {tuple(np.asarray(rgb).shape) for rgb in rgbs}
        if len(shapes) != 1:
            raise ValueError(
                "Batched roof-superstructure inference requires equal image shapes."
            )
        for rgb in rgbs:
            image = np.asarray(rgb, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB image with shape H x W x 3, got {tuple(image.shape)}"
                )
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if bool(
                training_config.get(
                    "normalize_imagenet", training_config.get("normalize_rgb", True)
                )
            ):
                tensor = (tensor - RGB_MEAN) / RGB_STD
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0).to(device)
        model_input_size = int(
            training_config.get("model_input_size") or batch.shape[-1]
        )

        with torch.inference_mode():
            if tuple(batch.shape[-2:]) == (model_input_size, model_input_size):
                model_batch = batch
            else:
                model_batch = torch.nn.functional.interpolate(
                    batch,
                    size=(model_input_size, model_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs = model(model_batch, output_size=tuple(batch.shape[-2:]))
            class_logits = outputs["superstructure_class"]
            class_count = int(class_logits.shape[1])
            rid_class_ids = _checkpoint_rid_class_ids(checkpoint, class_count)
            if len(rid_class_ids) != class_count:
                raise ValueError(
                    f"Checkpoint class mapping has {len(rid_class_ids)} entries, "
                    f"but model predicted {class_count} classes."
                )
            background_train_id = (
                rid_class_ids.index(BACKGROUND_CLASS_ID)
                if BACKGROUND_CLASS_ID in rid_class_ids
                else class_count - 1
            )
            train_class_ids = class_logits.argmax(dim=1)
            foreground = train_class_ids != background_train_id

        base_diagnostics = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_metrics": checkpoint.get("metrics", {}),
            "device": str(device),
            "checkpoint_rid_class_ids": [int(value) for value in rid_class_ids],
            "checkpoint_classes": [
                CLASS_NAMES.get(int(value), str(int(value))) for value in rid_class_ids
            ],
            "batch_size": len(rgbs),
            "model_input_size": model_input_size,
        }
        class_id_batches = train_class_ids.detach().cpu().numpy().astype(np.uint8)
        foreground_batches = foreground.detach().cpu().numpy().astype(bool)
        return tuple(
            (
                _map_train_ids_to_rid_class_ids(
                    class_id_batches[index], rid_class_ids
                ).astype(np.uint8),
                foreground_batches[index],
                dict(base_diagnostics),
            )
            for index in range(len(rgbs))
        )

    def _finalize_segmentation(
        self,
        class_ids: np.ndarray,
        foreground: np.ndarray,
        diagnostics: dict[str, Any],
        *,
        vector_roof_mask: np.ndarray | None = None,
    ) -> RoofSuperstructureSegmentation:
        included = np.asarray(self.options.include_class_ids, dtype=np.uint8)
        foreground_structure = foreground & (class_ids != BACKGROUND_CLASS_ID)
        hard_structure = np.isin(class_ids, included) & foreground
        if self.options.clip_to_vector_roof and vector_roof_mask is not None:
            vector_bool = np.asarray(vector_roof_mask, dtype=bool)
            foreground_structure &= vector_bool
            hard_structure &= vector_bool
        foreground_class_map = _foreground_class_map(class_ids, foreground_structure)
        hard_class_map = _hard_class_map(class_ids, hard_structure)

        class_counts = {
            CLASS_NAMES.get(int(class_id), str(int(class_id))): int(count)
            for class_id, count in zip(
                *np.unique(class_ids, return_counts=True), strict=True
            )
        }
        diagnostics.update(
            {
                "mask_role": self.mask_role,
                "included_class_ids": [int(value) for value in included],
                "included_classes": [
                    CLASS_NAMES.get(int(value), str(int(value))) for value in included
                ],
                "class_pixel_counts": class_counts,
                "foreground_pixels": int(np.count_nonzero(foreground)),
                "display_foreground_pixels": int(
                    np.count_nonzero(foreground_structure)
                ),
                "hard_structure_pixels": int(np.count_nonzero(hard_structure)),
                "clip_to_vector_roof": bool(self.options.clip_to_vector_roof),
            }
        )
        return RoofSuperstructureSegmentation(
            class_ids=class_ids,
            foreground=foreground,
            hard_mask=hard_structure,
            foreground_class_map=foreground_class_map,
            hard_class_map=hard_class_map,
            diagnostics=diagnostics,
        )

    def segment_roof(
        self,
        rgb: np.ndarray,
        *,
        vector_roof_mask: np.ndarray | None = None,
    ) -> RoofMaskPrediction:
        segmentation = self.predict_hard_segmentation(
            rgb, vector_roof_mask=vector_roof_mask
        )
        return RoofMaskPrediction(
            mask=segmentation.hard_mask,
            diagnostics=segmentation.diagnostics,
            mask_role=self.mask_role,
        )

    def predict_hard_segmentation_many(
        self,
        items: tuple[tuple[np.ndarray, np.ndarray | None], ...],
    ) -> tuple[RoofSuperstructureSegmentation, ...]:
        if not items:
            return ()
        outputs: list[RoofSuperstructureSegmentation | None] = [None] * len(items)
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, (rgb, vector_roof_mask) in enumerate(items):
            image = np.asarray(rgb, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB image with shape H x W x 3, got {tuple(image.shape)}"
                )
            if vector_roof_mask is not None and tuple(vector_roof_mask.shape) != tuple(
                image.shape[:2]
            ):
                raise ValueError(
                    f"vector_roof_mask has shape {tuple(vector_roof_mask.shape)}, "
                    f"expected {tuple(image.shape[:2])}"
                )
            grouped.setdefault((int(image.shape[0]), int(image.shape[1])), []).append(
                index
            )

        for indices in grouped.values():
            predictions = self._predict_arrays_many(
                tuple(np.asarray(items[index][0], dtype=np.uint8) for index in indices)
            )
            for local_index, item_index in enumerate(indices):
                class_ids, foreground, diagnostics = predictions[local_index]
                outputs[item_index] = self._finalize_segmentation(
                    class_ids,
                    foreground,
                    diagnostics,
                    vector_roof_mask=items[item_index][1],
                )
        return tuple(output for output in outputs if output is not None)

    def predict_hard_segmentation(
        self,
        rgb: np.ndarray,
        *,
        vector_roof_mask: np.ndarray | None = None,
    ) -> RoofSuperstructureSegmentation:
        return self.predict_hard_segmentation_many(((rgb, vector_roof_mask),))[0]


def load_or_compute_roof_superstructure_segmentation(
    *,
    workspace: EmbossWorkspace,
    building_fid: int,
    rgb: np.ndarray,
    vector_roof_mask: np.ndarray,
    client: RoofSuperstructureMaskClient | None = None,
    force_refresh: bool = False,
) -> CachedRoofSuperstructureSegmentation:
    """Load or compute cached semantic roof-superstructure segmentation."""

    return load_or_compute_roof_superstructure_segmentation_many(
        workspace=workspace,
        items=(
            RoofSuperstructureSegmentationInput(
                building_fid=int(building_fid),
                rgb=rgb,
                vector_roof_mask=vector_roof_mask,
            ),
        ),
        client=client,
        force_refresh=force_refresh,
    )[int(building_fid)]


def _cached_segmentation_if_valid(
    *,
    workspace: EmbossWorkspace,
    building_fid: int,
    expected_shape: tuple[int, int],
    expected_metadata: dict[str, Any],
    force_refresh: bool,
) -> CachedRoofSuperstructureSegmentation | None:
    class_map_path = roof_superstructure_class_cache_path(workspace, int(building_fid))
    mask_path = roof_superstructure_mask_cache_path(workspace, int(building_fid))
    if (
        force_refresh
        or not class_map_path.exists()
        or not mask_path.exists()
        or not _metadata_matches(class_map_path, expected_metadata)
    ):
        return None
    try:
        class_map = read_cached_hard_class_map(
            class_map_path, expected_shape=expected_shape
        )
    except ValueError:
        # The crop was regenerated with a different size; treat as a cache miss.
        return None
    mask = tifffile.imread(mask_path).astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if tuple(mask.shape) != expected_shape:
        return None
    return CachedRoofSuperstructureSegmentation(
        foreground_mask=np.asarray(mask, dtype=bool),
        class_map=class_map,
        mask_path=mask_path,
        class_map_path=class_map_path,
        source="cache",
        diagnostics=dict(_read_cache_metadata(class_map_path)),
    )


def _write_computed_segmentation(
    *,
    workspace: EmbossWorkspace,
    building_fid: int,
    segmentation: RoofSuperstructureSegmentation,
    client: RoofSuperstructureMaskClient,
    expected_metadata: dict[str, Any],
) -> CachedRoofSuperstructureSegmentation:
    class_map_path = roof_superstructure_class_cache_path(workspace, int(building_fid))
    mask_path = roof_superstructure_mask_cache_path(workspace, int(building_fid))
    foreground_mask = np.asarray(
        segmentation.foreground_class_map != BACKGROUND_CLASS_ID, dtype=bool
    )
    write_cached_hard_class_map(class_map_path, segmentation.foreground_class_map)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(mask_path, foreground_mask.astype(np.uint8))
    metadata = {
        **expected_metadata,
        "mask_role": client.mask_role,
        "included_class_ids": list(
            segmentation.diagnostics.get("included_class_ids", [])
        ),
        "included_classes": list(segmentation.diagnostics.get("included_classes", [])),
        "segmentation_classes": list(image_class_names()),
    }
    _write_cache_metadata(class_map_path, metadata)
    _write_cache_metadata(mask_path, metadata)
    return CachedRoofSuperstructureSegmentation(
        foreground_mask=foreground_mask,
        class_map=segmentation.foreground_class_map,
        mask_path=mask_path,
        class_map_path=class_map_path,
        source="computed",
        diagnostics=segmentation.diagnostics,
    )


def load_or_compute_roof_superstructure_segmentation_many(
    *,
    workspace: EmbossWorkspace,
    items: tuple[RoofSuperstructureSegmentationInput, ...],
    client: RoofSuperstructureMaskClient | None = None,
    force_refresh: bool = False,
) -> dict[int, CachedRoofSuperstructureSegmentation]:
    """Load cached image segmentation, batching model inference for cache misses."""

    if not items:
        return {}
    model_client = client or RoofSuperstructureMaskClient()
    model_metadata = model_client.cache_metadata()
    results: dict[int, CachedRoofSuperstructureSegmentation] = {}
    misses: list[tuple[RoofSuperstructureSegmentationInput, dict[str, Any]]] = []
    for item in items:
        image = np.asarray(item.rgb, dtype=np.uint8)
        expected_shape = (int(image.shape[0]), int(image.shape[1]))
        expected_metadata = {
            **model_metadata,
            "input_sha256": _segmentation_input_sha256(image, item.vector_roof_mask),
        }
        cached = _cached_segmentation_if_valid(
            workspace=workspace,
            building_fid=int(item.building_fid),
            expected_shape=expected_shape,
            expected_metadata=expected_metadata,
            force_refresh=force_refresh,
        )
        if cached is None:
            misses.append((item, expected_metadata))
        else:
            results[int(item.building_fid)] = cached

    if not misses:
        return results
    predictions = model_client.predict_hard_segmentation_many(
        tuple((item.rgb, item.vector_roof_mask) for item, _metadata in misses)
    )
    for (item, expected_metadata), segmentation in zip(
        misses, predictions, strict=True
    ):
        results[int(item.building_fid)] = _write_computed_segmentation(
            workspace=workspace,
            building_fid=int(item.building_fid),
            segmentation=segmentation,
            client=model_client,
            expected_metadata=expected_metadata,
        )
    return results
