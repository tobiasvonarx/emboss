"""Scaffold-prompted MobileSAM mask for orthophoto selection."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from building_data.orthophoto_correction.geometry import pixel_centers_from_bounds
from building_data.orthophoto_correction.models import LV95_BOUNDS


MODEL_REPOSITORY = "ChaoningZhang/MobileSAM"
MODEL_REVISION = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
MODEL_FILENAME = "mobile_sam.pt"
MODEL_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
MODEL_URL = (
    "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/"
    f"{MODEL_REVISION}/weights/{MODEL_FILENAME}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(path: Path) -> Path:
    """Download and verify the pinned official MobileSAM checkpoint once."""

    model_path = Path(path)
    if model_path.is_file():
        if _sha256(model_path) != MODEL_SHA256:
            raise RuntimeError(f"Unexpected MobileSAM checkpoint contents: {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "pv-placement-emboss"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=120) as response, NamedTemporaryFile(
            dir=model_path.parent,
            prefix=f".{MODEL_FILENAME}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(response, temporary)
        if _sha256(temporary_path) != MODEL_SHA256:
            raise RuntimeError("Downloaded MobileSAM checkpoint failed its SHA-256 check.")
        temporary_path.replace(model_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return model_path


@lru_cache(maxsize=2)
def _load_model(path_text: str) -> Any:
    import torch
    from mobile_sam import sam_model_registry

    path = ensure_checkpoint(Path(path_text))
    model = sam_model_registry["vit_t"](checkpoint=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _roof_prompt_box(
    roof_envelope: BaseGeometry,
    *,
    bounds_lv95: LV95_BOUNDS,
    shape: tuple[int, int],
    padding_m: float,
) -> np.ndarray:
    height, width = (int(shape[0]), int(shape[1]))
    raster_min_x, raster_min_y, raster_max_x, raster_max_y = bounds_lv95
    roof_min_x, roof_min_y, roof_max_x, roof_max_y = roof_envelope.bounds
    padding = float(padding_m)
    pixel_width = (float(raster_max_x) - float(raster_min_x)) / float(width)
    pixel_height = (float(raster_max_y) - float(raster_min_y)) / float(height)
    box = np.asarray(
        [
            (float(roof_min_x) - padding - float(raster_min_x)) / pixel_width,
            (float(raster_max_y) - float(roof_max_y) - padding) / pixel_height,
            (float(roof_max_x) + padding - float(raster_min_x)) / pixel_width,
            (float(raster_max_y) - float(roof_min_y) + padding) / pixel_height,
        ],
        dtype=np.float32,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width - 1))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height - 1))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("The scaffold roof envelope does not overlap the orthophoto crop.")
    return box


def rasterize_roof_envelope(
    roof_envelope: BaseGeometry,
    *,
    bounds_lv95: LV95_BOUNDS,
    shape: tuple[int, int],
) -> np.ndarray:
    xs, ys = pixel_centers_from_bounds(bounds_lv95, shape)
    return np.asarray(shapely.contains_xy(roof_envelope, xs, ys), dtype=bool)


def segment_buildings(
    images: tuple[np.ndarray, ...],
    *,
    roof_envelope: BaseGeometry,
    bounds_lv95: LV95_BOUNDS,
    checkpoint: Path,
    prompt_padding_m: float,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Segment same-grid orthophoto candidates in one MobileSAM batch."""

    import torch
    from mobile_sam.utils.transforms import ResizeLongestSide

    candidates = tuple(np.asarray(image) for image in images)
    if not candidates:
        raise ValueError("MobileSAM requires at least one candidate image.")
    shape = candidates[0].shape
    if any(
        image.shape != shape
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        for image in candidates
    ):
        raise ValueError("MobileSAM requires same-grid RGB uint8 images.")

    roof_mask = rasterize_roof_envelope(
        roof_envelope,
        bounds_lv95=bounds_lv95,
        shape=shape[:2],
    )
    if not np.any(roof_mask):
        raise ValueError("The scaffold roof envelope does not overlap the orthophoto crop.")

    prompt_box = _roof_prompt_box(
        roof_envelope,
        bounds_lv95=bounds_lv95,
        shape=shape[:2],
        padding_m=prompt_padding_m,
    )
    model = _load_model(str(Path(checkpoint).resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    transform = ResizeLongestSide(model.image_encoder.img_size)
    resized_box = transform.apply_boxes(prompt_box[None, :], shape[:2])
    inputs = []
    for image in candidates:
        resized = transform.apply_image(np.ascontiguousarray(image))
        inputs.append(
            {
                "image": torch.as_tensor(resized, device=device)
                .permute(2, 0, 1)
                .contiguous(),
                "original_size": shape[:2],
                "boxes": torch.as_tensor(
                    resized_box,
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
    with torch.inference_mode():
        outputs = model(inputs, multimask_output=False)
    masks = tuple(
        np.asarray(output["masks"][0, 0].detach().cpu(), dtype=bool)
        for output in outputs
    )
    if any(not np.any(mask) for mask in masks):
        raise ValueError("MobileSAM did not segment the prompted building.")
    return masks, roof_mask
