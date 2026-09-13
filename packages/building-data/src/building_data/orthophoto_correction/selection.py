"""Select the raw or corrected orthophoto whose building best matches the scaffold."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry.base import BaseGeometry

from building_data.orthophoto_correction.mobile_sam import rasterize_roof_envelope
from building_data.orthophoto_correction.mobile_sam import segment_buildings
from building_data.orthophoto_correction.geometry import build_displacement_field
from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.models import DisplacementField
from building_data.orthophoto_correction.models import LV95_BOUNDS
from building_data.orthophoto_correction.models import StripCandidate
from building_data.orthophoto_correction.warp import compute_occlusion_mask
from building_data.orthophoto_correction.warp import warp_rgb_with_displacement


RAW_CANDIDATE_ID = "raw"


@dataclass(frozen=True)
class OrthophotoCandidateResult:
    candidate_id: str
    strip: StripCandidate | None
    rgb: np.ndarray
    valid_mask: np.ndarray
    occlusion_mask: np.ndarray
    displacement: DisplacementField
    roof_iou: float | None
    mean_strip_distance_m: float | None


def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    union = np.asarray(left, dtype=bool) | np.asarray(right, dtype=bool)
    if not np.any(union):
        return 0.0
    intersection = np.asarray(left, dtype=bool) & np.asarray(right, dtype=bool)
    return float(np.count_nonzero(intersection) / np.count_nonzero(union))


def raw_candidate(
    rgb: np.ndarray,
    *,
    height_m: np.ndarray,
) -> OrthophotoCandidateResult:
    valid_mask = np.any(np.asarray(rgb) > 0, axis=2)
    zeros = np.zeros(np.asarray(height_m).shape, dtype=np.float32)
    field = DisplacementField(
        dx_m=zeros,
        dy_m=zeros.copy(),
        signed_distance_m=zeros.copy(),
        height_m=np.asarray(height_m, dtype=np.float32),
        valid_mask=valid_mask,
        corrected_height_mask=(np.asarray(height_m) > 0.0) & valid_mask,
    )
    return OrthophotoCandidateResult(
        candidate_id=RAW_CANDIDATE_ID,
        strip=None,
        rgb=np.asarray(rgb),
        valid_mask=valid_mask,
        occlusion_mask=np.zeros(valid_mask.shape, dtype=bool),
        displacement=field,
        roof_iou=None,
        mean_strip_distance_m=None,
    )


def corrected_candidate(
    rgb: np.ndarray,
    *,
    height_m: np.ndarray,
    roof_mask: np.ndarray,
    bounds_lv95: LV95_BOUNDS,
    strip: StripCandidate,
    config: CorrectionConfig,
) -> OrthophotoCandidateResult:
    field = build_displacement_field(
        bounds_lv95=bounds_lv95,
        height_m=height_m,
        strip=strip,
        config=config,
    )
    corrected, valid_mask = warp_rgb_with_displacement(rgb, field, bounds_lv95)
    occlusion_mask = compute_occlusion_mask(field.corrected_height_mask, field, bounds_lv95)
    distances = np.abs(field.signed_distance_m[roof_mask])
    return OrthophotoCandidateResult(
        candidate_id=strip.strip_id,
        strip=strip,
        rgb=corrected,
        valid_mask=valid_mask,
        occlusion_mask=occlusion_mask,
        displacement=field,
        roof_iou=None,
        mean_strip_distance_m=float(np.mean(distances)) if distances.size else float("inf"),
    )


def _usable_mask(candidate: OrthophotoCandidateResult) -> np.ndarray:
    return (
        np.asarray(candidate.valid_mask, dtype=bool)
        & ~np.asarray(candidate.occlusion_mask, dtype=bool)
        & np.asarray(candidate.displacement.valid_mask, dtype=bool)
    )


def _score_with_mobile_sam(
    candidates: tuple[OrthophotoCandidateResult, ...],
    *,
    roof_envelope: BaseGeometry,
    bounds_lv95: LV95_BOUNDS,
    config: CorrectionConfig,
) -> tuple[OrthophotoCandidateResult, ...]:
    usable_masks = tuple(_usable_mask(candidate) for candidate in candidates)
    images = []
    for candidate, usable in zip(candidates, usable_masks, strict=True):
        image = np.asarray(candidate.rgb, dtype=np.uint8).copy()
        image[~usable] = 0
        images.append(image)
    building_masks, roof_mask = segment_buildings(
        tuple(images),
        roof_envelope=roof_envelope,
        bounds_lv95=bounds_lv95,
        checkpoint=config.mobile_sam_checkpoint,
        prompt_padding_m=float(config.mobile_sam_prompt_padding_m),
    )
    return tuple(
        replace(
            candidate,
            roof_iou=_intersection_over_union(building & usable, roof_mask),
        )
        for candidate, building, usable in zip(
            candidates,
            building_masks,
            usable_masks,
            strict=True,
        )
    )


def select_orthophoto_candidate(
    rgb: np.ndarray,
    *,
    height_m: np.ndarray,
    bounds_lv95: LV95_BOUNDS,
    roof_envelope: BaseGeometry,
    strips: tuple[StripCandidate, ...],
    config: CorrectionConfig,
    include_raw: bool = True,
) -> tuple[OrthophotoCandidateResult, tuple[OrthophotoCandidateResult, ...]]:
    if not strips and not include_raw:
        raise ValueError("At least one orthophoto candidate is required.")

    roof_mask = rasterize_roof_envelope(
        roof_envelope=roof_envelope,
        bounds_lv95=bounds_lv95,
        shape=np.asarray(rgb).shape[:2],
    )
    candidates = tuple(
        [
            raw_candidate(
                rgb,
                height_m=height_m,
            )
        ]
        if include_raw
        else []
    ) + tuple(
        corrected_candidate(
            rgb,
            height_m=height_m,
            roof_mask=roof_mask,
            bounds_lv95=bounds_lv95,
            strip=strip,
            config=config,
        )
        for strip in strips
    )
    candidates = _score_with_mobile_sam(
        candidates,
        roof_envelope=roof_envelope,
        bounds_lv95=bounds_lv95,
        config=config,
    )
    selected = min(
        candidates,
        key=lambda item: (
            -float(item.roof_iou or 0.0),
            item.candidate_id != RAW_CANDIDATE_ID,
            float("inf") if item.mean_strip_distance_m is None else item.mean_strip_distance_m,
            item.candidate_id,
        ),
    )
    return selected, candidates
