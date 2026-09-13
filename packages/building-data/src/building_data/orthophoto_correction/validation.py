"""Lightweight validation summaries for orthophoto correction runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from building_data.orthophoto_correction.models import CorrectionResult


def numeric_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != values.shape:
            raise ValueError("Summary mask must have the same shape as the values.")
        values = values[mask_values]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0.0,
            "min_m": 0.0,
            "mean_m": 0.0,
            "median_m": 0.0,
            "p95_m": 0.0,
            "max_m": 0.0,
        }
    return {
        "count": float(finite.size),
        "min_m": float(np.min(finite)),
        "mean_m": float(np.mean(finite)),
        "median_m": float(np.median(finite)),
        "p95_m": float(np.quantile(finite, 0.95)),
        "max_m": float(np.max(finite)),
    }


def displacement_summary(displacement_m: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    return numeric_summary(displacement_m, mask=mask)


def mask_coverage(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def planimetric_residual_summary(reference_xy: np.ndarray, candidate_xy: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference_xy, dtype=np.float64)
    candidate = np.asarray(candidate_xy, dtype=np.float64)
    if reference.shape != candidate.shape or reference.size == 0:
        return {"count": 0.0, "mean_m": 0.0, "median_m": 0.0, "p95_m": 0.0, "max_m": 0.0}
    residuals = np.linalg.norm(candidate - reference, axis=1)
    return {
        "count": float(len(residuals)),
        "mean_m": float(np.mean(residuals)),
        "median_m": float(np.median(residuals)),
        "p95_m": float(np.quantile(residuals, 0.95)),
        "max_m": float(np.max(residuals)),
    }


def correction_quality_summary(result: CorrectionResult) -> dict[str, Any]:
    corrected_height = result.corrected_height_mask
    return {
        "displacement": displacement_summary(result.displacement.displacement_m),
        "active_displacement": displacement_summary(
            result.displacement.displacement_m,
            mask=corrected_height,
        ),
        "active_height": numeric_summary(result.displacement.height_m, mask=corrected_height),
        "valid_coverage": mask_coverage(result.valid_mask),
        "occlusion_coverage": mask_coverage(result.occlusion_mask),
        "corrected_height_coverage": mask_coverage(corrected_height),
        "strip_coverage": mask_coverage(result.displacement.valid_mask),
    }
