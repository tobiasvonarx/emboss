"""LV95 strip geometry and deterministic displacement-field helpers."""

from __future__ import annotations

import math

import numpy as np

from building_data.orthophoto_correction.models import CorrectionConfig
from building_data.orthophoto_correction.models import DisplacementField
from building_data.orthophoto_correction.models import LV95_BOUNDS
from building_data.orthophoto_correction.models import StripCandidate


def crop_bounds_lv95(
    bounds_lv95: LV95_BOUNDS,
    *,
    padding_m: float,
    min_side_m: float,
) -> LV95_BOUNDS:
    min_x, min_y, max_x, max_y = _bounds_tuple(bounds_lv95)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    width = max(max_x - min_x + 2.0 * float(padding_m), float(min_side_m))
    height = max(max_y - min_y + 2.0 * float(padding_m), float(min_side_m))
    return (
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    )


def expand_bounds(bounds_lv95: LV95_BOUNDS, padding_m: float) -> LV95_BOUNDS:
    min_x, min_y, max_x, max_y = _bounds_tuple(bounds_lv95)
    padding = float(padding_m)
    return (min_x - padding, min_y - padding, max_x + padding, max_y + padding)


def lv95_geotransform(bounds_lv95: LV95_BOUNDS, width: int, height: int) -> tuple[float, float, float, float, float, float]:
    min_x, min_y, max_x, max_y = _bounds_tuple(bounds_lv95)
    return (
        min_x,
        (max_x - min_x) / float(width),
        0.0,
        max_y,
        0.0,
        -(max_y - min_y) / float(height),
    )


def pixel_centers_from_bounds(bounds_lv95: LV95_BOUNDS, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    min_x, min_y, max_x, max_y = _bounds_tuple(bounds_lv95)
    pixel_width = (max_x - min_x) / float(width)
    pixel_height = (max_y - min_y) / float(height)
    xs = min_x + (np.arange(width, dtype=np.float64) + 0.5) * pixel_width
    ys = max_y - (np.arange(height, dtype=np.float64) + 0.5) * pixel_height
    return np.meshgrid(xs, ys)


def height_above_terrain(surface_m: np.ndarray, terrain_m: np.ndarray, *, min_height_m: float) -> np.ndarray:
    surface = np.asarray(surface_m, dtype=np.float32)
    terrain = np.asarray(terrain_m, dtype=np.float32)
    height = np.maximum(surface - terrain, 0.0).astype(np.float32, copy=False)
    height[height < float(min_height_m)] = 0.0
    return height


def _bounds_tuple(bounds_lv95: LV95_BOUNDS) -> LV95_BOUNDS:
    return tuple(float(value) for value in bounds_lv95)  # type: ignore[return-value]


def signed_distance_to_polyline(
    xs: np.ndarray,
    ys: np.ndarray,
    polyline_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed nearest-segment distance and unit vector toward each point."""

    x_flat = np.asarray(xs, dtype=np.float64).ravel()
    y_flat = np.asarray(ys, dtype=np.float64).ravel()
    line = np.asarray(polyline_xy, dtype=np.float64)
    shape = np.asarray(xs).shape

    best_dist2 = np.full(x_flat.shape, np.inf, dtype=np.float64)
    best_signed = np.full(x_flat.shape, np.inf, dtype=np.float64)
    best_ux = np.zeros(x_flat.shape, dtype=np.float64)
    best_uy = np.zeros(x_flat.shape, dtype=np.float64)

    if line.ndim != 2 or line.shape[0] < 2 or line.shape[1] < 2:
        return (
            best_signed.reshape(shape),
            best_ux.reshape(shape),
            best_uy.reshape(shape),
        )

    for start, end in zip(line[:-1], line[1:], strict=False):
        ax, ay = float(start[0]), float(start[1])
        vx = float(end[0] - start[0])
        vy = float(end[1] - start[1])
        length2 = vx * vx + vy * vy
        if length2 <= 1e-12:
            continue

        t = ((x_flat - ax) * vx + (y_flat - ay) * vy) / length2
        t = np.clip(t, 0.0, 1.0)
        closest_x = ax + t * vx
        closest_y = ay + t * vy
        diff_x = x_flat - closest_x
        diff_y = y_flat - closest_y
        dist2 = diff_x * diff_x + diff_y * diff_y

        length = math.sqrt(length2)
        left_nx = -vy / length
        left_ny = vx / length
        dist = np.sqrt(np.maximum(dist2, 0.0))
        side = (x_flat - ax) * left_nx + (y_flat - ay) * left_ny
        sign = np.where(side < 0.0, -1.0, 1.0)
        signed = dist * sign

        fallback_x = left_nx * sign
        fallback_y = left_ny * sign
        unit_x = np.divide(diff_x, dist, out=fallback_x, where=dist > 1e-9)
        unit_y = np.divide(diff_y, dist, out=fallback_y, where=dist > 1e-9)

        update = dist2 < best_dist2
        best_dist2[update] = dist2[update]
        best_signed[update] = signed[update]
        best_ux[update] = unit_x[update]
        best_uy[update] = unit_y[update]

    return (
        best_signed.reshape(shape),
        best_ux.reshape(shape),
        best_uy.reshape(shape),
    )


def build_displacement_field(
    *,
    bounds_lv95: LV95_BOUNDS,
    height_m: np.ndarray,
    strip: StripCandidate,
    config: CorrectionConfig,
) -> DisplacementField:
    height = np.asarray(height_m, dtype=np.float32)
    xs, ys = pixel_centers_from_bounds(bounds_lv95, height.shape)
    signed_distance, unit_x, unit_y = signed_distance_to_polyline(xs, ys, strip.geometry_xy)
    valid_mask = np.isfinite(signed_distance) & (
        np.abs(signed_distance) <= float(config.max_strip_distance_m)
    )
    signed_distance = signed_distance.copy()
    signed_distance[~valid_mask] = np.inf
    corrected_height_mask = (height > 0.0) & valid_mask
    magnitude = np.zeros(height.shape, dtype=np.float32)
    magnitude[corrected_height_mask] = (
        height[corrected_height_mask]
        * np.abs(signed_distance[corrected_height_mask]).astype(np.float32)
        / float(config.flight_height_m)
    )
    dx_m = (magnitude * unit_x.astype(np.float32)).astype(np.float32, copy=False)
    dy_m = (magnitude * unit_y.astype(np.float32)).astype(np.float32, copy=False)
    return DisplacementField(
        dx_m=dx_m,
        dy_m=dy_m,
        signed_distance_m=signed_distance.astype(np.float32, copy=False),
        height_m=height,
        valid_mask=valid_mask,
        corrected_height_mask=corrected_height_mask,
    )
