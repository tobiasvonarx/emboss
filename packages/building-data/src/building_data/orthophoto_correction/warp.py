"""Inverse image warp helpers for Bildsturz correction."""

from __future__ import annotations

import numpy as np

from building_data.orthophoto_correction.models import DisplacementField
from building_data.orthophoto_correction.models import LV95_BOUNDS


def _pixel_size(bounds_lv95: LV95_BOUNDS, shape: tuple[int, int]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_lv95]
    height, width = int(shape[0]), int(shape[1])
    return (max_x - min_x) / float(width), (max_y - min_y) / float(height)


def source_pixel_coordinates(field: DisplacementField, bounds_lv95: LV95_BOUNDS) -> tuple[np.ndarray, np.ndarray]:
    height, width = field.dx_m.shape
    pixel_width, pixel_height = _pixel_size(bounds_lv95, (height, width))
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    source_x = cols + field.dx_m.astype(np.float32) / float(pixel_width)
    source_y = rows - field.dy_m.astype(np.float32) / float(pixel_height)
    return source_x, source_y


def _bilinear_sample_2d(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, fill_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(image)
    height, width = array.shape[:2]
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    valid = (xs >= 0.0) & (xs <= width - 1.0) & (ys >= 0.0) & (ys <= height - 1.0)
    output = np.full(xs.shape, float(fill_value), dtype=np.float64)
    if not np.any(valid):
        return output, valid

    xv = xs[valid]
    yv = ys[valid]
    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = xv - x0
    wy = yv - y0
    output[valid] = (
        (1.0 - wx) * (1.0 - wy) * array[y0, x0]
        + wx * (1.0 - wy) * array[y0, x1]
        + (1.0 - wx) * wy * array[y1, x0]
        + wx * wy * array[y1, x1]
    )
    return output, valid


def nearest_sample_bool(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(mask, dtype=bool)
    height, width = array.shape
    col = np.rint(xs).astype(np.int64)
    row = np.rint(ys).astype(np.int64)
    valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    sampled = np.zeros(xs.shape, dtype=bool)
    sampled[valid] = array[row[valid], col[valid]]
    return sampled, valid


def warp_rgb_with_displacement(
    rgb: np.ndarray,
    field: DisplacementField,
    bounds_lv95: LV95_BOUNDS,
    *,
    fill_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb)
    source_x, source_y = source_pixel_coordinates(field, bounds_lv95)
    if image.ndim == 2:
        sampled, valid = _bilinear_sample_2d(image.astype(np.float64), source_x, source_y, fill_value=float(fill_value))
        return np.clip(np.rint(sampled), 0, 255).astype(image.dtype), valid

    output = np.empty_like(image)
    valid_mask = np.zeros(image.shape[:2], dtype=bool)
    for channel in range(image.shape[2]):
        sampled, valid = _bilinear_sample_2d(
            image[:, :, channel].astype(np.float64),
            source_x,
            source_y,
            fill_value=float(fill_value),
        )
        output[:, :, channel] = np.clip(np.rint(sampled), 0, 255).astype(image.dtype)
        valid_mask |= valid
    return output, valid_mask


def warp_mask_with_displacement(
    mask: np.ndarray,
    field: DisplacementField,
    bounds_lv95: LV95_BOUNDS,
) -> tuple[np.ndarray, np.ndarray]:
    source_x, source_y = source_pixel_coordinates(field, bounds_lv95)
    return nearest_sample_bool(mask, source_x, source_y)


def source_occupancy_mask(active_target_mask: np.ndarray, field: DisplacementField, bounds_lv95: LV95_BOUNDS) -> np.ndarray:
    active = np.asarray(active_target_mask, dtype=bool)
    source_x, source_y = source_pixel_coordinates(field, bounds_lv95)
    rows, cols = np.nonzero(active)
    output = np.zeros(active.shape, dtype=bool)
    if len(rows) == 0:
        return output
    source_cols = np.rint(source_x[rows, cols]).astype(np.int64)
    source_rows = np.rint(source_y[rows, cols]).astype(np.int64)
    valid = (
        (source_cols >= 0)
        & (source_cols < active.shape[1])
        & (source_rows >= 0)
        & (source_rows < active.shape[0])
    )
    output[source_rows[valid], source_cols[valid]] = True
    return output


def compute_occlusion_mask(active_target_mask: np.ndarray, field: DisplacementField, bounds_lv95: LV95_BOUNDS) -> np.ndarray:
    source_x, source_y = source_pixel_coordinates(field, bounds_lv95)
    source_occupied = source_occupancy_mask(active_target_mask, field, bounds_lv95)
    sampled_source_occupied, valid = nearest_sample_bool(source_occupied, source_x, source_y)
    active = np.asarray(active_target_mask, dtype=bool)
    return sampled_source_occupied & ~active & valid
