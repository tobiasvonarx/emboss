"""Orthophoto-derived roof superstructure segmentation support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from building_data.raster import connected_components
from emboss.segmentation.schema import RID2_BACKGROUND_CLASS_ID
from emboss.segmentation.schema import RID2_SUPERSTRUCTURE_CLASSES


CLASS_NAMES = {item.class_id: item.name for item in RID2_SUPERSTRUCTURE_CLASSES}
IMAGE_CLASS_IDS = tuple(
    int(item.class_id)
    for item in RID2_SUPERSTRUCTURE_CLASSES
    if int(item.class_id) != int(RID2_BACKGROUND_CLASS_ID)
)
MIN_IMAGE_FOOTPRINT_AREA_M2 = 0.03
AREA_COMPARISON_EPSILON_M2 = 1e-9


@dataclass(frozen=True)
class ImageClassComponent:
    class_id: int
    class_label: str
    geometry: BaseGeometry


def image_class_names() -> tuple[str, ...]:
    return tuple(CLASS_NAMES[class_id] for class_id in IMAGE_CLASS_IDS)


def image_class_components(
    *,
    class_map: np.ndarray,
    extent_lv95: tuple[float, float, float, float],
    class_ids: tuple[int, ...] = IMAGE_CLASS_IDS,
) -> tuple[ImageClassComponent, ...]:
    labels = np.asarray(class_map, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D superstructure class map, got shape {tuple(labels.shape)}")

    components: list[ImageClassComponent] = []
    for class_id in class_ids:
        class_id = int(class_id)
        if class_id == RID2_BACKGROUND_CLASS_ID:
            continue
        for component in connected_components(labels == class_id):
            geometry = _component_polygon(
                component,
                shape_hw=labels.shape,
                extent_lv95=extent_lv95,
            )
            if geometry.is_empty:
                continue
            components.append(
                ImageClassComponent(
                    class_id=class_id,
                    class_label=CLASS_NAMES.get(class_id, str(class_id)),
                    geometry=geometry,
                )
            )
    return tuple(components)


def _component_polygon(
    component: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    extent_lv95: tuple[float, float, float, float],
) -> BaseGeometry:
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    rows_by_index: dict[int, list[int]] = {}
    for row, col in np.asarray(component, dtype=np.int64):
        rows_by_index.setdefault(int(row), []).append(int(col))

    boxes = []
    for row, cols_raw in rows_by_index.items():
        cols = np.asarray(sorted(set(cols_raw)), dtype=np.int64)
        split_at = np.flatnonzero(np.diff(cols) != 1) + 1
        for run in np.split(cols, split_at):
            if run.size:
                boxes.append(
                    _pixel_box(
                        row_min=row,
                        row_max=row + 1,
                        col_min=int(run[0]),
                        col_max=int(run[-1]) + 1,
                        width=width,
                        height=height,
                        extent_lv95=extent_lv95,
                    )
                )
    return unary_union(boxes).buffer(0) if boxes else Polygon()


def _pixel_box(
    *,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    width: int,
    height: int,
    extent_lv95: tuple[float, float, float, float],
) -> Polygon:
    min_x, max_x, min_y, max_y = (float(value) for value in extent_lv95)
    x0 = min_x + float(col_min) / float(width) * (max_x - min_x)
    x1 = min_x + float(col_max) / float(width) * (max_x - min_x)
    y1 = max_y - float(row_min) / float(height) * (max_y - min_y)
    y0 = max_y - float(row_max) / float(height) * (max_y - min_y)
    return box(x0, y0, x1, y1)
