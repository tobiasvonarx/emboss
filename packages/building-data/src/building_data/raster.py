"""Rasterization utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Polygon


def xy_to_pixel(
    xy: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    """Map LV95 coordinates to fractional pixel-center coordinates.

    The extent describes the outer pixel edges (GDAL convention), so pixel
    (0, 0) is centered half a pixel inside the extent corner.
    """

    pts = np.asarray(xy, dtype=np.float64)
    min_x, max_x, min_y, max_y = [float(value) for value in extent]
    x = (pts[:, 0] - min_x) / max(max_x - min_x, 1e-9) * float(width) - 0.5
    y = (max_y - pts[:, 1]) / max(max_y - min_y, 1e-9) * float(height) - 0.5
    return np.column_stack([x, y])


def pixel_to_xy(
    pixels: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    pts = np.asarray(pixels, dtype=np.float64)
    min_x, max_x, min_y, max_y = [float(value) for value in extent]
    x = min_x + (pts[:, 0] + 0.5) / max(float(width), 1.0) * (max_x - min_x)
    y = max_y - (pts[:, 1] + 0.5) / max(float(height), 1.0) * (max_y - min_y)
    return np.column_stack([x, y])


def rasterize_polygon(
    polygon: Polygon,
    *,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(image)
    polygons: Iterable[Polygon]
    if polygon.geom_type == "MultiPolygon":
        polygons = polygon.geoms
    else:
        polygons = (polygon,)
    for geom in polygons:
        if geom.is_empty:
            continue
        exterior = xy_to_pixel(np.asarray(geom.exterior.coords), extent=extent, width=width, height=height)
        draw.polygon([tuple(map(float, point)) for point in exterior], fill=255)
        for interior in geom.interiors:
            hole = xy_to_pixel(np.asarray(interior.coords), extent=extent, width=width, height=height)
            draw.polygon([tuple(map(float, point)) for point in hole], fill=0)
    return np.asarray(image, dtype=np.uint8) > 0


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    return _connected_components(
        mask,
        neighbor_offsets=(
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ),
    )


def edge_connected_components(mask: np.ndarray) -> list[np.ndarray]:
    return _connected_components(
        mask,
        neighbor_offsets=((-1, 0), (0, -1), (0, 1), (1, 0)),
    )


def _connected_components(
    mask: np.ndarray,
    *,
    neighbor_offsets: tuple[tuple[int, int], ...],
) -> list[np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask_bool, dtype=bool)
    components: list[np.ndarray] = []
    h, w = mask_bool.shape
    for start_row, start_col in zip(*np.nonzero(mask_bool & ~seen), strict=False):
        if seen[start_row, start_col] or not mask_bool[start_row, start_col]:
            continue
        coords = []
        stack = [(int(start_row), int(start_col))]
        seen[start_row, start_col] = True
        while stack:
            row, col = stack.pop()
            coords.append((row, col))
            for row_offset, col_offset in neighbor_offsets:
                nr, nc = row + row_offset, col + col_offset
                if nr < 0 or nr >= h or nc < 0 or nc >= w or seen[nr, nc] or not mask_bool[nr, nc]:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))
        components.append(np.asarray(coords, dtype=np.int64))
    return components
