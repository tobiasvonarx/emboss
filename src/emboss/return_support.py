"""Roof-relative LiDAR returns and their nearest-return support partition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from building_data.geometry import clean_polygonal_geometry
from .lidar import residual_thresholds
from .models import ReturnSupportModel
from .models import VectorHouse


def _contains_xy(geometry: Polygon, xy: np.ndarray) -> np.ndarray:
    roof = geometry
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0 or roof.is_empty:
        return np.zeros(len(points), dtype=bool)
    return np.asarray(shapely.intersects_xy(roof, points[:, 0], points[:, 1]), dtype=bool)


def _empty_model() -> ReturnSupportModel:
    roof_band, super_threshold = residual_thresholds(pd.DataFrame())
    return ReturnSupportModel(
        roof_band_m=float(roof_band),
        super_threshold_m=float(super_threshold),
        return_xy=np.empty((0, 2), dtype=np.float64),
        residuals=np.empty(0, dtype=np.float64),
        return_area_weights_m2=np.empty(0, dtype=np.float64),
        source_indices=np.empty(0, dtype=object),
        diagnostics={
            "model_type": "empirical_returns",
            "roof_band_m": float(roof_band),
            "super_threshold_m": float(super_threshold),
            "return_count": 0,
            "support_weight_semantics": "nearest_return_partition_area",
        },
    )


def return_support_partition(
    return_xy: np.ndarray,
    roof: Polygon,
) -> tuple[tuple[BaseGeometry, ...], np.ndarray]:
    """Partition observed roof coverage among returns by nearest neighbour."""

    points = np.asarray(return_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return tuple(), np.empty(0, dtype=np.float64)

    unique_xy: list[tuple[float, float]] = []
    unique_index: dict[tuple[float, float], int] = {}
    inverse: list[int] = []
    multiplicity: list[int] = []
    for x, y in points:
        key = (float(x), float(y))
        index = unique_index.get(key)
        if index is None:
            index = len(unique_xy)
            unique_index[key] = index
            unique_xy.append(key)
            multiplicity.append(0)
        inverse.append(index)
        multiplicity[index] += 1

    sites = MultiPoint(unique_xy)
    observed_domain = clean_polygonal_geometry(sites.convex_hull.intersection(roof))
    if observed_domain.is_empty:
        cells = tuple(Point(x, y) for x, y in points)
        return cells, np.zeros(len(points), dtype=np.float64)

    diagram = shapely.voronoi_polygons(
        sites,
        extend_to=observed_domain.envelope,
        ordered=True,
    )
    unique_cells = tuple(
        clean_polygonal_geometry(cell.intersection(observed_domain))
        for cell in diagram.geoms
    )
    cells = tuple(unique_cells[index] for index in inverse)
    weights = np.asarray(
        [float(unique_cells[index].area) / float(multiplicity[index]) for index in inverse],
        dtype=np.float64,
    )
    return cells, weights


def build_return_support_model(
    observations: pd.DataFrame,
    house: VectorHouse,
) -> ReturnSupportModel:
    """Build roof-relative returns and nearest-return support areas."""

    if observations.empty:
        return _empty_model()

    xy = observations[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    residuals = observations["residual_calibrated"].to_numpy(dtype=np.float64)
    finite = np.all(np.isfinite(xy), axis=1) & np.isfinite(residuals) & _contains_xy(house.roof_envelope, xy)
    filtered = observations.loc[finite]
    if filtered.empty:
        return _empty_model()

    roof_band, super_threshold = residual_thresholds(filtered)
    return_xy = filtered[["x_cal", "y_cal"]].to_numpy(dtype=np.float64)
    residuals = filtered["residual_calibrated"].to_numpy(dtype=np.float64)
    _cells, weights = return_support_partition(return_xy, house.roof_envelope)

    roof_mask = np.abs(residuals) <= float(roof_band)
    elevated_mask = residuals >= float(super_threshold)
    return ReturnSupportModel(
        roof_band_m=float(roof_band),
        super_threshold_m=float(super_threshold),
        return_xy=return_xy,
        residuals=residuals,
        return_area_weights_m2=weights,
        source_indices=filtered.index.to_numpy(dtype=object),
        diagnostics={
            "model_type": "empirical_returns",
            "roof_band_m": float(roof_band),
            "super_threshold_m": float(super_threshold),
            "return_count": int(len(return_xy)),
            "roof_return_count": int(np.count_nonzero(roof_mask)),
            "elevated_return_count": int(np.count_nonzero(elevated_mask)),
            "return_area_weight_total_m2": float(np.sum(weights)),
            "roof_support_weight_m2": float(np.sum(weights[roof_mask])),
            "elevated_support_weight_m2": float(np.sum(weights[elevated_mask])),
            "support_weight_semantics": "nearest_return_partition_area",
        },
    )
