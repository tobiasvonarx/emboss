"""Shared serialization helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_ready(
    value: Any,
    *,
    sort_dicts: bool = False,
    missing: str = "keep",
    nonfinite_float_as_none: bool = False,
) -> Any:
    """Convert common scientific Python values to JSON-native values."""

    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if nonfinite_float_as_none and isinstance(value, float) and not math.isfinite(value):
        return None
    if missing != "keep" and not isinstance(value, (dict, list, tuple, set, np.ndarray)):
        try:
            if pd.isna(value):
                return None if missing == "none" else ""
        except (TypeError, ValueError):
            pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda pair: str(pair[0])) if sort_dicts else value.items()
        return {
            str(key): json_ready(
                item,
                sort_dicts=sort_dicts,
                missing=missing,
                nonfinite_float_as_none=nonfinite_float_as_none,
            )
            for key, item in items
        }
    if isinstance(value, (list, tuple)):
        return [
            json_ready(
                item,
                sort_dicts=sort_dicts,
                missing=missing,
                nonfinite_float_as_none=nonfinite_float_as_none,
            )
            for item in value
        ]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
