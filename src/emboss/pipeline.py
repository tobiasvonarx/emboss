"""The original reconstruction method, on explicit prepared inputs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .export import export_house_result
from .models import HouseResult, VectorHouse
from .return_support import build_return_support_model
from .superstructure_fitting import fit_superstructures

def run_emboss_house(
    *,
    output_dir: str | Path,
    workspace_label: str,
    house: VectorHouse,
    observations: pd.DataFrame,
    rgb: object,
    image_superstructure_mask: object,
    image_superstructure_class_map: object,
    image_segmentation_source: str,
    vector_roof_mask: object,
    extent_lv95: tuple[float, float, float, float],
    crs: str = "EPSG:2056",
) -> HouseResult:
    """Run the shared Emboss fitting/export core on prepared canonical inputs."""

    support_model = build_return_support_model(observations, house)
    solids = fit_superstructures(
        observations,
        house,
        support_model,
        image_class_map=image_superstructure_class_map,
        image_extent_lv95=extent_lv95,
    )
    return export_house_result(
        output_dir=Path(output_dir),
        workspace_label=workspace_label,
        house=house,
        solids=solids,
        support_model=support_model,
        rgb=rgb,
        image_superstructure_mask=image_superstructure_mask,
        image_superstructure_class_map=image_superstructure_class_map,
        image_segmentation_source=image_segmentation_source,
        vector_roof_mask=vector_roof_mask,
        extent_lv95=extent_lv95,
        crs=crs,
    )
