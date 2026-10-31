"""One imagery preparation path shared by reconstruction and preview galleries."""

from types import SimpleNamespace

import laspy
from building_data.geometry import load_vector_house
from building_data.orthophoto_correction.geometry import crop_bounds_lv95
from building_data.orthophoto_correction.pipeline import correct_house_orthophoto
from building_data.orthophoto_correction.prepare import prepare_assets


def prepare_house_image(client, item, config, progress):
    house = load_vector_house(item.surfaces_path, item.building_fid)
    if getattr(item, "provider", "switzerland") != "switzerland":
        provider = client.store.provider_for_house(item)
        return provider.orthophoto(house, config), house
    center = house.roof_envelope.centroid
    source = next(
        (
            source
            for source in item.sources
            if source["bounds"][0] <= center.x < source["bounds"][2]
            and source["bounds"][1] <= center.y < source["bounds"][3]
        ),
        item.sources[0],
    )
    with laspy.open(source["path"]) as reader:
        bounds = (
            *map(float, reader.header.mins[:2]),
            *map(float, reader.header.maxs[:2]),
        )
    assets = prepare_assets(
        tile_key=source["tile"],
        bounds_lv95=bounds,
        cache=config.cache_root,
        config=config,
        progress=progress,
        required_bounds_lv95=crop_bounds_lv95(
            house.bounds_xy, padding_m=8.0, min_side_m=config.min_crop_side_m
        ),
    )
    workspace = SimpleNamespace(
        tile_key=source["tile"],
        tile_bounds_xy=bounds,
        correction_raster_assets=assets,
    )
    progress("Previewing source correction")
    with client._inference_lock:
        corrected = correct_house_orthophoto(
            {"house": house}, workspace, padding_m=8.0, config=config
        )
    return corrected, house
