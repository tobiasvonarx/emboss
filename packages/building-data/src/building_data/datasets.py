"""Public picker configuration, independent of the consuming application."""

from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform


def describe(provider):
    if provider.name == "cambridge":
        project = Transformer.from_crs(provider.crs, 4326, always_xy=True)
        coverage = transform(project.transform, provider.coverage)
        return {
            "id": provider.name,
            "label": "Cambridge · 2023",
            "search": False,
            "center": [coverage.centroid.y, coverage.centroid.x],
            "zoom": 14,
            "coverage": mapping(coverage),
            "bounds": list(coverage.bounds),
            "description": "Zoom in to see building outlines. Select a building or an area inside the boundary.",
            "tiles": "/api/datasets/cambridge/tiles/{z}/{x}/{y}.png",
            "minZoom": 12,
            "attribution": "Imagery © Getmapping 2023 / EDINA Digimap",
            "footprint_attribution": "Building outlines: © Ordnance Survey / EDINA Digimap, May 2023",
        }
    return {
        "id": provider.name,
        "label": "Switzerland",
        "search": True,
        "center": [46.95, 7.45],
        "zoom": 12,
        "coverage": None,
        "description": "Search Switzerland, then select a roof or an area.",
        "tiles": "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg",
        "attribution": "© swisstopo",
    }
