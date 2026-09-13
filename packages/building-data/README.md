# building-data

A small standalone package for house/area acquisition, portable input manifests, and a shared browser map picker. It has no Emboss dependency. Emboss, shading-aware-pv, and 3Dlabel each install it into their own environment and use their own storage directory.

```python
from pathlib import Path
from building_data.store import AcquisitionStore

store = AcquisitionStore(Path("./data"), workers=2)
selection = store.acquire({
    "mode": "area", "bbox": [7.0555, 46.7792, 7.0563, 46.7802],
}, progress=print)
house = store.house(selection["houses"][0])
scene = store.export_scene(house.id, Path("./scenes") / house.id)
```

Mount the same API and UI in a FastAPI application with `building_data.api.mount_acquisition(app, store)`. It serves `/acquire`, `/api/acquisition`, `/api/houses`, and `/api/search`. The picker posts a same-origin `building-data-acquired` message containing the prepared house IDs to its parent window. Apps can then reconstruct, simulate, or annotate those houses independently.

## Providers

`BuildingProvider` in `providers.py` defines the extension boundary. A provider owns:

1. Its source revision, metric horizontal CRS, vertical datum, geographic coverage validation, and stable house IDs.
2. A vector adapter returning full roof/wall/floor geometry in the shared semantic surface format, with stable source feature IDs.
3. A LiDAR adapter returning all intersecting source tiles and a source manifest; `points_for_bounds` must retain meaningful repeated returns and respect pinned sources.

Pass an implementation as `AcquisitionStore(..., provider=...)`. Selection coordinates are WGS84; the store transforms them into the provider's declared metric CRS. Provider geometry and vertical coordinates must be mutually compatible. Tests include a provider using a different projected CRS and identity scheme.

`SwissProvider` is the only installed provider. Its source selection, 1 km tile enumeration, national vector database, and year handling live outside the generic store. The current address search and basemap are Swiss. Additional countries need source adapters and map/search configuration; their imagery must also have a correction adapter before they can run the full Emboss method. The Swiss orthophoto implementation lives separately under `orthophoto_correction/` and is installed with the `[imagery]` extra when needed.

## Storage and portable scenes

Inputs are stored in `houses/<id>/house.json` (`building-input-v1`) with paths relative to the data directory. The manifest includes source revision, source tiles/years, EPSG codes, metre units, full house bounds, and the vector file. Acquisition selections, downloaded assets, and jobs are persisted alongside these inputs. Move the entire data directory to preserve cached sources.

`export_scene` writes a `label3d-scene-v1` directory with `scene.json`, `points.las`, and an optional building outline, using metric coordinates and a declared CRS. Once a scene exists, reacquisition preserves it and any annotations. No model inference is required.

Area acquisition includes every building whose full roof intersects the polygon and every LiDAR tile intersecting the area. Each house also acquires whatever neighboring tiles its complete geometry requires. Empty coverage and malformed vector houses produce explicit failures. The shared download semaphore bounds concurrent source downloads across nested house/tile requests.

`uv.lock` in each consuming application pins its environment. The dependency on native GDAL is documented in the [Emboss setup instructions](../../README.md). 3Dlabel installs the base package; Emboss and solar additionally install `[imagery]`.
