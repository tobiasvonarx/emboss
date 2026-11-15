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

`SwissProvider` supplies Swiss source selection, 1 km tile enumeration, national
vectors and year handling. `CambridgeProvider` reads a locally installed 2023
source bundle, MasterMap footprints and orthorectified imagery, and prepares
City3D scaffolds. See the Emboss README for `building-data import-cambridge`.

Cambridge point selections accept an optional `source_ids` array of MasterMap
TOIDs. `/api/datasets/cambridge/target-preview` resolves these IDs against the
installed source and returns the selected footprint and unselected neighbours.
The browser uses this explicit selection instead of assuming the nearest single
feature is a whole building. Selected parts reconstruct atomically as one target;
area mode continues to prepare each intersecting source part separately.

Providers may include a `Target` layer alongside `Roof`, `Wall`, and `Floor` in
the surfaces GeoPackage. Its feature ID is the target ID; geometry is the original
source footprint. `scope`, JSON `source_ids`, JSON `touching_ids`, and JSON
`coverage` preserve target identity, topology and measured coverage independently
of reconstruction. `build_house_index` and `load_vector_house` prefer this source
topology; providers without the layer retain their existing behaviour. The store
saves the target metadata in `HouseInput.target` for downstream provenance.

`/api/datasets` describes installed providers, map imagery, search availability
and coverage. The shared `/acquire` picker sends the provider ID with each
selection. The store resolves a provider per acquisition and from each saved
house manifest; switching the browser's dataset never mutates a running job's
provider. Applications install this package rather than copying the picker.

Providers may implement `build_selection(geometry, mode, output, progress)` to
materialize only the selected scaffolds, `footprints(bounds)` for the picker
overlay, and `orthophoto(house, config)` for supplied imagery. Swiss imagery
continues through the separate flight-strip correction under
`orthophoto_correction/`. Preview and reconstruction use one preparation path.
Each app retains its own acquisition manifests, jobs, results and derived caches.

## Storage and portable scenes

Inputs are stored in `houses/<id>/house.json` (`building-input-v1`) with paths relative to the data directory. The manifest includes source revision, source tiles/years, EPSG codes, metre units, full house bounds, and the vector file. Acquisition selections, downloaded assets, and jobs are persisted alongside these inputs. Move the entire data directory to preserve cached sources.

`export_scene` writes a `label3d-scene-v1` directory with `scene.json`, `points.las`, and an optional building outline, using metric coordinates and a declared CRS. Once a scene exists, reacquisition preserves it and any annotations. No model inference is required.

Area acquisition includes every building whose full roof intersects the polygon and every LiDAR tile intersecting the area. Each house also acquires whatever neighboring tiles its complete geometry requires. Empty coverage and malformed vector houses produce explicit failures. The shared download semaphore bounds concurrent source downloads across nested house/tile requests.

`uv.lock` in each consuming application pins its environment. The dependency on native GDAL is documented in the [Emboss setup instructions](../../README.md). 3Dlabel installs the base package; Emboss and solar additionally install `[imagery]`.
