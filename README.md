# OSM Patch Processor

A lightweight Python workflow for downloading, exploring, visualising, and rasterising **OpenStreetMap (OSM)** features for image patches described in a metadata CSV.

This repository is designed for patch-based geospatial workflows, where each image patch has known spatial bounds and OSM data is downloaded for the exact same extent. It supports:

- downloading OSM vectors per patch
- caching vectors as `.gpkg` or `.geojson`
- summarising available OSM semantic keys and their most frequent values
- visualising selected OSM layers for one patch
- handling missing RGB/TIFF images gracefully during plotting
- exporting **binary masks** for chosen classes
- exporting a **single multiclass mask** for chosen classes

---

## Repository contents

The repository includes:

- `osm_patch_processor.py`  
  Main Python class containing the patch processing, downloading, visualisation, and rasterisation logic.

- `osm_demo.ipynb`  
  Example notebook showing the typical workflow step by step.

- `requirements.txt`  
  Python dependencies required to run the code.

- `image_metadata_summary.csv`  
  Patch metadata table containing filenames, image dimensions, CRS, projected bounds, and WGS84 bounds.

---

## Main idea

Each row in the metadata CSV describes **one image patch**.  
The code uses the patch bounds to query OSM and retrieve semantic features such as:

- `building`
- `highway`
- `landuse`
- `natural`
- `waterway`
- `railway`
- `amenity`
- `barrier`
- `man_made`

By default, the processor uses a **presence-only rule** for these semantic keys.  
That means:

- `building=*` → included as building
- `highway=*` → included as highway
- `landuse=*` → included as landuse
- and so on

At this exploratory stage, the subtype values are not required unless you explicitly want to filter them later.

---

## Expected metadata CSV structure

The processor expects the metadata CSV to contain at least the following columns:

- `filename`
- `filepath`
- `crs`
- `width_px`
- `height_px`
- `left`
- `bottom`
- `right`
- `top`
- `left_wgs84`
- `bottom_wgs84`
- `right_wgs84`
- `top_wgs84`

Example columns:

```text
filename, filepath, crs, width_px, height_px, bands, dtype,
pixel_size_x, pixel_size_y, left, bottom, right, top,
center_x, center_y, center_lon, center_lat,
left_wgs84, bottom_wgs84, right_wgs84, top_wgs84
```

The processor uses the **WGS84 bounds** for OSM querying and the **projected bounds + image size** for rasterisation.

---

## Installation

Create and activate your Python environment, then install the requirements:

```bash
pip install -r requirements.txt
```

Typical required packages include:

- `numpy`
- `pandas`
- `geopandas`
- `matplotlib`
- `rasterio`
- `osmnx`

---

## Quick start

### 1. Import the class

```python
from osm_patch_processor import OSMPatchProcessor
```

### 2. Create the processor

```python
processor = OSMPatchProcessor(
    patch_metadata="image_metadata_summary.csv",
    output_dir="osm_patch_outputs",
    verbose=True,
)
```

### 3. Process all patches

```python
results = processor.process_all_patches(
    patch_ids=None,
    semantic_keys=None,
    use_cache=True,
    cache_format="gpkg",
    continue_on_error=True,
    return_results=True,
)
```

This will:

- download or reuse cached OSM vectors for each patch
- summarise available semantic keys
- summarise top values for selected semantic keys
- optionally return the outputs in memory

---

## Visualisation

To visualise a patch, first load the cached vector file:

```python
from pathlib import Path
import geopandas as gpd

patch_id = "aachen_13.tif"
patch_stem = Path(patch_id).stem

patch_gdf = gpd.read_file(
    Path("osm_patch_outputs") / "cache_vectors" / f"{patch_stem}.gpkg"
)
```

Then visualise selected semantic layers:

```python
processor.plot_patch_rgb_and_keys(
    patch_id=patch_id,
    patch_gdf=patch_gdf,
    keys=["building", "highway", "landuse", "barrier", "natural"],
)
```

### If the RGB/TIFF image is missing

The plotting function does **not** crash.  
Instead, it shows a placeholder panel and prints a message explaining how to fix the image path.

To fix it, either:

1. update the `filepath` column in the metadata CSV  
2. pass a local image root when creating the processor

Example:

```python
processor = OSMPatchProcessor(
    patch_metadata="image_metadata_summary.csv",
    output_dir="osm_patch_outputs",
    image_root="/path/to/OpenEarthMap_wo_xBD",
    verbose=True,
)
```
- `outputsample.png` — example visualisation showing the RGB placeholder or image together with selected OSM semantic layers
---

## Binary masks

Binary masks are created **one class at a time**.  
Each output mask contains:

- `0` = background / class absent
- `1` = class present

### Example

```python
binary_masks = processor.export_binary_masks_for_patch(
    patch_id="aachen_13.tif",
    selected_keys=["building", "highway", "waterway"],
    save_png=True,
)
```

### Output location

By default, binary masks are saved inside a patch-specific folder:

```text
osm_patch_outputs/
└── binary_masks/
    └── aachen_13/
        ├── aachen_13_building_mask.tif
        ├── aachen_13_building_mask.png
        ├── aachen_13_highway_mask.tif
        ├── aachen_13_highway_mask.png
        ├── aachen_13_waterway_mask.tif
        └── aachen_13_waterway_mask.png
```

### When to use binary masks

Binary masks are the best first step because they are:

- easier to inspect
- easier to debug
- easier to compare against the OSM layers visually
- useful before deciding how to merge overlapping classes

---

## Multiclass mask

A multiclass mask stores several classes in **one raster**.

### Example

```python
multiclass_mask = processor.export_multiclass_mask_for_patch(
    patch_id="aachen_13.tif",
    selected_keys=["building", "highway", "waterway"],
    save_png=True,
)
```

### Default class ID behaviour

If no explicit `class_id_map` is provided, the class IDs are assigned automatically in the same order as `selected_keys`.

For example:

```python
selected_keys=["building", "highway", "waterway"]
```

becomes:

- `1` = building
- `2` = highway
- `3` = waterway

and:

- `0` = background

### Output location

By default, the multiclass mask is saved inside a patch-specific folder:

```text
osm_patch_outputs/
└── multiclass_masks/
    └── aachen_13/
        ├── aachen_13_multiclass_mask.tif
        └── aachen_13_multiclass_mask.png
```

### Explicit class IDs and priority

You can also control class IDs and overlap priority:

```python
multiclass_mask = processor.export_multiclass_mask_for_patch(
    patch_id="aachen_13.tif",
    selected_keys=["building", "highway", "waterway"],
    class_id_map={
        "building": 1,
        "highway": 2,
        "waterway": 3,
    },
    priority_order=["waterway", "highway", "building"],
    save_png=True,
)
```

### Why priority matters

Different semantic layers can overlap spatially.  
The processor writes classes in the order given by `priority_order`, and **later classes overwrite earlier classes**.

---

## Optional geometry filtering and buffering

At the exploratory stage, the processor keeps semantic classes broad using `key=*`.  
However, you can still refine this later.

For example:

```python
binary_masks = processor.export_binary_masks_for_patch(
    patch_id="aachen_13.tif",
    selected_keys=["building", "highway"],
    geometry_type_map={
        "building": ["Polygon", "MultiPolygon"],
        "highway": ["LineString", "MultiLineString"],
    },
    buffer_map={
        "highway": 1.5,
    },
    save_png=True,
)
```

This can be useful when:

- buildings should be limited to polygonal footprints
- highways should be buffered before rasterisation

---

## Cache and outputs

Typical output structure:

```text
osm_patch_outputs/
├── cache_vectors/
│   ├── aachen_13.gpkg
│   └── ...
├── summaries/
│   ├── aachen_13_key_summary.csv
│   ├── aachen_13_top_values.csv
│   └── ...
├── binary_masks/
│   ├── aachen_13/
│   └── ...
├── multiclass_masks/
│   ├── aachen_13/
│   └── ...
└── quicklooks/
    └── ...
```

### Meaning of each folder

- `cache_vectors/`  
  Cached OSM vector downloads per patch

- `summaries/`  
  Per-patch CSV summaries of available keys and top values

- `binary_masks/`  
  One binary mask per class, stored inside patch-specific folders

- `multiclass_masks/`  
  One multiclass mask per patch, also inside patch-specific folders

- `quicklooks/`  
  Optional saved plots or preview figures

---

## Suggested demo workflow

A simple notebook flow is:

1. import the class
2. create the processor
3. process all patches
4. inspect the returned results object
5. load one cached patch
6. summarise its OSM keys
7. plot selected semantic layers
8. export binary masks
9. export a multiclass mask
10. inspect the saved files on disk

---

