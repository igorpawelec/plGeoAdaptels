# plGeoAdaptels

<img src="https://raw.githubusercontent.com/igorpawelec/plGeoAdaptels/main/www/plGeoAdaptels.png" align="right" width="120"/>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Scale-Adaptive Superpixels (Adaptels) for geospatial raster data.**

A pure Python + Numba reimplementation of the Adaptel algorithm for creating scale-adaptive superpixels from geospatial raster data. Portable, pip-installable, no compiled binaries required.

## Background

Adaptels are superpixels that **automatically adapt their size** to local image texture: small in complex/textured regions, large in homogeneous areas. The only required parameter is the energy threshold *T*.

The algorithm was originally proposed by:

> R. Achanta, P. Marquez-Neila, P. Fua, S. Süsstrunk,
> *"Scale-Adaptive Superpixels"*, Color and Imaging Conference (CIC26), 2018.

The original C implementation (`plGeoAdaptels`) was developed by **Paweł Netzel** at the University of Agriculture in Kraków, Poland. This Python package is a faithful reimplementation of that C code using Numba JIT compilation for near-native performance.

The algorithm was applied to standing dead tree detection in:

> I. Pawelec, J. Socha, P. Netzel,
> *"Superpixel-based segmentation of standing dead trees from aerial photographs"*,
> International Journal of Applied Earth Observation and Geoinformation, 2026.

## Installation

**Recommended (conda + pip):**

```bash
# 1. Install native dependencies via conda
conda install -c conda-forge numpy numba rasterio fiona

# 2. Install plgeoadaptels
pip install --no-deps .          # from cloned repo
# or
pip install --no-deps git+https://github.com/igorpawelec/plgeoadaptels.git
```

> **Note:** Dependencies are installed via conda to avoid conflicts with GDAL/PROJ native libraries. The `--no-deps` flag prevents pip from overwriting conda packages.

## Quick start

### Python API

```python
from plgeoadaptels import create_adaptels, adaptels_from_array

# From GeoTIFF — reads, segments, writes output
labels, n = create_adaptels("input.tif", "output.tif", threshold=60.0)
print(f"Created {n} adaptels")

# Multiple input layers (treated as bands)
labels, n = create_adaptels(
    ["band1.tif", "band2.tif", "band3.tif"],
    "adaptels.tif",
    threshold=40.0,
    normalize=True,
)

# From numpy arrays (no file I/O)
import numpy as np
data = np.random.rand(3, 500, 500)
labels, n = adaptels_from_array(data, threshold=30.0)
```

### Vectorization (no geopandas needed)

```python
from plgeoadaptels.vectorize import vectorize_from_file

# Adaptel raster → Shapefile (or .gpkg, .geojson)
n_poly = vectorize_from_file("output.tif", "adaptels.shp")
```

### Command line

```bash
plgeoadaptels -i input.tif -o output.tif -t 60.0
plgeoadaptels -i b1.tif -i b2.tif -o result.tif -t 40.0 -8 -d cosine -n
```

### As Python module

```bash
python -m plgeoadaptels -i input.tif -o output.tif -t 60.0
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `threshold` | 60.0 | Energy threshold *T*. Lower → smaller adaptels, higher → larger |
| `distance` | `'minkowski'` | Distance metric: `'minkowski'`, `'cosine'`, `'angular'` |
| `minkowski_p` | 2.0 | Minkowski *p* parameter (2.0 = Euclidean) |
| `queen_topology` | `False` | `True` = 8-connectivity, `False` = 4-connectivity |
| `normalize` | `False` | Normalize inputs to [0, 1] before processing |

## Performance

First call includes Numba JIT compilation (~5 s). Subsequent calls run at near-C speed:

| Raster | Bands | Time | Adaptels |
|---|---|---|---|
| 400 × 400 | 3 | 0.03 s | ~2 800 |
| 2000 × 2000 | 3 | ~1 s | ~70 000 |

## Repository structure

```
plgeoadaptels/
├── plgeoadaptels/        # Package source
│   ├── __init__.py       # Public API (lazy imports)
│   ├── __main__.py       # CLI entry point
│   ├── adaptels.py       # High-level API
│   ├── cli.py            # Command-line interface
│   ├── core.py           # Numba JIT algorithm kernel
│   ├── io.py             # GeoTIFF read/write (rasterio)
│   └── vectorize.py      # Raster→vector (rasterio + fiona)
├── test_data/            # Sample rasters
├── www/                  # Logo & assets
├── pyproject.toml
├── environment.yaml
├── CITATION.cff
├── CHANGELOG.md
├── tests/                # Pytest suite
├── CONTRIBUTING.md
└── LICENSE
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- Numba ≥ 0.56
- Rasterio ≥ 1.3
- Fiona ≥ 1.9 *(for vectorization)*

## Citation

If you use this software in your research, please cite:

1. **This implementation:**

   > Pawelec, I. (2026). plGeoAdaptels — Scale-Adaptive Superpixels for geospatial data [Software]. https://github.com/igorpawelec/plgeoadaptels

2. **The original algorithm:**

   > Achanta, R., Marquez-Neila, P., Fua, P., & Süsstrunk, S. (2018). Scale-Adaptive Superpixels. *Color and Imaging Conference (CIC26)*.

3. **The original C implementation:**

   > Netzel, P. plGeoAdaptels [Software]. University of Agriculture in Kraków. https://github.com/pawelnetzel/plGeoAdaptels

See also [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
