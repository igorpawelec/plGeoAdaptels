# plGeoAdaptels

**Scale-Adaptive Superpixels (Adaptels) for geospatial raster data.**

A pure Python + Numba reimplementation of the plGeoAdaptels algorithm — portable, pip-installable, no compiled binaries required.

## Algorithm

Adaptels are superpixels that **automatically adapt their size** to local image texture: small in complex/textured regions, large in homogeneous areas. The only required parameter is the energy threshold *T*.

Based on:

> R. Achanta, P. Marquez-Neila, P. Fua, S. Süsstrunk,
> *"Scale-Adaptive Superpixels"*, Color and Imaging Conference (CIC26), 2018.

Original C implementation by Paweł Netzel, University of Agriculture in Kraków, Poland.

## Installation

**Recommended (conda + pip):**

```bash
# 1. Install native dependencies via conda
conda install -c conda-forge numpy numba rasterio

# 2. Install plgeoadaptels (no auto-deps, won't break your env)
pip install .          # from cloned repo
# or
pip install --no-deps git+https://github.com/igorpawelec/plgeoadaptels.git
```

**For vectorisation support:**

```bash
conda install -c conda-forge geopandas shapely
```

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

# From numpy arrays (e.g. in Jupyter)
import numpy as np
data = np.random.rand(3, 500, 500)
labels, n = adaptels_from_array(data, threshold=30.0)
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
| 400 × 400 | 3 | 0.03 s | ~580 |
| 200 × 200 | 3 | 0.01 s | ~7 800 |

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- Numba ≥ 0.56
- Rasterio ≥ 1.3

## License

GNU General Public License v3.0 — see [LICENSE.md](LICENSE.md).

## Citation

If you use this software in your research, please cite both the original paper and this implementation. See [CITATION.cff](CITATION.cff).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
