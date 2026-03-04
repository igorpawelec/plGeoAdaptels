"""
plGeoAdaptels — Scale-Adaptive Superpixels for geospatial data.

A pure Python + Numba implementation of the Adaptel algorithm for
creating scale-adaptive superpixels from geospatial raster data.

Based on:
    R. Achanta, P. Marquez-Neila, P. Fua, S. Süsstrunk,
    "Scale-Adaptive Superpixels", CIC26, 2018.

Original C implementation:
    Paweł Netzel, University of Agriculture in Kraków, Poland.

Python + Numba reimplementation:
    Igor Pawelec, Kraków, Poland.

Usage::

    from plgeoadaptels import create_adaptels, adaptels_from_array

    # From GeoTIFF files
    labels, n = create_adaptels('input.tif', 'output.tif', threshold=60.0)

    # From numpy arrays
    labels, n = adaptels_from_array(data_array, threshold=60.0)
"""

try:
    from importlib.metadata import version as _version
    __version__ = _version("plgeoadaptels")
except Exception:
    __version__ = "0.2.0"

__author__ = "Igor Pawelec"

from .adaptels import create_adaptels, adaptels_from_array
from .io import read_raster, write_raster, normalize_layers

__all__ = [
    "create_adaptels",
    "adaptels_from_array",
    "read_raster",
    "write_raster",
    "normalize_layers",
]
