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

# Lazy imports — Numba and rasterio are only loaded when actually used.
# This prevents "DLL load failed" crashes at import time.

_LAZY_IMPORTS = {
    "create_adaptels":   ".adaptels",
    "adaptels_from_array": ".adaptels",
    "read_raster":       ".io",
    "write_raster":      ".io",
    "normalize_layers":  ".io",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
