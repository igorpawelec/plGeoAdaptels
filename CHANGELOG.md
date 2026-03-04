# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] — 2026-03-04

### Added
- Full repo structure (pyproject.toml, CITATION.cff, CONTRIBUTING.md, CI)
- GitHub Actions CI smoke test
- `environment.yaml` for reproducible conda environments
- `requirements.txt` for reference
- `adaptels_from_array()` for numpy-only workflows
- `--no-deps` installation guide to avoid breaking conda envs

### Changed
- Replaced `setup.py` with `pyproject.toml`
- Single-source versioning via `importlib.metadata`
- Empty `dependencies` in pyproject.toml (install via conda)
- Version bump to 0.2.0

## [0.1.0] — 2026-03-04

### Added
- Initial Python + Numba reimplementation of plGeoAdaptels
- `create_adaptels()` — GeoTIFF file-based API
- `adaptels_from_array()` — numpy array API
- CLI: `plgeoadaptels -i input.tif -o output.tif -t 60`
- Distance metrics: Minkowski, cosine, angular
- 4-connectivity (rook) and 8-connectivity (queen)
- Layer normalization option
- Min-heap priority queue (Numba JIT)
- Rasterio-based I/O with nodata handling
