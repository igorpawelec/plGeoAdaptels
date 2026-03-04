# Contributing to plGeoAdaptels

Contributions are welcome! Here's how to get started.

## Setup

```bash
git clone https://github.com/igorpawelec/plgeoadaptels.git
cd plgeoadaptels
conda env create -f environment.yaml
conda activate plgeoadaptels
pip install -e . --no-deps
```

## Development workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Run the test script: `python pycrown_test.py` (or your own test data)
4. Commit: `git commit -m "Add my feature"`
5. Push: `git push origin feature/my-feature`
6. Open a Pull Request on GitHub

## Code style

- Follow PEP 8
- Use type hints where practical
- Numba `@njit(cache=True)` for performance-critical code
- Docstrings in NumPy style

## Reporting bugs

Open an issue on GitHub with:
- Python, NumPy, Numba, Rasterio versions
- Minimal reproducible example
- Full traceback

## License

By contributing you agree that your contributions will be licensed under GPLv3.
