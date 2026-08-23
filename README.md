# gbox:: Geometry Box

[![codecov](https://codecov.io/gh/338rajesh/gbox/graph/badge.svg)](https://codecov.io/gh/338rajesh/gbox)
[![Documentation Status](https://readthedocs.org/projects/gbox/badge/?version=latest)](https://gbox.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/338rajesh/gbox/actions/workflows/ci.yml/badge.svg)](https://github.com/338rajesh/gbox/actions)

A simple Python package for working with geometry related operations.
See documentation at [gbox.readthedocs.io](https://gbox.readthedocs.io)

## Installation

```bash
pip install gbox
```

## Usage

```py
import gbox
# Create a 2D point
from gbox import Point2D
point = Point2D(1.0, 2.0)
# Create a Cirle with center at point and radius 5.0
from gbox import Circle
circle = Circle(5.0, point)

```

## Notes

- Implement `__hash__` and `__eq__` for using points in sets and dicts
- Stick to numpy for now. Try with Numba in later releases.
- Use just `PointND` and `PointArrayND`, instead of `1D`, `2D`, `3D` etc.

## For Developers

### Install dependencies

Started using [uv](https://docs.astral.sh/uv/) for Python Package and Project management.

```bash
uv sync
```

### Add New dependencies

```bash
uv add package==version
```

### Add Dev-Dependency

```bash
uv add --group dev-dependencies dev-package==version
```

### Run Tests

```bash
uv run --group dev-dependencies pytest --cov=gbox tests/
```

> Check [uv](https://docs.astral.sh/uv/)'s documentation for more details




