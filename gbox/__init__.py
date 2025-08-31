# =================================================================
#                     Exporting classes and functions
# =================================================================

from .base import (
    PointND,
    Point1D,
    Point2D,
    Point3D,
    PointArrayND,
    PointArray1D,
    PointArray2D,
    PointArray3D,
    BoundingBox,
)
from .gshape.gshape import GShape, GShape2D, GShape3D
from .gshape.ellipse import (
    Circle,
    Ellipse,
    CirclesArray,
)

from .utils import configure_axes

__all__ = [
    "PointND",
    "Point1D",
    "Point2D",
    "Point3D",
    "PointArrayND",
    "PointArray1D",
    "PointArray2D",
    "PointArray3D",
    "BoundingBox",
    #
    "GShape",
    "GShape2D",
    "GShape3D",
    #
    "Circle",
    "Ellipse",
    "CirclesArray",
    #
    "configure_axes",
]


# module order
# 0 __init__.py
# gshape
#   b_box, ellipse, lines
#   __init__.py
# 2 base
# 3 utils
