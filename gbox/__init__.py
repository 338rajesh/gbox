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
    Rectangle,
    CirclesArray,
)

from .utils import configure_axes

# Constants
from .base import PI as PI

__all__ = [
    "PI",
    #
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
    "Rectangle",
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
