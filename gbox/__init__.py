"""
Geometry Box
============



"""

from .closed_shapes import (
    Circle,
    Ellipse,
    Rectangle,
    Capsule,
    RegularPolygon,
    CShape,
    NLobeShape,
    BoundingBox2D,
    #
    Circles,
    Ellipses,
    Rectangles,
    Capsules,
    RegularPolygons,
    CShapes,
    NLobeShapes,
)
from .gbox import ShapesList, ClosedShapesList, ClosedShape2D, Curve2D, ShapePlotter
from .curves import StraightLine, EllipticalArc, CircularArc

from .points import (
    Point,
    Points,
)

from .utils import PLOT_OPTIONS
