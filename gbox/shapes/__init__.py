"""
gshape, a subpackage of gbox, provides geometric shapes and their operations.
"""

from .shapes_2d import (
    Circle,
    Ellipse,
    CirclesArray,
    Shape2D,
    Shapes2DArray,
)

SHAPES_2D_MAPPING: dict[str, Shape2D] = {
    "circle": Circle,
    "ellipse": Ellipse,
}
SHAPES_2D_ARRAY_MAPPING: dict[str, Shapes2DArray] = {"circles_array": CirclesArray}

__all__ = [
    "Shape2D",
    "Shapes2DArray",
    "Circle",
    "Ellipse",
    "CirclesArray",
]
