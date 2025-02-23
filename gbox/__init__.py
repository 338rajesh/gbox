# =================================================================
#                     Exporting classes and functions
# =================================================================

from .core import TypeConfig

from .base import (
    BoundingBox,
    PointND,
    Point2D,
    PointArrayND,
    PointArray1D,
    PointArray2D,
    _T_ClosedShape2D,
)


__all__ = [
    'TypeConfig',
    'BoundingBox',
    'PointND',
    'Point2D',
    'PointArrayND',
    'PointArray1D',
    'PointArray2D',
    '_T_ClosedShape2D',
]
