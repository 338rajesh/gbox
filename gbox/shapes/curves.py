"""
Curves modules implements creation, modification and other related
operations, with major support upto 3D space while making an attempt
to work with arbitrary dimensions.
"""

from .. import (
    PointND,
    Point2D,
    PointArrayND,
)
from ..base import _T_CurveND
from ..core import float_type


class StraightLineND(_T_CurveND):
    """
    Constructs a straight line from two points lying in n-dimensional
    space. If the points are not of type PointND, they will be converted
    to PointND else they will be used as they are. This will ensure that
    if a subclass of PointND is used, it will be used as it is, but if
    the supplied points are not of type PointND, they will be converted
    to PointND.

    Parameters
    ----------
    p1, p2 : list[float] | PointND
        First and second point of the line
    """

    def __init__(
            self, p1: list[float_type] | PointND,
            p2: list[float_type] | PointND
    ):
        self.p1 = p1 if isinstance(p1, Point2D) else PointND.from_(p1)
        self.p2 = p2 if isinstance(p2, Point2D) else PointND.from_(p2)
        _points = PointArrayND.from_points(self.p1, self.p2)
        super(StraightLineND, self).__init__(_points)

    @property
    def length(self) -> float_type:
        """Length of the line in n-dimensional euclidean space"""
        return self.p1.distance_to(self.p2)

    @property
    def dim(self) -> int:
        """Dimension of the line"""
        return self.p1.dim

    def equation(self):
        p, q = self.p1.as_array(), self.p2.as_array()
        direction = q - p

        def _line_eqn(t):
            return p + t * direction

        return _line_eqn


class StraightLine2D(StraightLineND):
    def __init__(
        self,
        p1: list[float_type] | Point2D,
        p2: list[float_type] | Point2D
    ):
        if len(p1) != 2:
            raise ValueError(f"Expecting 2D points, but first point is {p1}D")
        if len(p2) != 2:
            raise ValueError(f"Expecting 2D points, but second point is {p2}D")

        self.p1: Point2D = Point2D.from_(p1)
        self.p2: Point2D = Point2D.from_(p2)
        super(StraightLine2D, self).__init__(self.p1, self.p2)

    def angle(self, rad=True) -> float_type:
        """Returns the angle of the line w.r.t positive x-axis in [0, 2 * pi]
        """
        return self.p1.angle(self.p2, rad)


class BezierCurveND(_T_CurveND):
    pass


class RationalBezierCurveND(BezierCurveND):
    pass
