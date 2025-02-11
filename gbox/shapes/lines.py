from .. import (
    Point,
    TopologicalCurve,
    PointND,
    Point2D,
)
from ..base import PointType


class StraightLine(TopologicalCurve):
    """Base class for all straight lines"""

    def __init__(self, p1: PointType, p2: PointType):
        super(StraightLine, self).__init__()
        self.p1: PointND = Point.from_seq(p1)
        self.p2: PointND = Point.from_seq(p2)

    def length(self) -> float:
        return self.p1.distance_to(self.p2)

    def equation(self):
        p, q = self.p1.as_array(), self.p2.as_array()
        direction = q - p

        def _line_eqn(t):
            return p + t * direction

        return _line_eqn


class StraightLine2D(StraightLine):
    def __init__(self, p1: PointType, p2: PointType):
        assert len(p1) == 2 and len(p2) == 2, "Expecting 2D points"
        super(StraightLine2D, self).__init__(p1, p2)

        self.p1 = Point2D.from_seq(p1)
        self.p2 = Point2D.from_seq(p2)

    def angle(self, rad=True) -> float:
        """Returns the angle of the line w.r.t positive x-axis in [0, 2 * pi]"""
        return self.p1.angle(self.p2, rad)
