"""Curves modules implements creation, modification and other related
operations, with major support upto 3D space while making an attempt
to work with arbitrary dimensions.
"""

from .base import (
    Sequence,
    #
    DEFAULT_FLOAT,
    PointND,
    PointArrayND,
    Point2D,
)


class _TopologicalCurveND:
    def __init__(self, points: PointArrayND):
        self.points = points


class LineSegmentND(_TopologicalCurveND):
    """Constructs a straight line from two points lying in n-dimensional
    space.

    Parameters
    ----------
    p1, p2 : list[float] | PointND
        First and second point of the line

    """

    def __init__(
        self, p1: Sequence[float] | PointND, p2: Sequence[float] | PointND
    ):
        self.p1 = PointND._from_(p1)
        self.p2 = PointND._from_(p2)
        super().__init__(PointArrayND.from_points([self.p1, self.p2]))

    def __len__(self) -> DEFAULT_FLOAT:
        return self.length

    @property
    def length(self) -> DEFAULT_FLOAT:
        """Length of the line in n-dimensional euclidean space"""
        return self.p1.distance_to(self.p2)

    @property
    def dim(self) -> int:
        """Dimension of the line"""
        return self.p1.dim

    def equation(self):
        direction = self.p2.coor - self.p1.coor

        def _line_eqn(t):
            return self.p1.coor + t * direction

        return _line_eqn


class LineSegment2D(LineSegmentND):
    def __init__(
        self,
        p1: Sequence[float] | Point2D,
        p2: Sequence[float] | Point2D,
    ):
        if len(p1) != 2:
            raise ValueError(f"Expecting 2D points, but first point is {p1}D")
        if len(p2) != 2:
            raise ValueError(f"Expecting 2D points, but second point is {p2}D")

        self.p1: Point2D = Point2D._from_(p1)
        self.p2: Point2D = Point2D._from_(p2)
        super().__init__(self.p1, self.p2)

    def angle(self, rad=True) -> DEFAULT_FLOAT:
        """Returns the angle of the line w.r.t positive x-axis in [0, 2 * pi]"""
        return self.p1.angle(self.p2, rad)
