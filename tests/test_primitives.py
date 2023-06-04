from geometry_box import Vector, Point


class TestPoint:
    p1 = Point(0.0, 0.0)
    p2 = Point(2.0, 3.6, 5, 6.9, -0.3)

    def test_point_dimension(self):
        assert self.p1.dim == 2
        assert self.p2.dim == 5


class TestLineSegment:
    l1 = Vector(Point(0.0, 0.0), Point(3.0, 4.0))

    def test_ls_length(self):
        assert self.l1.len() == 5.0
        return
