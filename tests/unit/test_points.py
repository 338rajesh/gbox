import numpy as np
import gbox as gb
import pytest

# ==================== FIXTURES =====================


@pytest.fixture
def point():
    return gb.Point(1.0, 2.0)


@pytest.fixture
def origin():
    return gb.Point(0.0, 0.0)


@pytest.fixture
def point_3d():
    return gb.Point(1.0, 2.0, -3.0)


@pytest.fixture
def point_2d():
    return gb.Point2D(1.0, 2.0)


@pytest.fixture
def points():
    return gb.Points([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


@pytest.fixture
def points_2d():
    return gb.Points2D([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


@pytest.fixture(params=[0.0, 0.2, 1.24, 2.7, 3.14, 4.0])
def transformation_triplet(request):
    return (np.pi * 0.5 * request.param, np.random.rand(), np.random.rand())


# ==================== TESTS =====================


class TestPoint:
    def test_point(self):
        p = gb.Point(1.0, 2.0)

    def test_point_1(self):
        with pytest.raises(AssertionError):
            gb.Point(1.0, 2.0, 3)

    def test_point_eq(self, point):
        assert point == gb.Point(1.0, 2.0)

    def test_point_repr(self, point):
        assert str(point) == "Point:  (1.0, 2.0)"

    def test_point_add(self, point):
        assert point + point == gb.Point(2.0, 4.0)

    def test_point_sub(self, point):
        assert point - point == gb.Point(0.0, 0.0)

    def test_adding_different_dimensions(self, point, point_3d):
        with pytest.raises(AssertionError):
            point + point_3d

    def test_distance_to(self, point, origin):
        assert point.distance_to(origin) == pytest.approx(2.23606, rel=1e-5)

    def test_point_in_bounds(self, point, origin):
        assert point.in_bounds(gb.BoundingBox(origin, origin + point + point))
        assert not origin.in_bounds(gb.BoundingBox(point, point + point))

    def test_point_on_bounds(self, point, origin):
        assert point.in_bounds(
            gb.BoundingBox(origin, origin + point), include_bounds=True
        )
        assert not origin.in_bounds(
            gb.BoundingBox(origin, origin + point), include_bounds=False
        )


class TestPoint2D:
    def test_point_2d(self, point_2d):
        assert point_2d.dim == 2
        assert point_2d.x == 1.0
        assert point_2d.y == 2.0
        assert point_2d == gb.Point2D(1.0, 2.0)

    def test_slope(self, point_2d):
        assert point_2d.slope(gb.Point2D(3.0, 4.0)) == pytest.approx(1.0, rel=1e-5)
        assert point_2d.slope(gb.Point2D(-1.0, -6.0)) == pytest.approx(4.0, rel=1e-5)


class TestPoints:
    def test_points_constructor(self):
        gb.Points(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        gb.Points([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        gb.Points(([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]))
        gb.Points(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))

    def test_points_properties(self, points):
        assert points.dim == 2
        assert len(points) == 4

    def test_points_repr(self, points):
        out = "Points:\n[[1. 2.]\n [3. 4.]\n [5. 6.]\n [7. 8.]]"
        assert str(points) == out

    def test_points_eq(self, points):
        assert points == gb.Points([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    def test_points_copy(self, points):
        assert points.copy() == points
        # Check that the copy is not the same object
        assert points.copy() is not points

    def test_points_bounding_box(self, points):
        assert points.bounding_box == gb.BoundingBox([1.0, 2.0], [7.0, 8.0])


class TestPoints2D:
    def test_constructor(self, points_2d):
        assert points_2d.dim == 2
        assert np.array_equal(points_2d.x, [1.0, 3.0, 5.0, 7.0])
        assert np.array_equal(points_2d.y, [2.0, 4.0, 6.0, 8.0])

    def test_transform(self, points_2d, transformation_triplet):
        dth, dx, dy = transformation_triplet
        old_point_coordinates = points_2d.copy().coordinates
        trasnformed_old_coordinates = old_point_coordinates @  np.array(
            [[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]]
        ) + [dx, dy]
        new_point_coordinates = points_2d.transform(dth, dx, dy).coordinates
        assert np.allclose(trasnformed_old_coordinates, new_point_coordinates)

    def test_reverse(self, points_2d):
        points_2d.reverse()
        assert np.array_equal(points_2d.x, [7.0, 5.0, 3.0, 1.0])
        assert np.array_equal(points_2d.y, [8.0, 6.0, 4.0, 2.0])
