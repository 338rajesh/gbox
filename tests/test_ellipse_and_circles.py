import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from gbox.ellipse import (
    EllipticalArc,
    Ellipse,
    CircularArc,
    Circle,
    CirclesArray,
    Point2D,
    PI,
)


# Fixtures
@pytest.fixture
def sample_elliptical_arc():
    return EllipticalArc(
        semi_major=4.0,
        semi_minor=2.0,
        centre=(1.0, 1.0),
        major_axis_angle=np.pi / 4,
        theta_start=0.0,
        theta_end=np.pi,
    )


@pytest.fixture
def sample_ellipse():
    return Ellipse(
        semi_major=4.0,
        semi_minor=2.0,
        centre=(1.0, 2.0),
        major_axis_angle=np.pi / 4,
    )


@pytest.fixture
def sample_circle():
    return Circle(radius=3.0, centre=(2.0, 2.0))


@pytest.fixture
def sample_circles_array():
    circles = [
        Circle(1.0, (0.0, 0.0)),
        Circle(2.0, (3.0, 4.0)),
    ]
    return CirclesArray(circles)


# Helper functions
def rotation_matrix(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


# Tests for EllipticalArc
class TestEllipticalArc:
    def test_init_validation(self):
        # Semi-minor <= 0
        with pytest.raises(ValueError):
            EllipticalArc(5.0, -1.0, (0, 0))
        # Semi-major < semi-minor
        with pytest.raises(ValueError):
            EllipticalArc(1.0, 2.0, (0, 0))

    def test_area(self, sample_elliptical_arc):
        expected = 0.5 * 4.0 * 2.0 * (np.pi - 0.0)
        assert_allclose(sample_elliptical_arc.area, expected, rtol=1e-6)

    def test_perimeter_open(self, sample_elliptical_arc):
        # Compare with quad result directly
        arc_length, _ = quad(
            sample_elliptical_arc._arc_len_integrand,
            sample_elliptical_arc.theta_start,
            sample_elliptical_arc.theta_end,
        )
        assert_allclose(
            sample_elliptical_arc.perimeter("open"), arc_length, rtol=1e-6
        )

    def test_perimeter_e2e(self, sample_elliptical_arc):
        arc_length = sample_elliptical_arc.perimeter("open")
        chord = sample_elliptical_arc._end_point_1.distance_to(
            sample_elliptical_arc._end_point_2
        )
        assert_allclose(
            sample_elliptical_arc.perimeter("e2e"),
            arc_length + chord,
            rtol=1e-6,
        )

    def test_point_at_angle(self, sample_elliptical_arc):
        # Test theta=0 (aligned with major axis)
        point = sample_elliptical_arc.point_at_angle(0.0)
        # Major axis rotated by 45 degrees, so x and y should be equal
        assert_allclose(point.x - 1.0, point.y - 1.0, atol=1e-6)

    def test_sample_points(self, sample_elliptical_arc: EllipticalArc):
        points = sample_elliptical_arc.sample_points(num_points=10)
        assert len(points) == 10
        # First point should match theta_start
        assert_allclose(
            points[0], sample_elliptical_arc._end_point_1.coor, atol=1e-6
        )


# Tests for Ellipse
class TestEllipse:
    def test_area(self, sample_ellipse: Ellipse):
        expected = PI * 4.0 * 2.0
        assert_allclose(sample_ellipse.area, expected, rtol=1e-6)

    def test_perimeter_approximation(self, sample_ellipse: Ellipse):
        # Ramanujan's approximation for perimeter
        a, b = 4.0, 2.0
        h = ((a - b) / (a + b)) ** 2
        approx_perimeter = (
            PI * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        )
        assert_allclose(sample_ellipse.perimeter, approx_perimeter, rtol=1e-2)

    def test_bounding_box(self, sample_ellipse: Ellipse):
        bbox = sample_ellipse.get_bounding_box()
        xlb, ylb = bbox.p_min.coor
        xub, yub = bbox.p_max.coor
        xc, yc = sample_ellipse.centre.coor 
        a2 = sample_ellipse.semi_major ** 2
        b2 = sample_ellipse.semi_minor ** 2
        s2 = np.sin(sample_ellipse.major_axis_angle) ** 2
        c2 = np.cos(sample_ellipse.major_axis_angle) ** 2
        xlb_true = xc - np.sqrt(a2 * c2 + b2 * s2)
        xub_true = xc + np.sqrt(a2 * c2 + b2 * s2)
        ylb_true = yc - np.sqrt(a2 * s2 + b2 * c2)
        yub_true = yc + np.sqrt(a2 * s2 + b2 * c2)
        # After rotation by 45 degrees, the bounding box should expand
        assert xlb == pytest.approx(xlb_true, rel=1e-6)
        assert ylb == pytest.approx(ylb_true, rel=1e-6)
        assert xub == pytest.approx(xub_true, rel=1e-6)
        assert yub == pytest.approx(yub_true, rel=1e-6) 

    def test_contains(self, sample_ellipse: Ellipse):
        # Point inside
        p_inside = Point2D(1.0, 1.0)
        assert sample_ellipse.contains(p_inside) == 1

        # Point outside
        p_outside = Point2D(10.0, 10.0)
        assert sample_ellipse.contains(p_outside) == -1

        # Point on ellipse (approx)
        p_on = sample_ellipse.arc.point_at_angle(0.0)
        assert sample_ellipse.contains(p_on, atol=1e-06) == 0, (
            "Expecting point to lie on the ellipse"
        )

    def test_union_of_circles(self, sample_ellipse: Ellipse):
        dh= 0.001  # Small distance for circle approximation
        circles = sample_ellipse.union_of_circles(dh=dh)
        assert isinstance(circles, CirclesArray)
        assert len(circles) > 0
        # After transformation, circles should cover the ellipse's bounding box
        bbox_ellipse = sample_ellipse.get_bounding_box()
        bbox_circles = circles.bounding_box()
        
        # The circles' bounding box should be slightly larger (by ~dh) than the ellipse's
        # Check each coordinate with appropriate tolerance
        np.testing.assert_allclose(
            bbox_circles.p_min.coor,
            bbox_ellipse.p_min.coor,
            rtol=1e-3,  # Relative tolerance of 0.1%
            atol=2*dh,  # Absolute tolerance of 2*dh
            err_msg="Lower bounds of bounding boxes don't match within tolerance"
        )
        np.testing.assert_allclose(
            bbox_circles.p_max.coor,
            bbox_ellipse.p_max.coor,
            rtol=1e-3,
            atol=2*dh,
            err_msg="Upper bounds of bounding boxes don't match within tolerance"
        )


# Tests for CircularArc
class TestCircularArc:
    def test_perimeter(self):
        arc = CircularArc(radius=3.0, theta_1=0.0, theta_2=np.pi)
        expected = 3.0 * np.pi  # Half circumference
        assert_allclose(arc.perimeter("open"), expected, rtol=1e-6)


# Tests for Circle
class TestCircle:
    def test_area(self, sample_circle):
        expected = PI * 3.0**2
        assert_allclose(sample_circle.area, expected, rtol=1e-6)

    def test_contains(self, sample_circle):
        # Inside
        assert sample_circle.contains((2.0, 2.0)) == 1
        # On edge
        assert sample_circle.contains((2.0 + 3.0, 2.0)) == 0
        # Outside
        assert sample_circle.contains((10.0, 10.0)) == -1

    def test_distance_to(self, sample_circle):
        other = Circle(2.0, (5.0, 6.0))
        expected = np.sqrt((5 - 2) ** 2 + (6 - 2) ** 2)
        assert_allclose(sample_circle.distance_to(other), expected, rtol=1e-6)


# Tests for CirclesArray
class TestCirclesArray:
    def test_add_circles(self, sample_circles_array: CirclesArray):
        assert len(sample_circles_array) == 2

    def test_transform_translation(self, sample_circles_array: CirclesArray):
        sample_circles_array.transform(dx=1.0, dy=2.0)
        assert_allclose(sample_circles_array.xc, [1.0, 4.0], rtol=1e-6)
        assert_allclose(sample_circles_array.yc, [2.0, 6.0], rtol=1e-6)

    def test_transform_rotation(self, sample_circles_array: CirclesArray):
        sample_circles_array.transform(angle=np.pi / 2, pivot=(0.0, 0.0))
        # First circle: (0,0) rotated 90 becomes (0,0). No change.
        # Second circle: (3,4) rotated 90 becomes (-4,3)
        assert_allclose(
            sample_circles_array.centres[1], [-4.0, 3.0], rtol=1e-6
        )
        sample_circles_array.transform(
            angle=-np.pi, pivot=(0.0, 0.0)
        )
        # After rotating back by 180 degrees, second circle should be at (4, -3)
        assert_allclose(
            sample_circles_array.centres[1], [4.0, -3.0], rtol=1e-6
        )

    def test_clip(self, sample_circles_array: CirclesArray):
        clipped = sample_circles_array.clip(
            x_lim=(0.0, 2.0), y_lim=(0.0, 3.0), inplace=False
        )
        assert clipped is not None, "Clip returned None"
        assert clipped[0, 0] == 0.0  # x clipped to 0
        assert clipped[1, 1] == 3.0  # y clipped to 3

    def test_bounding_box(self, sample_circles_array: CirclesArray):
        bbox = sample_circles_array.bounding_box()
        # circle_1: -1.0, -1.0, 1.0, 1.0
        # circle_2: 1.0, 2.0, 5.0, 6.0
        # bbox: -1.0, -1.0, 5.0, 6.0
        assert_allclose(bbox.p_min.coor[0], -1.0, rtol=1e-6)
        assert_allclose(bbox.p_min.coor[1], -1.0, rtol=1e-6)
        assert_allclose(bbox.p_max.coor[0], 5.0, rtol=1e-6)
        assert_allclose(bbox.p_max.coor[1], 6.0, rtol=1e-6)

    def test_contains(self, sample_circles_array: CirclesArray):
        # Point inside first circle
        assert sample_circles_array.contains((0.0, 0.0)) == 1
        # Point outside all circles
        assert sample_circles_array.contains((10.0, 10.0)) == -1


