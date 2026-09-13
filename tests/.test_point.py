import pytest
import numpy as np
from gbox.base import PointND, Point1D, Point2D, Point3D, DEFAULT_FLOAT

# =============================================
#               TEST FIXTURES
# =============================================


@pytest.fixture
def sample_point_nd():
    return PointND(1.0, 2.0, 3.0, 4.0)


@pytest.fixture
def sample_point_1d():
    return Point1D(5.0)


@pytest.fixture
def sample_point_2d():
    return Point2D(1.0, 2.0)


@pytest.fixture
def sample_point_3d():
    return Point3D(1.0, 2.0, 3.0)


# =============================================
#               POINTND TESTS
# =============================================


class TestPointND:
    def test_initialization(self):
        # Test valid initialization
        p = PointND(1.0, 2.0, 3.0)
        assert len(p) == 3
        assert p.dim == 3
        assert isinstance(p.coor, np.ndarray)
        assert p.coor.dtype == DEFAULT_FLOAT

        # Test from sequence
        p_from_seq = PointND._from_([4.0, 5.0, 6.0])
        assert isinstance(p_from_seq, PointND)
        assert np.array_equal(
            p_from_seq.coor, np.array([4.0, 5.0, 6.0], dtype=DEFAULT_FLOAT)
        )

        # Test from another PointND
        p_from_point = PointND._from_(p)
        assert p_from_point is p  # Should return same object

        # Test invalid initialization
        with pytest.raises(ValueError):
            PointND()  # Empty coordinates

        with pytest.raises(TypeError):
            PointND("a", "b")  # type: ignore

        with pytest.raises(TypeError):
            PointND._from_("invalid")  # type: ignore

    def test_magic_methods(self, sample_point_nd):
        p = sample_point_nd

        # Test __len__
        assert len(p) == 4

        # Test __getitem__
        assert p[0] == 1.0
        assert p[1] == 2.0
        assert p[2] == 3.0
        assert p[3] == 4.0
        with pytest.raises(IndexError):
            _ = p[4]  # Out of bounds

        # Test __iter__
        coords = list(p)
        assert coords == [1.0, 2.0, 3.0, 4.0]

        # Test __array__
        arr = np.array(p)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(
            arr, np.array([1.0, 2.0, 3.0, 4.0], dtype=DEFAULT_FLOAT)
        )

        # Test __eq__
        assert p == PointND(1.0, 2.0, 3.0, 4.0)
        assert p == [1.0, 2.0, 3.0, 4.0]
        assert not p == PointND(1.0, 2.0, 4.0, 3.0)
        assert not p == [1.0, 2.0]
        with pytest.raises(TypeError):
            _ = p == "invalid"

        # Test __repr__
        assert repr(p) == "PointND(1.0, 2.0, 3.0, 4.0)"

    def test_properties(self, sample_point_nd):
        p = sample_point_nd
        assert p.dim == 4

    def test_geometric_methods(self):
        # Test distance_to
        p1 = PointND(0.0, 0.0)
        p2 = PointND(3.0, 4.0)
        assert p1.distance_to(p2) == 5.0
        assert p1.distance_to([3.0, 4.0]) == 5.0

        # Test in_bounds
        lower = PointND(0.0, 0.0)
        upper = PointND(5.0, 5.0)
        assert PointND(2.0, 3.0).in_bounds(lower, upper)
        assert not PointND(-1.0, 3.0).in_bounds(lower, upper)
        assert not PointND(6.0, 3.0).in_bounds(lower, upper)

        # Test dimension mismatch
        with pytest.raises(ValueError):
            PointND(1.0, 2.0).in_bounds([0.0], [5.0])

        # Test is_close_to
        p = PointND(1.0, 2.0)
        assert p.is_close_to([1.0 + 1e-9, 2.0 + 1e-9])
        assert not p.is_close_to([1.1, 2.0])

        # Test reflection (not implemented)
        with pytest.raises(NotImplementedError):
            p.reflection(
                PointND(0.0, 0.0), PointND(1.0, 0.0), PointND(0.0, 1.0)
            )


# =============================================
#               POINT1D TESTS
# =============================================


class TestPoint1D:
    def test_initialization(self, sample_point_1d):
        p = sample_point_1d
        assert p.dim == 1
        assert p.x == 5.0
        assert repr(p) == "Point1D(x=5.0)"

        # Test invalid initialization
        with pytest.raises(TypeError):
            Point1D()  # type: ignore

        with pytest.raises(TypeError):
            Point1D(1.0, 2.0)  # type: ignore

    def test_inherited_methods(self, sample_point_1d):
        p = sample_point_1d
        assert len(p) == 1
        assert p[0] == 5.0
        assert p == Point1D(5.0)
        assert p == [5.0]


# =============================================
#               POINT2D TESTS
# =============================================


class TestPoint2D:
    def test_initialization(self, sample_point_2d):
        p = sample_point_2d
        assert p.dim == 2
        assert p.x == 1.0
        assert p.y == 2.0
        assert repr(p) == "Point2D(x=1.0, y=2.0)"

        # Test invalid initialization
        with pytest.raises(TypeError):
            Point2D("a", "b")  # type: ignore # Invalid types

        with pytest.raises(TypeError):
            Point2D(1.0)  # type: ignore  # Not enough coordinates

        with pytest.raises(TypeError):
            Point2D(1.0, 2.0, 3.0)  # type: ignore  # Too many coordinates

    def test_special_methods(self, sample_point_2d):
        p = sample_point_2d

        # Test slope
        q = Point2D(3.0, 4.0)
        assert p.slope(q) == 1.0  # (4-2)/(3-1) = 1.0

        # Test vertical line slope
        vert_q = Point2D(1.0, 4.0)
        assert p.slope(vert_q) == np.inf

        # Test angle
        angle_rad = p.angle(q)
        assert np.isclose(angle_rad, np.pi / 4)  # 45 degrees in radians

        angle_deg = p.angle(q, rad=False)
        assert np.isclose(angle_deg, 45.0)

        # Test transform
        transformed = p.transform(angle=np.pi / 2)  # 90 degree rotation
        assert np.isclose(transformed.x, -2.0, atol=1e-6)
        assert np.isclose(transformed.y, 1.0, atol=1e-6)

        translated = p.transform(dx=1.0, dy=2.0)
        assert translated.x == 2.0
        assert translated.y == 4.0


# =============================================
#               POINT3D TESTS
# =============================================


class TestPoint3D:
    def test_initialization(self, sample_point_3d):
        p = sample_point_3d
        assert p.dim == 3
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0
        assert repr(p) == "Point3D(x=1.0, y=2.0, z=3.0)"

        # Test invalid initialization
        with pytest.raises(TypeError):
            Point3D(1.0, 2.0)  # type: ignore # Not enough coordinates

        with pytest.raises(TypeError):
            Point3D(1.0, 2.0, 3.0, 4.0)  # type: ignore # Too many coordinates

    def test_transform(self, sample_point_3d):
        p = sample_point_3d

        # Create a translation matrix
        matrix = np.eye(4, dtype=DEFAULT_FLOAT)
        matrix[:3, 3] = [1.0, 2.0, 3.0]  # Translation

        transformed = p.transform(matrix)
        assert transformed.x == 2.0
        assert transformed.y == 4.0
        assert transformed.z == 6.0

        # Test invalid matrix
        with pytest.raises(ValueError):
            p.transform(np.eye(3))  # Wrong size
