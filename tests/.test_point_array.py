import pytest
import numpy as np
from gbox.base import (
    PointArrayND,
    PointArray1D,
    PointArray2D,
    PointArray3D,
    PointND,
    Point2D,
    BoundingBox,
    DEFAULT_FLOAT,
)

# =============================================
#               TEST FIXTURES
# =============================================


@pytest.fixture
def sample_points_nd():
    return np.array(
        [[1.0, 2.0, 3.0, 9.0], [4.0, 5.0, 6.0, 8.0]], dtype=DEFAULT_FLOAT
    )


@pytest.fixture
def sample_point_array_nd(sample_points_nd):
    return PointArrayND(sample_points_nd)


@pytest.fixture
def sample_point_array_1d():
    return PointArray1D(np.array([1.0, 2.0, 3.0], dtype=DEFAULT_FLOAT))


@pytest.fixture
def sample_point_array_2d():
    return PointArray2D(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_FLOAT)
    )


@pytest.fixture
def sample_point_array_3d():
    return PointArray3D(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DEFAULT_FLOAT)
    )


@pytest.fixture
def sample_points_list_2d():
    return [Point2D(1.0, 2.0), Point2D(3.0, 4.0)]


# =============================================
#               POINTARRAYND TESTS
# =============================================


class TestPointArrayND:
    def test_initialization(self, sample_points_nd):
        # Test valid initialization with numpy array
        pa = PointArrayND(sample_points_nd)
        assert len(pa) == 2
        assert pa.dim == 4
        assert pa.dtype == DEFAULT_FLOAT
        assert isinstance(pa.coordinates, np.ndarray)
        assert np.array_equal(pa.coordinates, sample_points_nd)

        # Test from_points with Point objects
        points = [PointND(1.0, 2.0, 3.0, 9.0), PointND(4.0, 5.0, 6.0, 8.0)]
        pa_from_points = PointArrayND.from_points(points)
        assert len(pa_from_points) == 2
        assert np.array_equal(pa_from_points.coordinates, sample_points_nd)

        # Test from_points with sequences
        seq_points = [[1.0, 2.0, 3.0, 9.0], [4.0, 5.0, 6.0, 8.0]]
        pa_from_seq = PointArrayND.from_points(seq_points)
        assert np.array_equal(pa_from_seq.coordinates, sample_points_nd)

        # Test invalid initialization
        with pytest.raises(TypeError):
            PointArrayND("invalid")  # type: ignore # Not a numpy array

        with pytest.raises(ValueError):
            PointArrayND(np.array([], dtype=DEFAULT_FLOAT))  # Empty array

        with pytest.raises(ValueError):
            PointArrayND(
                np.array([1.0, 2.0, 3.0], dtype=DEFAULT_FLOAT)
            )  # 1D array

        with pytest.raises(ValueError):
            PointArrayND.from_points([])  # Empty points list

        with pytest.raises(ValueError):
            PointArrayND.from_points(
                [[1.0], [2.0, 3.0]]
            )  # Inconsistent dimensions

    def test_magic_methods(self, sample_point_array_nd):
        pa = sample_point_array_nd

        # Test __len__
        assert len(pa) == 2

        # Test __getitem__
        assert np.array_equal(
            pa[0], np.array([1.0, 2.0, 3.0, 9.0], dtype=DEFAULT_FLOAT)
        )
        assert np.array_equal(
            pa[1], np.array([4.0, 5.0, 6.0, 8.0], dtype=DEFAULT_FLOAT)
        )
        assert np.array_equal(
            pa[0:1], np.array([[1.0, 2.0, 3.0, 9.0]], dtype=DEFAULT_FLOAT)
        )
        assert pa[0, 0] == 1.0  # Test tuple indexing

        # Test __iter__
        points = list(pa)
        assert len(points) == 2
        assert np.array_equal(points[0], pa[0])

        # Test __array__
        arr = np.array(pa)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, pa.coordinates)

        # Test __repr__
        assert "PointArrayND(2 points; dim=4; dtype=float32)" in repr(pa)

    def test_properties(self, sample_point_array_nd):
        pa = sample_point_array_nd

        # Test dim property
        assert pa.dim == 4

        # Test dtype property
        assert pa.dtype == DEFAULT_FLOAT

        # Test coordinates property
        assert isinstance(pa.coordinates, np.ndarray)
        assert pa.coordinates.flags["C_CONTIGUOUS"]  # Check contiguous

        # Test cycle property
        assert pa.cycle is False
        pa.cycle = True
        assert pa.cycle is True
        with pytest.raises(TypeError):
            pa.cycle = "not a boolean"

    def test_geometry_methods(self, sample_point_array_3d):
        pa = sample_point_array_3d

        # Test bounding_box
        bb = pa.bounding_box()
        assert isinstance(bb, BoundingBox)
        assert np.array_equal(
            bb.p_min.coor, np.array([1.0, 2.0, 3.0], dtype=DEFAULT_FLOAT)
        )
        assert np.array_equal(
            bb.p_max.coor, np.array([4.0, 5.0, 6.0], dtype=DEFAULT_FLOAT)
        )

        # Test transform (homogeneous coordinates)
        matrix = np.eye(4, dtype=DEFAULT_FLOAT)
        matrix[:3, 3] = [1.0, 2.0, 3.0]  # Translation

        transformed = pa.transform(matrix)
        expected = np.array(
            [[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]], dtype=DEFAULT_FLOAT
        )
        assert np.allclose(transformed.coordinates, expected)

        # Test in-place transform
        pa.transform(matrix, in_place=True)
        assert np.allclose(pa.coordinates, expected)

        # Test invalid transform matrix
        with pytest.raises(ValueError):
            pa.transform(np.eye(3))  # Wrong size

        # Test reverse
        reversed_pa = pa.reverse()
        assert np.array_equal(
            reversed_pa.coordinates, np.flip(pa.coordinates, axis=0)
        )

        # Test in-place reverse
        original = pa.coordinates.copy()
        pa.reverse(in_place=True)
        assert np.array_equal(pa.coordinates, np.flip(original, axis=0))

        # Test reflection (not implemented)
        with pytest.raises(NotImplementedError):
            pa.reflection([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    def test_utility_methods(self, sample_point_array_nd):
        pa = sample_point_array_nd

        # Test copy
        copy_pa = pa.copy()
        assert copy_pa is not pa
        assert np.array_equal(copy_pa.coordinates, pa.coordinates)

        # Test to_points_list
        points_list = pa.to_points_list()
        assert len(points_list) == 2
        assert isinstance(points_list[0], PointND)
        assert points_list[0] == PointND(1.0, 2.0, 3.0, 9.0)


# =============================================
#               POINTARRAY1D TESTS
# =============================================


class TestPointArray1D:
    def test_initialization(self, sample_point_array_1d):
        pa = sample_point_array_1d
        assert pa.dim == 1
        assert np.array_equal(
            pa.x, np.array([1.0, 2.0, 3.0], dtype=DEFAULT_FLOAT)
        )

        # Test invalid initialization
        with pytest.raises(ValueError):
            PointArray1D(
                np.array([[1.0, 2.0]], dtype=DEFAULT_FLOAT)
            )  # Wrong dim

    def test_transform(self, sample_point_array_1d):
        pa = sample_point_array_1d

        # Test transform with translation
        transformed = pa.transform(dx=2.0)
        expected = np.array([[3.0], [4.0], [5.0]], dtype=DEFAULT_FLOAT)
        assert np.array_equal(transformed.coordinates, expected)

        # Test in-place transform
        original = pa.coordinates.copy()
        pa.transform(dx=1.0, in_place=True)
        assert np.array_equal(pa.coordinates, original + 1.0)


# =============================================
#               POINTARRAY2D TESTS
# =============================================


class TestPointArray2D:
    def test_initialization(
        self, sample_point_array_2d, sample_points_list_2d
    ):
        pa = sample_point_array_2d
        assert pa.dim == 2
        assert np.array_equal(pa.x, np.array([1.0, 3.0], dtype=DEFAULT_FLOAT))
        assert np.array_equal(pa.y, np.array([2.0, 4.0], dtype=DEFAULT_FLOAT))

        # Test from_points with Point2D objects
        pa_from_points = PointArray2D.from_points(sample_points_list_2d)
        assert np.array_equal(
            pa_from_points.coordinates,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DEFAULT_FLOAT),
        )

        # Test invalid initialization
        with pytest.raises(ValueError):
            PointArray2D(
                np.array([1.0, 2.0, 3.0], dtype=DEFAULT_FLOAT)
            )  # Wrong shape

    def test_transform(self, sample_point_array_2d):
        pa = sample_point_array_2d

        # Test transform with rotation and translation
        transformed = pa.transform(
            angle=np.pi / 2, dx=1.0, dy=2.0
        )  # 90 degree rotation
        expected = np.array(
            [[-2.0 + 1.0, 1.0 + 2.0], [-4.0 + 1.0, 3.0 + 2.0]],
            dtype=DEFAULT_FLOAT,
        )
        assert np.allclose(transformed.coordinates, expected)

        # Test in-place transform
        original = pa.coordinates.copy()
        pa.transform(dx=1.0, dy=1.0, in_place=True)
        assert np.array_equal(pa.coordinates, original + np.array([1.0, 1.0]))

    def test_periodic_tiles(self, sample_point_array_2d):
        with pytest.raises(NotImplementedError):
            sample_point_array_2d.make_periodic_tiles()


# =============================================
#               POINTARRAY3D TESTS
# =============================================


class TestPointArray3D:
    def test_initialization(self, sample_point_array_3d):
        pa = sample_point_array_3d
        assert pa.dim == 3
        assert np.array_equal(pa.x, np.array([1.0, 4.0], dtype=DEFAULT_FLOAT))
        assert np.array_equal(pa.y, np.array([2.0, 5.0], dtype=DEFAULT_FLOAT))
        assert np.array_equal(pa.z, np.array([3.0, 6.0], dtype=DEFAULT_FLOAT))

        # Test invalid initialization
        with pytest.raises(ValueError):
            PointArray3D(
                np.array([1.0, 2.0, 3.0], dtype=DEFAULT_FLOAT)
            )  # Wrong shape

    def test_periodic_tiles(self, sample_point_array_3d):
        with pytest.raises(NotImplementedError):
            sample_point_array_3d.make_periodic_tiles()
