import pytest
from gbox import (
    TypeConfig,
    PointND,
    Point2D,
    PointArrayND,
    BoundingBox,
)
from gbox.core import get_type, cast_to
from gbox.base import PointArray1D, PointArray2D, PointArray3D
import numpy as np
from utils import gb_plotter, get_output_dir
import pathlib

from hypothesis import given
from hypothesis import strategies as st

PI = cast_to(np.pi, "float")
OUTPUT_DIR = get_output_dir(
    pathlib.Path(__file__).parent / "__output" / "test_base_point"
)


def _test_floats_approx_equality_(a, b, message=""):
    try:
        assert a == pytest.approx(b, abs=TypeConfig.float_precision()), message
        return True
    except AssertionError as e:
        raise e


@pytest.fixture(scope="module")
def point_2d():
    return PointND(1.0, 2.0)


@pytest.fixture
def origin():
    return PointND(0.0, 0.0)


@pytest.fixture
def point_3d():
    return PointND(1.0, 2.0, -3.0)


@pytest.fixture
def point_array_4x2():
    return PointArrayND.from_sequences([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0])


@pytest.fixture
def point_array_4x1():
    return PointArray1D.from_sequences([2], [4], [6], [8])


@pytest.fixture
def point_array_5x2():
    return PointArray2D.from_sequences(
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]
    )


@pytest.fixture
def point_array_5x3():
    return PointArray3D.from_sequences(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    )


class TestPointND:
    def test_point(self):
        """Tests Point class constructor"""
        p = PointND(1.0, 2.0)
        assert p.dim == 2, "Point dimension should be 2"

        p1 = PointND._make_with_(p)
        assert p == p1, "p and p1 should be equal"

        p2 = PointND._make_with_([1.0, 2.0])
        assert p == p2

        p3 = PointND._make_with_((1.0, 2.0))
        assert p == p3

        p4 = PointND._make_with_(np.array([1.0, 2.0]))
        assert p == p4

        with pytest.raises(ValueError):
            PointND._make_with_(np.array([[1.0, 2.0, 3.0]]))

    def test_point_repr(self, point_2d):
        """Tests Point.__repr__ method"""
        dt = type(point_2d[0])
        assert f"{dt(1.0)}" in str(point_2d)
        assert f"{dt(2.0)}" in str(point_2d)

    def test_point_equality(self, point_2d):
        """Tests Point.__eq__ method"""
        assert point_2d == PointND(1.0, 2.0)

    def test_point_dimension(self, point_2d):
        """Tests Point.dim attribute"""
        assert point_2d.dim == 2
        assert PointND(1.0, 2.0, 3.0).dim == 3

    def test_points_compatibility(self, point_2d, origin, point_3d):
        """Tests Point._assert_points_compatibility_ method"""
        PointND._assert_points_compatibility_(point_2d, origin)
        with pytest.raises(ValueError):
            PointND._assert_points_compatibility_(point_2d, point_3d)
        with pytest.raises(TypeError):
            PointND._assert_points_compatibility_(point_2d, 1.0)

    def test_distance_to(self, point_2d, origin):
        _test_floats_approx_equality_(
            point_2d.distance_to(origin), cast_to(np.sqrt(5.0), "float")
        )
        _test_floats_approx_equality_(point_2d.distance_to(point_2d), 0.0)
        _test_floats_approx_equality_(point_2d.distance_to((1.0, 2.0)), 0.0)
        _test_floats_approx_equality_(
            point_2d.distance_to((0.0, 0.0)), cast_to(np.sqrt(5.0), "float")
        )

    def test_distance_data_type(self, point_2d):
        TypeConfig.set_float_type(np.float32)
        d1 = point_2d.distance_to((0.0, 0.0))
        assert type(d1) is np.float32

        TypeConfig.set_float_type(np.float64)
        point_2d = PointND(1.0, 2.0)
        d2 = point_2d.distance_to((0.0, 0.0))
        assert type(d2) is np.float64, "d2 is expected to be np.float64"

        TypeConfig.set_float_type(np.float16)
        d3 = point_2d.distance_to((0.0, 0.0))
        assert type(d3) is np.float16

    def test_point_in_bounds(self):
        p, q, r, _ = (
            PointND(1.0, 2.0),
            PointND(13.0, 14.0),
            PointND(0.0, 0.0),
            PointND(10.0, 10.0),
        )
        lb, ub = (0.0, 0.0), (10.0, 10.0)

        # testing if point is within bounds with valid bounds
        assert p.in_bounds(BoundingBox(lb, ub))
        assert p.in_bounds((lb, ub))
        assert not q.in_bounds(BoundingBox(lb, ub))
        assert r.in_bounds(BoundingBox(lb, ub), include_bounds=True)
        assert not r.in_bounds(BoundingBox(lb, ub))

        # testing if bounds type mismatch causes error
        with pytest.raises(TypeError):
            p.in_bounds(1.0)

        # testing if dimension mismatch causes error
        with pytest.raises(ValueError):
            p.in_bounds((1.0, 2.0, 3.0))

    def test_as_list(self, point_2d):
        lst = point_2d.as_list()
        assert isinstance(lst, list)
        assert lst == [1.0, 2.0]

    def test_as_array(self, point_2d):
        arr = point_2d.as_array()
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1.0, 2.0]))
        assert arr.dtype == get_type("float")
        assert arr.shape == (2,)

    def test_reflection(self, point_2d):
        with pytest.raises(NotImplementedError):
            point_2d.reflection((1.0, 2.0), (1.0, 2.0), (3.0, 4.0))

    def test_point_is_close(self, point_2d, origin):
        assert point_2d.is_close_to(point_2d)
        assert point_2d.is_close_to((1.0 + 1e-08, 2.0 + 1e-07), eps=1e-6)
        assert not point_2d.is_close_to(origin)


class TestPoint2D:
    def test_point_2d_constructor(self):
        point2d = Point2D(1.0, 2.0)
        assert point2d.dim == 2
        assert point2d.x == 1.0
        assert point2d.y == 2.0
        p1_2d = Point2D._make_with_(point2d)
        assert isinstance(p1_2d, Point2D)
        p2_2d = Point2D._make_with_([1.0, 2.0])
        assert isinstance(p2_2d, Point2D)
        assert p2_2d.x == 1.0
        assert p2_2d.y == 2.0

    def test_point_inheritance(self):
        p_2d = Point2D(1.0, 2.0)
        assert isinstance(p_2d, PointND)
        assert isinstance(p_2d, Point2D)
        assert p_2d.distance_to((1.0, 2.0)) == 0.0

    def test_slope(self):
        point_2d = Point2D(1.0, 2.0)
        assert point_2d.slope((3.0, 4.0)) == pytest.approx(
            1.0, abs=TypeConfig.float_precision()
        )
        assert type(point_2d.slope((3.0, 4.0))) == TypeConfig.float_type().dtype

    def test_angle(self):
        point_2d = Point2D(1.0, 2.0)
        eps = TypeConfig.float_precision()
        assert type(point_2d.angle((2.0, 3.0))) == TypeConfig.float_type().dtype

        assert point_2d.angle((2.0, 3.0)) == pytest.approx(np.pi * 0.25, eps)
        assert point_2d.angle((0.0, 3.0)) == pytest.approx(np.pi * 0.75, eps)
        assert point_2d.angle((0.0, 1.0)) == pytest.approx(np.pi * 1.25, eps)
        assert point_2d.angle((2.0, 1.0)) == pytest.approx(np.pi * 1.75, eps)

    def test_transform(self):
        eps = TypeConfig.float_precision()
        assert (
            Point2D(0.0, 1.0)
            .transform(angle=np.pi * 1.5, dx=5.0, dy=6.0)
            .is_close_to(Point2D(6.0, 6.0), eps)
        )
        assert (
            Point2D(-3.0, -3.0)
            .transform(angle=np.pi * 1.0, dx=-3.0, dy=-3.0)
            .is_close_to(Point2D(0.0, 0.0), eps)
        )
        assert (
            Point2D(3.0, -3.0)
            .transform(angle=np.pi * 0.25, dx=-3.0 * np.sqrt(2.0), dy=0.0)
            .is_close_to(Point2D(0.0, 0.0), eps)
        )


class TestPointArray:
    """Tests for PointArray class"""

    def test_point_array_constructor(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p_arr = PointArrayND(p)
        assert isinstance(p_arr, PointArrayND)
        assert p_arr.dim == 2
        assert p_arr.dtype == TypeConfig.float_type().dtype
        assert p_arr.coordinates.shape == (3, 2)
        p_arr = PointArrayND(p, dtype=np.int32)
        assert p_arr.dtype == np.int32

        with pytest.raises(TypeError):
            PointArrayND([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        with pytest.raises(NotImplementedError):
            PointArrayND(np.random.rand(4, 3, 2))

    def test_point_array_from_sequence(self):
        p = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        r = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0))

        with pytest.raises(ValueError):
            PointArrayND.from_sequences()

        with pytest.raises(ValueError):
            PointArrayND.from_sequences(
                [1.0, 2.0],
                [3.0, 4.0],
                [
                    5.0,
                ],
            )

        for a in (p, r):
            point_arr = PointArrayND.from_sequences(*a)
            assert isinstance(point_arr, PointArrayND)
            assert point_arr.dim == 2
            assert point_arr.coordinates.shape == (4, 2)
            assert point_arr.dtype == TypeConfig.float_type().dtype

    def test_point_array_from_dimensions_data(self):
        px = [1.0, 2.0, 3.0, 4.0, 5.0]
        py = [2.0, 3.0, 4.0, 5.0, 6.0]
        TypeConfig.set_float_type(np.float16)
        point_arr = PointArrayND.from_dimensions_data(px, py)
        assert isinstance(point_arr, PointArrayND)
        assert point_arr.dim == 2
        assert point_arr.coordinates.shape == (5, 2)
        assert len(point_arr) == 5
        assert point_arr.dtype == np.float16
        TypeConfig.set_float_type(np.float32)
        point_arr = PointArrayND.from_dimensions_data(px, py)
        assert point_arr.dtype == np.float32

    def test_point_array_from_points(self):
        p = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        TypeConfig.set_float_type(float)
        points = [PointND(*x) for x in p]
        point_arr = PointArrayND.from_points(*points)
        assert isinstance(point_arr, PointArrayND)
        assert point_arr.dim == 3
        assert point_arr.coordinates.shape == (4, 3)
        assert point_arr.dtype == float
        assert len(point_arr) == 4

    def test_point_array_repr(self, point_array_4x2):
        out = "PointArray:\n[[1. 2.]\n [3. 4.]\n [5. 6.]\n [7. 8.]]"
        assert str(point_array_4x2) == out

    def test_point_array_eq(self, point_array_4x2):
        q = PointArrayND.from_dimensions_data(
            [1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]
        )
        assert point_array_4x2 == q

    def test_point_array_copy(self, point_array_4x2):
        assert point_array_4x2.copy() == point_array_4x2
        assert point_array_4x2.copy() is not point_array_4x2

    def test_point_array_bounding_box(self, point_array_4x2):
        assert point_array_4x2.bounding_box == BoundingBox([1.0, 2.0], [7.0, 8.0])

    def test_point_array_reflection(self, point_array_4x2, origin, point_2d):
        with pytest.raises(NotImplementedError):
            point_array_4x2.reflection(origin, point_2d)

    def test_points_cycle(self, point_array_4x2):
        assert point_array_4x2.cycle is False
        point_array_4x2.cycle = True
        assert point_array_4x2.cycle is True


class TestPointArray1D:
    def test_constructor(self, point_array_4x1):
        assert isinstance(point_array_4x1, PointArrayND)
        assert isinstance(point_array_4x1, PointArray1D)
        assert point_array_4x1.dim == 1
        assert np.array_equal(point_array_4x1.x, [2.0, 4.0, 6.0, 8.0])

    def test_transform(self, point_array_4x1):
        assert np.array_equal(
            point_array_4x1.transform(dx=0.25).coordinates,
            np.array([[2.25], [4.25], [6.25], [8.25]]),
        )

    def test_reverse(self, point_array_4x1):
        point_array_4x1.reverse()
        assert np.array_equal(point_array_4x1.x, [8.0, 6.0, 4.0, 2.0])

    def test_1d_plots(self, point_array_4x1, test_plots):
        if not test_plots:
            pytest.skip()

        with gb_plotter(OUTPUT_DIR / "points_1d_simple.png") as (fig, axs):
            point_array_4x1.plot(axs)
            axs.grid()
            axs.set_title("PointSet")


class TestPointArray2D:
    def test_constructor(self, point_array_5x2):
        assert point_array_5x2.dim == 2
        assert np.array_equal(point_array_5x2.x, [1.0, 3.0, 5.0, 7.0, 9.0])
        assert np.array_equal(point_array_5x2.y, [2.0, 4.0, 6.0, 8.0, 10.0])

    @given(
        dth=st.floats(min_value=0.0, max_value=np.pi * 2.0),
        dx=st.floats(min_value=-1.0, max_value=1.0),
        dy=st.floats(min_value=-1.0, max_value=1.0),
    )
    def test_transform(self, dth, dx, dy):
        dth = float(np.random.choice([0.0, np.pi * 2.0]))
        init_xy = np.random.rand(10, 2)
        trasnformed_x = init_xy[:, 0] * np.cos(dth) - init_xy[:, 1] * np.sin(dth) + dx
        trasnformed_y = init_xy[:, 0] * np.sin(dth) + init_xy[:, 1] * np.cos(dth) + dy
        trasnformed_xy = np.column_stack((trasnformed_x, trasnformed_y))
        #
        points_ = PointArray2D(init_xy)
        new_point_coordinates = points_.transform(dth, dx, dy).coordinates
        assert np.allclose(trasnformed_xy, new_point_coordinates)

    def test_reverse(self, point_array_5x2):
        point_array_5x2.reverse()
        assert np.array_equal(point_array_5x2.x, [9.0, 7.0, 5.0, 3.0, 1.0])
        assert np.array_equal(point_array_5x2.y, [10.0, 8.0, 6.0, 4.0, 2.0])

    def test_plots(self, test_plots):
        point_array = PointArray2D(np.random.rand(100, 2))
        if not test_plots:
            pytest.skip()

        with gb_plotter(OUTPUT_DIR / "points_simple.png") as (fig, axs):
            point_array.plot(axs)
            axs.grid()
            axs.set_title("PointSet")

        with gb_plotter(OUTPUT_DIR / "points_bb.png") as (fig, axs):
            point_array.plot(axs, b_box=True)
            axs.grid()
            axs.set_title("PointSet")

        with gb_plotter(OUTPUT_DIR / "points_bb_black_dashed.png") as (
            fig,
            axs,
        ):
            point_array.plot(
                axs,
                b_box=True,
                b_box_plt_opt={"color": "k", "linewidth": 2, "linestyle": "dashed"},
            )
            axs.grid()
            axs.set_title("PointSet")

        with gb_plotter(OUTPUT_DIR / "points_blue_color_cross.png") as (
            fig,
            axs,
        ):
            point_array.plot(
                axs,
                points_plt_opt={
                    "color": "blue",
                    "marker": "x",
                    "linestyle": "None",
                    "markersize": 10,
                },
            )
            axs.grid()
            axs.set_title("PointSet")

        with gb_plotter(OUTPUT_DIR / "points_as_line.png") as (fig, axs):
            point_array.plot(
                axs,
                points_plt_opt={
                    "color": "blue",
                    "marker": "x",
                    "linestyle": "solid",
                    "markersize": 10,
                },
            )
            axs.grid()
            axs.set_title("PointSet")


class TestPointArray3D:
    def test_constructor(self, point_array_5x3):
        assert point_array_5x3.dim == 3
        assert np.array_equal(point_array_5x3.x, [1.0, 4.0, 7.0, 10.0, 13.0])
        assert np.array_equal(point_array_5x3.y, [2.0, 5.0, 8.0, 11.0, 14.0])
        assert np.array_equal(point_array_5x3.z, [3.0, 6.0, 9.0, 12.0, 15.0])
        assert point_array_5x3.dtype == TypeConfig.float_type().dtype
