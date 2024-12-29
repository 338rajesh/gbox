import pathlib

import numpy as np
import gbox as gb
import pytest
from hypothesis import given, strategies as st

from gbox import utilities

OUTPUT_DIR = utilities.get_output_dir(
    pathlib.Path(__file__).parent / "__output" / "test_points"
)


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
def points_1d():
    return gb.Points1D(
        [
            [
                1.0,
            ],
            [
                3.0,
            ],
            [
                5.0,
            ],
            [
                7.0,
            ],
        ]
    )


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
        p.dim == 2
        p1 = gb.Point.from_seq(p)
        p2 = gb.Point.from_seq([1.0, 2.0])
        p3 = gb.Point.from_seq((1.0, 2.0))
        p4 = gb.Point.from_seq(np.array([1.0, 2.0]))
        with pytest.warns(UserWarning):
            p5 = gb.Point.from_seq(
                np.array(
                    [
                        [
                            1.0,
                        ],
                        [
                            2.0,
                        ],
                    ]
                )
            )
        assert p == p1
        assert p == p2
        assert p == p3
        assert p == p4
        assert p == p5

    def test_point_1(self):
        with pytest.raises(AssertionError):
            gb.Point(1.0, 2.0, "3")

    def test_point_eq(self, point):
        assert point == gb.Point(1.0, 2.0)

    def test_point_repr(self, point):
        assert str(point) == "Point:  (1.0, 2.0)"

    def test_point_add(self, point):
        assert point + point == gb.Point(2.0, 4.0)
        assert point + 25.0 == gb.Point(26.0, 27.0)

    def test_point_sub(self, point):
        assert point - point == gb.Point(0.0, 0.0)
        assert point - 35.0 == gb.Point(-34.0, -33.0)

    def test_point_mul(self, point):
        assert point * 7.0 == gb.Point(7.0, 14.0)
        assert point * point * point == gb.Point(1.0, 8.0)

    def test_adding_different_dimensions(self, point, point_3d):
        with pytest.raises(AssertionError):
            point + point_3d

    def test_distance_to(self, point, origin):
        assert point.distance_to(origin) == pytest.approx(2.23606, rel=1e-5)

    def test_point_in_bounds(self, point, origin):
        assert point.in_bounds(([-2.0, -2.0], [3.0, 3.0]))
        assert point.in_bounds(gb.BoundingBox(origin, origin + point + point))
        assert not origin.in_bounds(gb.BoundingBox(point, point + point))

    def test_point_on_bounds(self, point, origin):
        assert point.in_bounds(
            gb.BoundingBox(origin, origin + point), include_bounds=True
        )
        assert not origin.in_bounds(
            gb.BoundingBox(origin, origin + point), include_bounds=False
        )

    def test_point_conversions(self, point):
        assert np.array_equal(point.as_array(), np.array([1.0, 2.0]))
        assert point.as_list() == [1.0, 2.0]
    
    def test_point_is_close(self):
        assert gb.Point(1.0, 2.0).is_close_to(gb.Point(1.0, 2.0))
        assert not gb.Point(1.0, 2.0).is_close_to(gb.Point(1.0, 3.0))
        assert gb.Point(1.0, 2.0).is_close_to(gb.Point(1.0, 2.0), eps=1e-5)


class TestPoint2D:
    def test_point_2d(self, point_2d):
        assert point_2d.dim == 2
        assert point_2d.x == 1.0
        assert point_2d.y == 2.0
        assert point_2d == gb.Point2D(1.0, 2.0)

    def test_slope(self, point_2d):
        assert point_2d.slope(gb.Point2D(3.0, 4.0)) == pytest.approx(1.0, rel=1e-5)
        assert point_2d.slope(gb.Point2D(-1.0, -6.0)) == pytest.approx(4.0, rel=1e-5)

    def test_angle(self, point_2d):
        assert point_2d.angle(gb.Point2D(2.0, 3.0)) == pytest.approx(
            np.pi * 0.25, rel=1e-5
        )
        assert point_2d.angle(gb.Point2D(0.0, 3.0)) == pytest.approx(
            np.pi * 0.75, rel=1e-5
        )
        assert point_2d.angle(gb.Point2D(0.0, 1.0)) == pytest.approx(
            np.pi * 1.25, rel=1e-5
        )
        assert point_2d.angle(gb.Point2D(2.0, 1.0)) == pytest.approx(
            np.pi * 1.75, rel=1e-5
        )

    def test_transform(self):
        assert gb.Point2D(0.0, 1.0).transform(
            angle=np.pi * 1.5, dx=5.0, dy=6.0
        ).is_close_to(gb.Point2D(6.0, 6.0), eps=1e-5)
        assert gb.Point2D(-3.0, -3.0).transform(
            angle=np.pi * 1.0, dx=-3.0, dy=-3.0
        ).is_close_to(gb.Point2D(0.0, 0.0), eps=1e-5)
        assert gb.Point2D(3.0, -3.0).transform(
            angle=np.pi * 0.25, dx=-3.0*np.sqrt(2.0), dy=0.0
        ).is_close_to(gb.Point2D(0.0, 0.0), eps=1e-5)


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

    def test_points_from_dimension_data(self, points):
        new_points = gb.Points.from_dimension_data(
            np.array([1.0, 3.0, 5.0, 7.0]), np.array([2.0, 4.0, 6.0, 8.0])
        )
        assert new_points == points
        assert new_points.dim == 2
        assert len(new_points) == 4

    def test_points_cycle(self, points):
        points.cycle = True


class TestPoints1D:
    def test_constructor(self):
        p_1d = gb.Points1D([[1.0], [2.0], [3.0]])
        assert p_1d.dim == 1
        assert np.array_equal(p_1d.x, [1.0, 2.0, 3.0])

    def test_transform(self, points_1d):
        assert np.array_equal(
            points_1d.transform(dx=0.25).coordinates, [[1.25], [3.25], [5.25], [7.25]]
        )

    def test_reverse(self, points_1d):
        points_1d.reverse()
        assert np.array_equal(points_1d.x, [7.0, 5.0, 3.0, 1.0])

    def test_1d_plots(self, points_1d, test_plots):
        if not test_plots:
            pytest.skip()

        with utilities.gb_plotter(OUTPUT_DIR / "points_1d_simple.png") as (fig, axs):
            points_1d.plot(axs)
            axs.grid()
            axs.set_title("Points")


class TestPoints2D:
    def test_constructor(self, points_2d):
        assert points_2d.dim == 2
        assert np.array_equal(points_2d.x, [1.0, 3.0, 5.0, 7.0])
        assert np.array_equal(points_2d.y, [2.0, 4.0, 6.0, 8.0])

    # TODO FIXME 
    # def test_transform(self, points_2d, transformation_triplet):
    #     dth, dx, dy = transformation_triplet
    #     old_point_coordinates = points_2d.copy().coordinates
    #     trasnformed_old_coordinates = old_point_coordinates @ np.array(
    #         [[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]]
    #     ) + [dx, dy]
    #     new_point_coordinates = points_2d.transform(dth, dx, dy).coordinates
    #     assert np.allclose(trasnformed_old_coordinates, new_point_coordinates)

    def test_points_2d_angle(self, point_2d):
        assert point_2d.angle(gb.Point2D(2.0, 3.0)) == pytest.approx(
            np.pi * 0.25, rel=1e-5
        )

    def test_reverse(self, points_2d):
        points_2d.reverse()
        assert np.array_equal(points_2d.x, [7.0, 5.0, 3.0, 1.0])
        assert np.array_equal(points_2d.y, [8.0, 6.0, 4.0, 2.0])

    def test_plots(self, points_2d, test_plots):
        if not test_plots:
            pytest.skip()

        with utilities.gb_plotter(OUTPUT_DIR / "points_simple.png") as (fig, axs):
            points_2d.plot(axs)
            axs.grid()
            axs.set_title("Points")

        with utilities.gb_plotter(OUTPUT_DIR / "points_bb.png") as (fig, axs):
            points_2d.plot(axs, b_box=True)
            axs.grid()
            axs.set_title("Points")

        with utilities.gb_plotter(OUTPUT_DIR / "points_bb_black_dashed.png") as (
            fig,
            axs,
        ):
            points_2d.plot(
                axs,
                b_box=True,
                b_box_plt_opt={"color": "k", "linewidth": 2, "linestyle": "dashed"},
            )
            axs.grid()
            axs.set_title("Points")

        with utilities.gb_plotter(OUTPUT_DIR / "points_blue_color_cross.png") as (
            fig,
            axs,
        ):
            points_2d.plot(
                axs,
                points_plt_opt={
                    "color": "blue",
                    "marker": "x",
                    "linestyle": "None",
                    "markersize": 10,
                },
            )
            axs.grid()
            axs.set_title("Points")

        with utilities.gb_plotter(OUTPUT_DIR / "points_as_line.png") as (fig, axs):
            points_2d.plot(
                axs,
                points_plt_opt={
                    "color": "blue",
                    "marker": "x",
                    "linestyle": "solid",
                    "markersize": 10,
                },
            )
            axs.grid()
            axs.set_title("Points")
