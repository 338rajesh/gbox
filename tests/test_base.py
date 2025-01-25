import pathlib

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings

import gbox as gb

from utils import get_output_dir, gb_plotter

OUTPUT_DIR = get_output_dir(pathlib.Path(__file__).parent / "__output" / "test_points")

# ==================== TESTS ====================


class TestBoundingBox:
    @hypothesis_settings(max_examples=10)
    @given(
        n=st.integers(min_value=2, max_value=10),
        l=st.floats(min_value=-10.0, max_value=0.0),
        u=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_b_box(self, n, l, u):
        lbl, ubl = [l] * n, [u] * n
        bb = gb.BoundingBox(lower_bound=lbl, upper_bound=ubl)
        bb_t = gb.BoundingBox(lower_bound=tuple(lbl), upper_bound=tuple(ubl))
        bb_a = gb.BoundingBox(lower_bound=np.array(lbl), upper_bound=np.array(ubl))
        #
        assert bb == bb_t
        assert bb == bb_a
        assert bb.dim == n
        assert bb.vertices is not None

        assert bb.has_point([(l + u) * 0.5] * n)
        assert not bb.has_point([l - 1.0] * n)

        with pytest.raises(ValueError):
            gb.BoundingBox(lower_bound=[l] * (n + 1), upper_bound=[u] * n)

        with pytest.raises(ValueError):
            gb.BoundingBox(lower_bound=ubl, upper_bound=lbl)

    def test_bounds_ele_type(self):
        with pytest.raises(ValueError):
            gb.BoundingBox(lower_bound=[1.0, 1.0], upper_bound=[0.0, "0.0"])

    @given(
        st.integers(min_value=1, max_value=6),
        st.floats(min_value=-10.0, max_value=0.0, allow_nan=False),
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
    )
    def test_volume(self, n, l, u):
        b_box = gb.BoundingBox([l for _ in range(n)], [u for _ in range(n)])
        assert all(np.isclose(b_box.side_lengths(), [u - l for _ in range(n)]))
        assert np.isclose(b_box.volume, (u - l) ** n)

    def test_full_overlap(self):
        # Test case where the boxes are identical, so they fully overlap
        bb1 = gb.BoundingBox([0, 0], [5, 5])
        bb2 = gb.BoundingBox([0, 0], [5, 5])
        bb3 = gb.BoundingBox([1, 1], [3, 3])
        assert bb1.overlaps_with(bb2)
        assert bb1.overlaps_with(bb3)

    def test_partial_overlap(self):
        # Test case where the boxes partially overlap
        bb1 = gb.BoundingBox([0, 0], [5, 5])
        bb2 = gb.BoundingBox([3, 3], [7, 7])
        bb3 = gb.BoundingBox([-3, -3], [1, 1])
        bb4 = gb.BoundingBox([-3, 2], [1, 7])
        assert bb1.overlaps_with(bb2)
        assert bb1.overlaps_with(bb3)
        assert bb1.overlaps_with(bb4)

    def test_no_overlap(self):
        # Test case where the boxes don't overlap at all
        bb1 = gb.BoundingBox([0, 0], [5, 5])
        bb2 = gb.BoundingBox([6, 6], [10, 10])
        assert not bb1.overlaps_with(bb2)

    def test_edge_touching_overlap(self):
        # Test case where boxes touch at the edge but don't overlap
        bb1 = gb.BoundingBox([0, 0], [5, 5])
        bb2 = gb.BoundingBox([5, 0], [10, 5])
        assert bb1.overlaps_with(bb2, incl_bounds=True)
        assert not bb1.overlaps_with(bb2, incl_bounds=False)

    def test_bbox_plots(self, test_plots):
        if not test_plots:
            pytest.skip()

        with gb_plotter(OUTPUT_DIR / "bbox.png") as (fig, axs):
            bb = gb.BoundingBox([0, 0], [5, 5])
            bb.plot(axs)
            axs.set_title("Bounding Box")

        with pytest.raises(ValueError):
            with gb_plotter(OUTPUT_DIR / "bbox_1.png") as (fig, axs):
                gb.BoundingBox([0, 0, 1], [5, 5, 10]).plot(axs)


# ==================== Testing Point =====================






@pytest.fixture
def points():
    return gb.PointArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


@pytest.fixture
def points_1d():
    return gb.PointArray1D(np.array([1.0, 3.0, 5.0, 7.0]).reshape(-1, 1))


@pytest.fixture
def points_2d():
    return gb.PointSet2D([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


# @pytest.fixture(params=[0.0, 0.2, 1.24, 2.7, 3.14, 4.0])
# def transformation_triplet(request):
#     return (np.pi * 0.5 * request.param, np.random.rand(), np.random.rand())


# ==================== TESTS =====================

# class TestPoint2D:
    # def test_point_2d(self, point_2d):
    #     assert point_2d.dim == 2
    #     assert point_2d.x == 1.0
    #     assert point_2d.y == 2.0
    #     assert point_2d == gb.Point2D(1.0, 2.0)

    # def test_slope(self, point_2d):
    #     assert point_2d.slope(gb.Point2D(3.0, 4.0)) == pytest.approx(1.0, rel=1e-5)
    #     assert point_2d.slope(gb.Point2D(-1.0, -6.0)) == pytest.approx(4.0, rel=1e-5)

    # def test_angle(self, point_2d):
    #     assert point_2d.angle(gb.Point2D(2.0, 3.0)) == pytest.approx(
    #         np.pi * 0.25, rel=1e-5
    #     )
    #     assert point_2d.angle(gb.Point2D(0.0, 3.0)) == pytest.approx(
    #         np.pi * 0.75, rel=1e-5
    #     )
    #     assert point_2d.angle(gb.Point2D(0.0, 1.0)) == pytest.approx(
    #         np.pi * 1.25, rel=1e-5
    #     )
    #     assert point_2d.angle(gb.Point2D(2.0, 1.0)) == pytest.approx(
    #         np.pi * 1.75, rel=1e-5
    #     )

    # def test_transform(self):
    #     assert (
    #         gb.Point2D(0.0, 1.0)
    #         .transform(angle=np.pi * 1.5, dx=5.0, dy=6.0)
    #         .is_close_to(gb.Point2D(6.0, 6.0), eps=1e-5)
    #     )
    #     assert (
    #         gb.Point2D(-3.0, -3.0)
    #         .transform(angle=np.pi * 1.0, dx=-3.0, dy=-3.0)
    #         .is_close_to(gb.Point2D(0.0, 0.0), eps=1e-5)
    #     )
    #     assert (
    #         gb.Point2D(3.0, -3.0)
    #         .transform(angle=np.pi * 0.25, dx=-3.0 * np.sqrt(2.0), dy=0.0)
    #         .is_close_to(gb.Point2D(0.0, 0.0), eps=1e-5)
    #     )


# class TestPoints:
#     def test_points_constructor(self):
#         gb.PointSet(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
#         gb.PointSet([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#         gb.PointSet(([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]))
#         gb.PointSet(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))

#     def test_points_properties(self, points):
#         assert points.dim == 2
#         assert len(points) == 4

#     def test_points_repr(self, points):
#         out = "PointSet:\n[[1. 2.]\n [3. 4.]\n [5. 6.]\n [7. 8.]]"
#         assert str(points) == out

#     def test_points_eq(self, points):
#         assert points == gb.PointSet([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

#     def test_points_copy(self, points):
#         assert points.copy() == points
#         # Check that the copy is not the same object
#         assert points.copy() is not points

#     def test_points_bounding_box(self, points):
#         assert points.bounding_box == gb.BoundingBox([1.0, 2.0], [7.0, 8.0])

#     def test_points_from_dimension_data(self, points):
#         new_points = gb.PointSet.from_dimension_data(
#             np.array([1.0, 3.0, 5.0, 7.0]), np.array([2.0, 4.0, 6.0, 8.0])
#         )
#         assert new_points == points
#         assert new_points.dim == 2
#         assert len(new_points) == 4

#     def test_points_cycle(self, points):
#         points.cycle = True


# class TestPointSet1D:
#     def test_constructor(self):
#         p_1d = gb.PointArray1D([[1.0], [2.0], [3.0]])
#         assert p_1d.dim == 1
#         assert np.array_equal(p_1d.x, [1.0, 2.0, 3.0])

#     def test_transform(self, points_1d):
#         assert np.array_equal(
#             points_1d.transform(dx=0.25).coordinates,
#             np.array([[1.25], [3.25], [5.25], [7.25]]),
#         )

#     def test_reverse(self, points_1d):
#         points_1d.reverse()
#         assert np.array_equal(points_1d.x, [7.0, 5.0, 3.0, 1.0])

#     def test_1d_plots(self, points_1d, test_plots):
#         if not test_plots:
#             pytest.skip()

#         with gb.gb_plotter(OUTPUT_DIR / "points_1d_simple.png") as (fig, axs):
#             points_1d.plot(axs)
#             axs.grid()
#             axs.set_title("PointSet")


# class TestPointSet1D:
#     def test_constructor(self, points_2d):
#         assert points_2d.dim == 2
#         assert np.array_equal(points_2d.x, [1.0, 3.0, 5.0, 7.0])
#         assert np.array_equal(points_2d.y, [2.0, 4.0, 6.0, 8.0])

#     @given(
#         dth=st.floats(min_value=0.0, max_value=np.pi * 2.0),
#         dx=st.floats(min_value=-1.0, max_value=1.0),
#         dy=st.floats(min_value=-1.0, max_value=1.0),
#     )
#     def test_transform(self, dth, dx, dy):
#         dth = float(np.random.choice([0.0, np.pi * 2.0]))
#         init_xy = np.random.rand(10, 2)
#         trasnformed_x = init_xy[:, 0] * np.cos(dth) - init_xy[:, 1] * np.sin(dth) + dx
#         trasnformed_y = init_xy[:, 0] * np.sin(dth) + init_xy[:, 1] * np.cos(dth) + dy
#         trasnformed_xy = np.column_stack((trasnformed_x, trasnformed_y))
#         #
#         points_ = gb.PointSet2D(init_xy)
#         new_point_coordinates = points_.transform(dth, dx, dy).coordinates
#         assert np.allclose(trasnformed_xy, new_point_coordinates)

#     def test_points_2d_angle(self, point_2d):
#         assert point_2d.angle(gb.Point2D(2.0, 3.0)) == pytest.approx(
#             np.pi * 0.25, rel=1e-5
#         )

#     def test_reverse(self, points_2d):
#         points_2d.reverse()
#         assert np.array_equal(points_2d.x, [7.0, 5.0, 3.0, 1.0])
#         assert np.array_equal(points_2d.y, [8.0, 6.0, 4.0, 2.0])

#     def test_plots(self, points_2d, test_plots):
#         if not test_plots:
#             pytest.skip()

#         with gb.gb_plotter(OUTPUT_DIR / "points_simple.png") as (fig, axs):
#             points_2d.plot(axs)
#             axs.grid()
#             axs.set_title("PointSet")

#         with gb.gb_plotter(OUTPUT_DIR / "points_bb.png") as (fig, axs):
#             points_2d.plot(axs, b_box=True)
#             axs.grid()
#             axs.set_title("PointSet")

#         with gb.gb_plotter(OUTPUT_DIR / "points_bb_black_dashed.png") as (
#             fig,
#             axs,
#         ):
#             points_2d.plot(
#                 axs,
#                 b_box=True,
#                 b_box_plt_opt={"color": "k", "linewidth": 2, "linestyle": "dashed"},
#             )
#             axs.grid()
#             axs.set_title("PointSet")

#         with gb.gb_plotter(OUTPUT_DIR / "points_blue_color_cross.png") as (
#             fig,
#             axs,
#         ):
#             points_2d.plot(
#                 axs,
#                 points_plt_opt={
#                     "color": "blue",
#                     "marker": "x",
#                     "linestyle": "None",
#                     "markersize": 10,
#                 },
#             )
#             axs.grid()
#             axs.set_title("PointSet")

#         with gb.gb_plotter(OUTPUT_DIR / "points_as_line.png") as (fig, axs):
#             points_2d.plot(
#                 axs,
#                 points_plt_opt={
#                     "color": "blue",
#                     "marker": "x",
#                     "linestyle": "solid",
#                     "markersize": 10,
#                 },
#             )
#             axs.grid()
#             axs.set_title("PointSet")
