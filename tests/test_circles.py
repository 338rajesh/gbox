import math
import pathlib
import datetime

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings

import gbox as gb
from gbox.utilities import get_output_dir

# ---------------------------------------------------------------------------

radius_strategy = st.floats(min_value=0.1, max_value=5.0)

point_sel_strategy = st.tuples(
    st.floats(min_value=-1.0, max_value=1.0),
    st.floats(min_value=-1.0, max_value=1.0),
)

num_circles_strategy = st.integers(min_value=2, max_value=10)

theta_strategy = st.tuples(
    st.floats(min_value=0.0, max_value=1.9 * np.pi),
    st.floats(min_value=0.0, max_value=2.0 * np.pi),
).map(lambda t: (min(t), max(t) if min(t) < max(t) else min(2.0 * np.pi, max(t) + 0.1)))

PI = np.pi
TWO_PI = 2.0 * PI
OUTPUT_DIR = get_output_dir(
    pathlib.Path(__file__).parent / "__output" / "test_circles"
)

# ---------------------------------------------------------------------------


class TestCircle:

    @hypothesis_settings(max_examples=20)
    @given(radius_strategy, point_sel_strategy, theta_strategy)
    def test_circle_construction(self, radius, centre, angles):
        xc, yc = centre
        th_1, th_2 = sorted(angles)
        circle = gb.Circle(radius, (xc, yc), th_1, th_2)

        #
        assert circle.radius == radius, "Radius does not match"
        assert circle.centre == (xc, yc), "Centre does not match"
        assert circle.theta_1 == th_1, "Theta 1 does not match"
        assert circle.theta_2 == th_2, "Theta 2 does not match"

        #

    @hypothesis_settings(max_examples=20)
    @given(radius_strategy, point_sel_strategy, theta_strategy)
    def test_circle_properties(self, radius, centre, angles):
        circle = gb.Circle(radius, centre, *angles)
        assert circle.area == math.pi * radius * radius, "Area does not match"
        assert circle.perimeter == 2 * math.pi * radius, "Perimeter does not match"

    @given(
        radius_strategy,
        point_sel_strategy,
        theta_strategy,
        st.integers(min_value=1, max_value=1000),  # noqa
    )
    def test_circle_eval_boundary(self, radius, centre, angles, n):
        th_1, th_2 = angles
        circle = gb.Circle(radius, centre, th_1, th_2)
        circle.eval_boundary(n)
        points = circle.boundary
        assert len(points) == n
        assert points.dim == 2
        assert points.coordinates.shape == (n, 2)

        tht = np.linspace(th_1, th_2, n)
        assert np.array_equal(
            points.coordinates[:, 0], radius * np.cos(tht) + centre[0]
        )
        assert np.array_equal(
            points.coordinates[:, 1], radius * np.sin(tht) + centre[1]
        )

    @given(radius_strategy, point_sel_strategy, theta_strategy)
    def test_circle_has_a_point(self, radius, centre, angles):
        xc, yc = centre
        th_1, th_2 = angles
        circle = gb.Circle(radius, centre, th_1, th_2)
        for i in [0.1 * i for i in range(1, 20)]:
            xp_in = (0.8 * radius) * np.cos(i) + xc
            yp_in = (0.8 * radius) * np.sin(i) + yc
            assert circle.contains_point((xp_in, yp_in)) == pytest.approx(1.0)
            #
            xp_on = (1.0 * radius) * np.cos(i) + xc
            yp_on = (1.0 * radius) * np.sin(i) + yc
            assert circle.contains_point((xp_on, yp_on)) == pytest.approx(0.0)
            #
            xp_out = (1.2 * radius) * np.cos(i) + xc
            yp_out = (1.2 * radius) * np.sin(i) + yc
            assert circle.contains_point((xp_out, yp_out)) == pytest.approx(-1.0)

    @given(radius_strategy, point_sel_strategy, theta_strategy, point_sel_strategy)
    def test_circles_distance(self, radius, centre, angles, point):
        xc, yc = centre
        th_1, th_2 = angles
        circle = gb.Circle(radius, centre, th_1, th_2)
        xp, yp = point
        circle_2 = gb.Circle(radius, point, th_1, th_2)
        assert circle.distance_to(circle_2) == (
            gb.Point2D(xp, yp).distance_to(gb.Point2D(xc, yc))
        )

    def test_circle_plot(self, test_plots):
        if not test_plots:
            pytest.skip()

        circle = gb.Circle(2.0, (3.0, 6.0), 0.0, 2.0 * math.pi)
        f_path = OUTPUT_DIR / "circle.png"
        with gb.utilities.gb_plotter(f_path) as (fig, axs):
            circle.plot(
                axs,
                b_box=True,
                b_box_plt_opt={
                    "color": "b",
                    "linewidth": 2,
                    "linestyle": "dashed",
                },
                points_plt_opt={
                    "color": "k",
                    "linewidth": 2,
                    "marker": "None",
                    "linestyle": "solid",
                },
            )
            axs.axis("off")


class TestCircles:

    def test_empty_circles_construction(self):
        with pytest.raises(ValueError):
            gb.Circles()  # Must have at least one circle
        with pytest.raises(TypeError):
            gb.Circles(
                1.0, 2.0, 3.0
            )  # 'Circle' type of elements are expected not floats

    @given(radius_strategy, point_sel_strategy, theta_strategy, num_circles_strategy)
    def test_circles_construction(self, radius, centre, angles, n):
        th_1, th_2 = angles
        xc, yc = centre
        radii = [radius * i * 0.1 for i in range(1, n)]
        x_c = [xc + np.random.rand() for i in range(1, n)]
        y_c = [yc + np.random.rand() for i in range(1, n)]
        m = n - 1
        circles = gb.Circles(
            *[gb.Circle(radii[i], (x_c[i], y_c[i]), th_1, th_2) for i in range(m)]
        )

        # Check number of circles
        assert circles.size == m, "Number of circles does not match"
        assert len(circles) == m, "Number of circles does not match"
        assert circles.data.shape == (m, 3), "Circles data shape does not match"

    @hypothesis_settings(
        max_examples=20,
        deadline=datetime.timedelta(milliseconds=300),
    )
    @given(radius_strategy, point_sel_strategy, theta_strategy, num_circles_strategy)
    def test_circles_iteration(self, radius, centre, angles, n):
        xc, yc = centre
        radii = np.array([radius * i * 0.1 for i in range(1, n)])
        x_c = [xc + np.random.rand() for i in range(1, n)]
        y_c = [yc + np.random.rand() for i in range(1, n)]
        xy_c = np.stack((x_c, y_c), axis=1)
        m = n - 1
        circles = gb.Circles(
            *[gb.Circle(radii[i], (x_c[i], y_c[i]), *angles) for i in range(m)]
        )
        # check iteration
        for i, a_c in enumerate(circles):
            assert a_c[0] == x_c[i], f"{a_c[0]} != {x_c[i]}"
            assert a_c[1] == y_c[i], f"{a_c[1]} != {y_c[i]}"
            assert a_c[2] == radii[i], f"{a_c[2]} != {radii[i]}"

        # Check data
        assert np.array_equal(circles.xc, x_c), "X coordinates do not match"
        assert np.array_equal(circles.yc, y_c), "Y coordinates do not match"
        assert np.array_equal(circles.radii, radii), "Radii do not match"
        assert np.array_equal(circles.centres, xy_c), "Centres do not match"

        # eval boundaries
        circles.evaluate_boundaries()
        assert len(circles.boundaries) == m, "Number of boundaries does not match"

        # perimeters and areas
        assert np.array_equal(circles.perimeters(), 2 * np.pi * radii)
        assert np.array_equal(circles.areas(), np.pi * radii * radii)

    def test_circles_plotting(self, test_plots):
        if not test_plots:
            pytest.skip("Skipping test because test_plots is False")
        circles = gb.Circles(
            gb.Circle(1.0, (0.0, 1.0)),
            gb.Circle(0.8, (0.0, 0.0)),
            gb.Circle(0.6, (1.0, 0.0)),
            gb.Circle(0.4, (1.0, 1.0)),
            gb.Circle(0.2, (0.0, 1.0)),
        )
        with gb.utilities.gb_plotter(OUTPUT_DIR / "circles.png") as (fig, axs):
            circles.plot(
                axs,
                b_box=True,
                b_box_plt_opt={
                    "color": "b",
                    "linewidth": 2,
                    "linestyle": "dashed",
                },
                points_plt_opt={
                    "color": "k",
                    "linewidth": 2,
                    "marker": "None",
                    "linestyle": "solid",
                },
            )

    def test_circles_have_point(self):
        r, m = 0.1, 10

        circles = gb.Circles(*[gb.Circle(r, (-i, i)) for i in range(m)])

        for i in range(m):
            for j, check in [(0.2, 1.0), (1.0, 0.0), (2.0, -1.0)]:
                assert (
                    circles.contains_point(
                        (-i + r * j * np.cos(PI * 0.25), i + r * j * np.sin(PI * 0.25))
                    )
                    == check
                )

    @given(point_sel_strategy, radius_strategy)
    def test_circles_distance_to(self, point, r):
        m = 10
        px, py = point

        circles = gb.Circles(*[gb.Circle(r, (-i, i)) for i in range(m)])

        assert np.array_equal(
            circles.distances_to((px, py)),
            np.stack((np.sqrt((circles.xc - px) ** 2 + (circles.yc - py) ** 2))),
        ), "Distance to points does not match"

    @given(point_sel_strategy, radius_strategy)
    def test_circles_eval_boundaries(self, point, r):
        (px, py), m = point, 10
        circles_list = [gb.Circle(r, (-i, i)) for i in range(m)]
        circles = gb.Circles(*circles_list)
        circles.evaluate_boundaries(num_points=123)
        circles_boundaries = circles.boundaries
        for idx, a_circ in enumerate(circles_list):
            a_circ.eval_boundary(num_points=123)
            assert np.array_equal(circles_boundaries[idx].x, a_circ.boundary.x)
            assert np.array_equal(circles_boundaries[idx].y, a_circ.boundary.y)
