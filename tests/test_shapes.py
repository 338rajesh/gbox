import pytest

import numpy as np
import gbox as gb

# ==================== FIXTURES =====================


@pytest.fixture
def ell_arc_1():
    return gb.EllipticalArc(2.0, 0.5, 0.0, 0.25 * np.pi, 0.0, (0.0, 0.0))


@pytest.fixture
def cir_arc_1():
    return gb.CircularArc(0.25, -0.5 * np.pi, 0.5 * np.pi, (0.0, 0.0))


@pytest.fixture(params=[i for i in (1, 3, 4)])
def ell_arcs(request):
    return (
        gb.EllipticalArc(
            request.param * 0.265,
            0.225,
            -np.pi * 0.1 * request.param,
            np.pi * 0.2 * request.param,
            np.pi * 0.15 * request.param,
            (0.2 * request.param, -0.12 * request.param),
        ),
        request.param,
    )


@pytest.fixture
def ell_arc_2():

    def _ell_arc_2(i):
        return gb.EllipticalArc(
            2.0 * i,
            1.0,
            -np.pi * 0.1 * i,
            np.pi * 0.15 * i,
            np.pi * 0.25 * i,
            (0.2 * i, -0.12 * i),
        )

    return _ell_arc_2


# # ==================== TESTS ========================


class TestStraightLine:
    def test_straight_line_1(self):
        line = gb.StraightLine((0.0, 0.0), (1.0, 1.0))
        assert line.p1 == (0.0, 0.0)
        assert line.p2 == (1.0, 1.0)
        assert line.length() == 1.4142135623730951
        eqn = line.equation()
        assert np.array_equal(eqn(0.0), (0.0, 0.0))
        assert np.array_equal(eqn(0.5), (0.5, 0.5))
        assert np.array_equal(eqn(1.0), (1.0, 1.0))

    def test_straight_line_2(self):
        line = gb.StraightLine((0.0, 0.0, 1.0), (1.0, 1.0, -1.0))
        assert line.p1 == (0.0, 0.0, 1.0)
        assert line.p2 == (1.0, 1.0, -1.0)
        assert line.length() == np.sqrt(6.0)
        assert np.array_equal(line.equation()(0.0), (0.0, 0.0, 1.0))
        assert np.array_equal(line.equation()(0.5), (0.5, 0.5, 0.0))
        assert np.array_equal(line.equation()(1.0), (1.0, 1.0, -1.0))

    def test_straight_line_2d(self):
        line = gb.StraightLine2D((1.0, 2.0), (3.0, 4.0))
        with pytest.raises(AssertionError):
            gb.StraightLine2D((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))

        assert line.p1 == (1.0, 2.0)
        assert line.p2 == (3.0, 4.0)
        assert line.length() == np.sqrt(8)
        assert np.array_equal(line.equation()(0.0), (1.0, 2.0))

    def test_straight_line_2d_angle(self):

        assert gb.StraightLine2D((0.0, 0.0), (1.0, 0.0)).angle() == 0.0
        assert gb.StraightLine2D((0.0, 0.0), (1.0, 0.0)).angle(rad=False) == 0.0

        assert gb.StraightLine2D((0.0, 0.0), (1.0, 1.0)).angle() == np.pi * 0.25
        assert gb.StraightLine2D((0.0, 0.0), (1.0, 1.0)).angle(rad=False) == 45.0

        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (2.0, 2.0 + np.sqrt(3.0))).angle(), np.pi / 3
        )
        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (2.0, 2.0 + np.sqrt(3.0))).angle(rad=False),
            60.0,
        )

        assert gb.StraightLine2D((0.0, 0.0), (0.0, 1.0)).angle() == np.pi * 0.5
        assert gb.StraightLine2D((0.0, 0.0), (0.0, 1.0)).angle(rad=False) == 90.0

        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (-2.0, 5.0)).angle(), np.pi * 0.75
        )
        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (-2.0, 5.0)).angle(rad=False), 135.0
        )

        assert gb.StraightLine2D((0.0, 0.0), (-0.5, 0.0)).angle() == np.pi
        assert gb.StraightLine2D((0.0, 0.0), (-0.5, 0.0)).angle(rad=False) == 180.0

        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (-2.0, -1.0)).angle(), np.pi * 1.25
        )
        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (-2.0, -1.0)).angle(rad=False), 225.0
        )

        assert gb.StraightLine2D((0.0, 0.0), (0.0, -1.0)).angle() == np.pi * 1.5
        assert gb.StraightLine2D((0.0, 0.0), (0.0, -1.0)).angle(rad=False) == 270.0

        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (4.0, -1.0)).angle(), np.pi * 1.75
        )
        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (4.0, -1.0)).angle(rad=False), 315.0
        )

        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (2.0, 2.0 - np.sqrt(3.0))).angle(),
            np.pi * 5.0 / 3.0,
        )
        assert np.isclose(
            gb.StraightLine2D((1.0, 2.0), (2.0, 2.0 - np.sqrt(3.0))).angle(rad=False),
            300.0,
        )


import pathlib
import math

import pytest
from hypothesis import given, strategies as st
import numpy as np

import gbox as gb
from gbox import LineStyles as lst

# ----------------------------------------------------------------------------

PI = np.pi
TWO_PI = 2.0 * PI
OUTPUT_DIR = gb.get_output_dir(
    pathlib.Path(__file__).parent / "__output" / "test_ellipses"
)


@pytest.fixture
def ellipse_1():
    return gb.Ellipse(1.0, 0.5, math.pi * 0.25, (0.0, 0.0))


point_sel_strategy = st.tuples(
    st.floats(min_value=-1.0, max_value=1.0),
    st.floats(min_value=-1.0, max_value=1.0),
)

theta_strategy = st.tuples(
    st.floats(min_value=0.0, max_value=1.9 * np.pi),
    st.floats(min_value=0.0, max_value=2.0 * np.pi),
).map(lambda t: (min(t), max(t) if min(t) < max(t) else min(2.0 * np.pi, max(t) + 0.1)))


# ----------------------------------------------------------------------------


class TestEllipse:
    def test_ellipse_construction(self, ellipse_1):
        assert ellipse_1.smj == 1.0
        assert ellipse_1.smn == 0.5
        assert ellipse_1.mjx_angle == math.pi * 0.25
        assert ellipse_1.centre == (0.0, 0.0)

    @given(
        st.floats(min_value=1.0, max_value=5.0),
        st.floats(min_value=0.1, max_value=100.0),
        theta_strategy,
        point_sel_strategy,
    )
    def test_ellipse_properties(self, aspect_ratio, smj, mjx_angle, centre):
        smn = smj / aspect_ratio
        ell = gb.Ellipse(smj, smn, mjx_angle, centre)

        perimeter = math.pi * (
            (3.0 * (smj + smn)) - math.sqrt(((3.0 * smj) + smn) * (smj + (3.0 * smn)))
        )
        area = math.pi * smj * smn
        assert ell.area == pytest.approx(area)
        assert ell.perimeter == pytest.approx(perimeter, rel=1e-03)
        assert ell.aspect_ratio == pytest.approx(smj / smn)
        assert ell.eccentricity == pytest.approx(math.sqrt(1 - (smn / smj) ** 2))
        assert ell.shape_factor == pytest.approx(
            perimeter / math.sqrt(4.0 * math.pi * area)
        )
        if smj == smn:
            assert ell.aspect_ratio == pytest.approx(1.0)
            assert ell.perimeter == pytest.approx(2.0 * math.pi * smj)
            assert ell.area == pytest.approx(math.pi * smj**2)
            assert ell.eccentricity == pytest.approx(0.0)
            assert ell.shape_factor == pytest.approx(1.0)

    def test_contains_point(self):
        ell = gb.Ellipse(1.0, 0.5, math.pi * 0.0, (0.0, 0.0))
        assert ell.contains_point((0.0, 0.0)) == pytest.approx(1.0)
        assert ell.contains_point((1.0, 0.0)) == pytest.approx(0.0)
        assert ell.contains_point((0.0, 1.0)) == pytest.approx(-1.0)

        for i in range(10):
            tht = math.pi * 0.2 * i
            xc, yc = -5.0, 5.0
            ell = gb.Ellipse(1.0, 0.5, tht, (xc, yc))
            assert ell.contains_point(
                (xc + ell.smn * 0.2, yc + ell.smn * 0.2)
            ) == pytest.approx(1.0)
            assert ell.contains_point(
                (xc + ell.smj * np.cos(tht), yc + ell.smj * np.sin(tht))
            ) == pytest.approx(0.0)

    def test_ellipse_eval_boundary(self):
        xc, yc, tht, a, b, n = 1.0, 2.0, math.pi * 0.25, 1.0, 0.5, 100
        ell = gb.Ellipse(a, b, tht, (xc, yc))
        ell.eval_boundary(n)
        assert len(ell.boundary) == n
        assert ell.boundary.dim == 2
        assert ell.boundary.coordinates.shape == (n, 2)
        assert ell.boundary.x[0] == pytest.approx(xc + a * np.cos(tht))
        assert ell.boundary.x[-1] == pytest.approx(xc + a * np.cos(tht))
        assert ell.boundary.y[0] == pytest.approx(yc + a * np.sin(tht))
        assert ell.boundary.y[-1] == pytest.approx(yc + a * np.sin(tht))
        #
        t = np.linspace(0.0, 2.0 * math.pi, n)
        xs = a * np.cos(t)
        ys = b * np.sin(t)
        assert np.array_equal(
            ell.boundary.x, (xs * np.cos(tht) - ys * np.sin(tht)) + xc
        )
        assert np.array_equal(
            ell.boundary.y, (xs * np.sin(tht) + ys * np.cos(tht)) + yc
        )

        #
        with pytest.raises(AssertionError):
            ell.eval_boundary(n, False)
            assert ell.boundary.coordinates[-1, 0] == pytest.approx(
                xc + a * np.cos(tht)
            )
            assert ell.boundary.coordinates[-1, 1] == pytest.approx(
                yc + b * np.sin(tht)
            )

    def test_ellipse_plot(self, test_plots):
        if not test_plots:
            pytest.skip("test_plots not enabled, use --plots option to enable")

        for i in range(6):
            xc, yc = np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)
            a, b = np.random.uniform(5.0, 8.0), np.random.uniform(1.0, 2.0)
            tht = np.random.uniform(0.0, 2.0 * math.pi)
            t = np.linspace(0.0, 2.0 * math.pi, 100)
            n = 100
            ell = gb.Ellipse(a, b, tht, (xc, yc))
            ell.eval_boundary(n)

            xi_ = xc + (a * np.cos(t) * np.cos(tht)) - (b * np.sin(t) * np.sin(tht))
            yi_ = yc + (a * np.cos(t) * np.sin(tht)) + (b * np.sin(t) * np.cos(tht))
            with gb.gb_plotter(OUTPUT_DIR / f"ellipse_{i}.png") as (fig, axs):
                ell.plot(
                    axs, True, lst.THEME_2, {**lst.THEME_1, "label": "Test Ellipse"}
                )

                axs.plot(
                    xi_, yi_, "k", linestyle="--", linewidth=2, label="Target Ellipse"
                )
                axs.plot([xc, xc + a * np.cos(tht)], [yc, yc + a * np.sin(tht)], "g-")

                axs.grid()
                axs.set_title(
                    f"Ellipse: a={a:4.3f},  b={b:4.3f}, tht={np.rad2deg(tht):4.3f}"
                )
                axs.plot(xc, yc, "ro")  # Add the point as a red dot
                axs.annotate(
                    f"P({xc:4.3f}, {yc:4.3f})",  # Text to display
                    (xc, yc),  # Point to annotate
                    textcoords="offset points",  # Positioning with an offset
                    xytext=(10, 10),  # Offset in pixels
                    ha="center",  # Horizontal alignment
                    arrowprops=dict(arrowstyle="->", color="black"),  # Optional arrow
                )
                axs.axis("equal")

    def test_ellipse_r_shortest(self):
        ell = gb.Ellipse(1.0, 0.5, math.pi * 0.25, (0.0, 0.0))
        xi = ell.smj * ell.eccentricity**2
        assert ell.r_shortest(0.0) == pytest.approx(0.5)
        assert ell.r_shortest(xi) == pytest.approx(ell.smn / ell.aspect_ratio)
        assert ell.r_shortest(-xi) == pytest.approx(ell.smn / ell.aspect_ratio)

    def test_ellipse_uns(self):
        for i in range(20):
            a = np.random.uniform(5.0, 8.0)
            b = np.random.uniform(1.0, 4.0)
            tht = np.random.uniform(0.0, TWO_PI)
            xc = np.random.uniform(-5.0, 5.0)
            yc = np.random.uniform(-5.0, 5.0)
            dh = np.random.uniform(0.01, 0.02)
            ell = gb.Ellipse(a, b, tht, (xc, yc))
            circles: gb.CircleSet = ell.uns(dh)
            file_path = OUTPUT_DIR / f"uns_{i}.png"
            with gb.gb_plotter(file_path) as (fig, axs):
                ell.plot(
                    axs,
                    True,
                    lst.THEME_3,
                    {"linewidth": 2, "color": "k", "label": "Test Ellipse"},
                )
                circles.plot(
                    axs, False, points_plt_opt={**lst.THEME_1, "label": "Test Circles"}
                )
                axs.set_title(
                    f"Ellipse: a={a:4.3f},  b={b:4.3f}, tht={np.rad2deg(tht):4.3f}, dh={dh:4.3f}, N={circles.size}"
                )
                axs.plot(xc, yc, "ro")  # Add the point as a red dot
                axs.axis("equal")

import math
import pathlib
import datetime

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings

import gbox as gb

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
OUTPUT_DIR = gb.get_output_dir(
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
        with gb.gb_plotter(f_path) as (fig, axs):
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


class TestCircleSet:

    def test_empty_circles_construction(self):
        with pytest.raises(ValueError):
            gb.CircleSet()  # Must have at least one circle
        with pytest.raises(TypeError):
            gb.CircleSet(
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
        circles = gb.CircleSet(
            *[gb.Circle(radii[i], (x_c[i], y_c[i]), th_1, th_2) for i in range(m)]
        )

        # Check number of circles
        assert circles.size == m, "Number of circles does not match"
        assert len(circles) == m, "Number of circles does not match"
        assert circles.data.shape == (m, 3), "CircleSet data shape does not match"

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
        circles = gb.CircleSet(
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
        circles = gb.CircleSet(
            gb.Circle(1.0, (0.0, 1.0)),
            gb.Circle(0.8, (0.0, 0.0)),
            gb.Circle(0.6, (1.0, 0.0)),
            gb.Circle(0.4, (1.0, 1.0)),
            gb.Circle(0.2, (0.0, 1.0)),
        )
        with gb.gb_plotter(OUTPUT_DIR / "circles.png") as (fig, axs):
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

    def test_circles_transformation(self):
        r, m = 0.1, 10

        for i in range(50):
            xc_yc = np.random.uniform(-5.0, 5.0, (m, 2))
            r = np.random.uniform(1.0, 5.0, m)
            (dx, dy) = np.random.uniform(-5.0, 5.0, 2)
            tht = np.random.uniform(-np.pi, np.pi)
            sc = np.random.uniform(-2.0, 2.0)

            circles = gb.CircleSet(
                *[gb.Circle(r[k], (xc, yc)) for k, (xc, yc) in enumerate(xc_yc)]
            )
            circles.transform(dx, dy, tht, sc)
            assert np.allclose(
                circles.radii, np.array([r * sc for j in range(m)])
            ), "Radii do not match"

            x_ = xc_yc[:, 0] * np.cos(tht) - xc_yc[:, 1] * np.sin(tht) + dx
            y_ = xc_yc[:, 0] * np.sin(tht) + xc_yc[:, 1] * np.cos(tht) + dy
            assert np.allclose(circles.xc, x_), "X coordinates do not match"
            assert np.allclose(circles.yc, y_), "Y coordinates do not match"

    def test_circles_have_point(self):
        r, m = 0.1, 10

        circles = gb.CircleSet(*[gb.Circle(r, (-i, i)) for i in range(m)])

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

        circles = gb.CircleSet(*[gb.Circle(r, (-i, i)) for i in range(m)])

        assert np.array_equal(
            circles.distances_to((px, py)),
            np.stack((np.sqrt((circles.xc - px) ** 2 + (circles.yc - py) ** 2))),
        ), "Distance to points does not match"

    @given(point_sel_strategy, radius_strategy)
    def test_circles_eval_boundaries(self, point, r):
        (px, py), m = point, 10
        circles_list = [gb.Circle(r, (-i, i)) for i in range(m)]
        circles = gb.CircleSet(*circles_list)
        circles.evaluate_boundaries(num_points=123)
        circles_boundaries = circles.boundaries
        for idx, a_circ in enumerate(circles_list):
            a_circ.eval_boundary(num_points=123)
            assert np.array_equal(circles_boundaries[idx].x, a_circ.boundary.x)
            assert np.array_equal(circles_boundaries[idx].y, a_circ.boundary.y)

