import pathlib
import math

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings
import numpy as np

import gbox as gb
from gbox.utilities import get_output_dir, LineStyles as lst

# ----------------------------------------------------------------------------

PI = np.pi
TWO_PI = 2.0 * PI
OUTPUT_DIR = get_output_dir(
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
            with gb.utilities.gb_plotter(OUTPUT_DIR / f"ellipse_{i}.png") as (fig, axs):
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
            with gb.utilities.gb_plotter(file_path) as (fig, axs):
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
