import pytest

import numpy as np
import gbox as gb
import matplotlib.pyplot as plt

import pathlib
import shutil

from gbox.plot_utilities import LineStyles as lst

OUTPUT_DIR = pathlib.Path(__file__).parent / "__output" / "test_shapes"

# if OUTPUT_DIR.exists():
if OUTPUT_DIR.exists():
    print("Removing output directory...", OUTPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)


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


# ==================== TESTS ========================


class TestCurves:
    def test_elliptical_arc_construction(self, ell_arc_1):
        assert ell_arc_1.aspect_ratio == 4.0
        assert ell_arc_1.smj == 2.0
        assert ell_arc_1.smn == 0.5
        assert ell_arc_1.centre == (0.0, 0.0)
        assert ell_arc_1.mjx_angle == 0.0
        assert ell_arc_1.theta_1 == 0.0
        assert ell_arc_1.theta_2 == np.pi * 0.25
        assert ell_arc_1.eccentricity == np.sqrt(1 - (0.5 / 2.0) ** 2)

    def test_elliptical_arc_boundary(self, ell_arcs):
        ell_arc, i = ell_arcs
        boundary = ell_arc.eval_boundary(265).points
        assert boundary.coordinates.shape == (265, 2)
        assert boundary.dim == 2
        assert len(boundary) == 265
        assert ell_arc.centre == (0.2 * i, -0.12 * i)
        assert ell_arc.mjx_angle == np.pi * 0.15 * i
        assert ell_arc.theta_1 == -np.pi * 0.1 * i
        assert ell_arc.theta_2 == np.pi * 0.2 * i
        assert ell_arc.smj == i * 0.265
        assert ell_arc.smn == 0.225
        assert ell_arc.eccentricity == np.sqrt(1 - (0.225 / (i * 0.265)) ** 2)
        # Testing if the boundary ends are correct
        rot_matrix = gb.utilities.rotation_matrix_2d(ell_arc.mjx_angle)
        test_boundary_ends = np.array(
            [
                [0.265 * i * np.cos(ell_arc.theta_1), 0.225 * np.sin(ell_arc.theta_1)],
                [0.265 * i * np.cos(ell_arc.theta_2), 0.225 * np.sin(ell_arc.theta_2)],
            ]
        ) @ rot_matrix + np.array([0.2 * i, -0.12 * i])
        assert boundary.coordinates[0, 0] == test_boundary_ends[0, 0]
        assert boundary.coordinates[0, 1] == test_boundary_ends[0, 1]
        assert boundary.coordinates[-1, 0] == test_boundary_ends[-1, 0]
        assert boundary.coordinates[-1, 1] == test_boundary_ends[-1, 1]

    @pytest.mark.parametrize(
        "cycle, b_box, j, line_style",
        [
            (False, False, 1, lst.THEME_1),
            (False, True, 2, lst.THEME_2),
            (True, False, 3, lst.THEME_3),
            (True, True, 4, lst.THEME_4),
        ],
    )
    def test_elliptical_arc_plot(self, ell_arc_2, cycle, b_box, j, line_style):
        #
        fp = OUTPUT_DIR / f"elliptical_arc_j{j}_CYC{int(cycle)}_BB{int(b_box)}.png"
        with gb.utilities.plot_context(fp) as (fig, axs):
            ell_arc_2(j).eval_boundary(265).plot(
                axs,
                b_box=b_box,
                cycle=cycle,
                points_plt_opt=line_style,
            )
        #

    def test_circular_arc_construction(self, cir_arc_1):
        assert cir_arc_1.radius == 0.25
        assert cir_arc_1.theta_1 == -np.pi * 0.5
        assert cir_arc_1.theta_2 == np.pi * 0.5
        assert cir_arc_1.centre == (0.0, 0.0)
        assert cir_arc_1.eccentricity == 0.0
        assert cir_arc_1.aspect_ratio == 1.0
        cir_boundary = cir_arc_1.eval_boundary(265)

    @pytest.mark.parametrize(
        "num_points, cycle, b_box",
        [(20, True, False), (100, True, True), (500, False, True)],
    )
    def test_circular_arc_plot(self, cir_arc_1, num_points, cycle, b_box):
        #
        fp = (
            OUTPUT_DIR
            / f"circular_arc_CYCLE{int(cycle)}_BB{int(b_box)}_N{num_points}.png"
        )
        with gb.utilities.plot_context(fp) as (_, axs):
            cir_arc_1.eval_boundary(num_points).plot(
                axs,
                b_box=b_box,
                cycle=cycle,
                points_plt_opt=lst.THEME_1,
            )
