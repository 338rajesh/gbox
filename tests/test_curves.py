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
