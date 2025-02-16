import pytest
from gbox import PointND
from gbox.shapes import (
    StraightLineND,
    StraightLine2D,
)
import numpy as np


class TestStraightLine:
    def test_straight_line_1(self):
        line = StraightLineND((0.0, 0.0, 1.0), (1.0, 1.0, -1.0))        
        assert isinstance(line, StraightLineND)
        assert line.p1 == (0.0, 0.0, 1.0)
        assert line.p2 == (1.0, 1.0, -1.0)
        assert line.length == pytest.approx(6.0**0.5)
        eqn = line.equation()
        assert np.array_equal(eqn(0.0), (0.0, 0.0, 1.0))
        assert np.array_equal(eqn(0.5), (0.5, 0.5, 0.0))
        assert np.array_equal(eqn(1.0), (1.0, 1.0, -1.0))

    def test_straight_line_2d(self):
        line = StraightLine2D((1.0, 2.0), (3.0, 4.0))
        assert isinstance(line, StraightLine2D)
        assert line.dim == 2
        with pytest.raises(ValueError):
            StraightLine2D((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))
        
        assert line.p1 == (1.0, 2.0)
        assert line.p2 == (3.0, 4.0)
        assert line.length == pytest.approx(8.0**0.5)
        assert np.array_equal(line.equation()(0.0), (1.0, 2.0))

    def test_straight_line_2d_angle(self):
        assert StraightLine2D((0.0, 0.0), (1.0, 0.0)).angle() == 0.0
        assert StraightLine2D((0.0, 0.0), (1.0, 0.0)).angle(rad=False) == 0.0

        assert StraightLine2D((0.0, 0.0), (1.0, 1.0)).angle() == np.pi * 0.25
        assert StraightLine2D((0.0, 0.0), (1.0, 1.0)).angle(rad=False) == 45.0

        assert np.isclose(
            StraightLine2D((1.0, 2.0), (2.0, 2.0 + np.sqrt(3.0))).angle(), np.pi / 3
        )
        assert np.isclose(
            StraightLine2D((1.0, 2.0), (2.0, 2.0 + np.sqrt(3.0))).angle(rad=False),
            60.0,
        )

        assert StraightLine2D((0.0, 0.0), (0.0, 1.0)).angle() == np.pi * 0.5
        assert StraightLine2D((0.0, 0.0), (0.0, 1.0)).angle(rad=False) == 90.0

        assert np.isclose(
            StraightLine2D((1.0, 2.0), (-2.0, 5.0)).angle(), np.pi * 0.75
        )
        assert np.isclose(
            StraightLine2D((1.0, 2.0), (-2.0, 5.0)).angle(rad=False), 135.0
        )

        assert StraightLine2D((0.0, 0.0), (-0.5, 0.0)).angle() == np.pi
        assert StraightLine2D((0.0, 0.0), (-0.5, 0.0)).angle(rad=False) == 180.0

        assert np.isclose(
            StraightLine2D((1.0, 2.0), (-2.0, -1.0)).angle(), np.pi * 1.25
        )
        assert np.isclose(
            StraightLine2D((1.0, 2.0), (-2.0, -1.0)).angle(rad=False), 225.0
        )

        assert StraightLine2D((0.0, 0.0), (0.0, -1.0)).angle() == np.pi * 1.5
        assert StraightLine2D((0.0, 0.0), (0.0, -1.0)).angle(rad=False) == 270.0

        assert np.isclose(
            StraightLine2D((1.0, 2.0), (4.0, -1.0)).angle(), np.pi * 1.75
        )
        assert np.isclose(
            StraightLine2D((1.0, 2.0), (4.0, -1.0)).angle(rad=False), 315.0
        )

        assert np.isclose(
            StraightLine2D((1.0, 2.0), (2.0, 2.0 - np.sqrt(3.0))).angle(),
            np.pi * 5.0 / 3.0,
        )
        assert np.isclose(
            StraightLine2D((1.0, 2.0), (2.0, 2.0 - np.sqrt(3.0))).angle(rad=False),
            300.0,
        )
