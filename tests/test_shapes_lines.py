import pytest
from gbox import PointND
from gbox.shapes import (
    StraightLineND as line_nd,
    StraightLine2D as line_2d,
)
import numpy as np


@pytest.fixture
def origin_2d():
    return PointND(0.0, 0.0)


class TestStraightLine:
    def test_straight_line_1(self):

        line = line_nd([0.0, 0.0, 1.0], [1.0, 1.0, -1.0])
        assert isinstance(line, line_nd)
        assert line.p1 == (0.0, 0.0, 1.0)
        assert line.p2 == (1.0, 1.0, -1.0)
        assert line.length == pytest.approx(6.0**0.5)
        eqn = line.equation()
        assert np.array_equal(eqn(0.0), (0.0, 0.0, 1.0))
        assert np.array_equal(eqn(0.5), (0.5, 0.5, 0.0))
        assert np.array_equal(eqn(1.0), (1.0, 1.0, -1.0))

    def test_straight_line_2d(self):
        line = line_2d([1.0, 2.0], [3.0, 4.0])
        assert isinstance(line, line_nd)
        assert isinstance(line, line_2d)
        assert line.dim == 2
        with pytest.raises(ValueError):
            line_2d([1.0, 2.0, 3.0], [3.0, 4.0, 5.0])

        assert line.p1 == (1.0, 2.0)
        assert line.p2 == (3.0, 4.0)
        assert line.length == pytest.approx(8.0**0.5)
        assert np.array_equal(line.equation()(0.0), (1.0, 2.0))

    def test_straight_line_2d_angle(self, origin_2d):
        rotation_data = {
            0.0: (1.0, 0.0),
            60.0: (1.0, np.sqrt(3.0)),
            90.0: (0.0, 1.0),
            135.0: (-3.0, 3.0),
            180.0: (-0.5, 0.0),
            225.0: (-3.0, -3.0),
            270.0: (0.0, -1.0),
            315.0: (3.0, -3.0),
            300.0: (1.0, -np.sqrt(3.0)),
        }
        for (ang_deg, (x, y)) in rotation_data.items():
            for r in [True, False]:
                v = np.deg2rad(ang_deg) if r else ang_deg
                assert line_2d(origin_2d, [x, y]).angle(r) == pytest.approx(v)
