import numpy as np
import geometry_box as gb
import unittest


class TestEllipse(unittest.TestCase):
    a1, b1 = 2.56, 2.01
    e1 = gb.Ellipse(a1, b1)
    e2 = gb.Ellipse(a1, a1)

    def test_ellipse_area(self):
        assert self.e1.area == np.pi * self.a1 * self.b1

    def test_shape_factor(self):
        assert abs(self.e2.shape_factor() - 1.0) < 1e-06


class TestRegularPolygon(unittest.TestCase):
    n1, n2, n3 = 3, 4, 13
    a1 = 2.695
    crr0, crr1, crr2, crr3, crr4 = 0.0, 0.126, 1.02, -0.96, 1.0
    alpha_1, alpha_2, alpha_3 = np.pi / n1, np.pi / n2, np.pi / n3

    def test_regular_polygon_area_1(self):
        assert abs(
            gb.RegularPolygon(self.n1, self.crr0, side_len=self.a1, ).area - (
                    np.sqrt(3 / 16) * (self.a1 ** 2))
        ) < 1e-06

    def test_regular_polygon_area_2(self):
        assert abs(
            gb.RegularPolygon(self.n2, self.crr0, side_len=self.a1).area - (self.a1 * self.a1)
        ) < 1e-06

    def test_regular_polygon_area_3(self):
        rc = (self.a1 * self.crr4) / (2.0 * np.tan(np.pi / self.n3))
        assert abs(
            gb.RegularPolygon(self.n3, rc, side_len=self.a1).area -
            (np.pi * (self.a1 / (2.0 * np.tan(self.alpha_3))) ** 2)) < 1e-06  # testing if crr=1.0 leads to circle?

    def test_regular_polygon_shape_factor_1(self):
        assert abs(
            gb.RegularPolygon(self.n2, self.crr0, side_len=self.a1, ).shape_factor() - (2.0 / np.sqrt(np.pi))
        ) < 1e-06
        #

    def test_regular_polygon_shape_factor_2(self):
        # testing if crr=1.0 leads to shf of 1.0
        n, a = 3, 5
        rc = a / (2.0 * np.tan(np.pi / n))  # this ensures that the reg-polygon is a circle
        assert abs(gb.RegularPolygon(n, rc, a, ).shape_factor() - 1.00) < 1e-06


if __name__ == '__main__':
    unittest.main()
