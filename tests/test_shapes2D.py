import numpy as np
from geometry_box import shapes2D


class TestEllipse:
    a1, b1 = 2.56, 2.01
    e1 = shapes2D.Ellipse(a1, b1)
    e2 = shapes2D.Ellipse(a1, a1)

    def test_ellipse_area(self):
        assert self.e1.area() == np.pi * self.a1 * self.b1
        return

    def test_shape_factor(self):
        assert abs(self.e2.shape_factor() - 1.0) < 1e-06
        return


class TestRegularPolygon:
    n1, n2, n3 = 3, 4, 13
    a1 = 2.695
    crr0, crr1, crr2, crr3, crr4 = 0.0, 0.126, 1.02, -0.96, 1.0
    alpha_1, alpha_2, alpha_3 = np.pi / n1, np.pi / n2, np.pi / n3

    def test_regular_polygon_area(self):
        assert abs(
            shapes2D.RegularPolygon(self.n1, self.crr0, side_len=self.a1, ).area() - (
                    np.sqrt(3) * 0.25 * (self.a1 ** 2))
        ) < 1e-06
        assert abs(
            shapes2D.RegularPolygon(self.n2, self.crr0, side_len=self.a1).area() - (self.a1 * self.a1)
        ) < 1e-06
        assert abs(
            shapes2D.RegularPolygon(self.n3, self.crr4, side_len=self.a1).area() -
            (np.pi * (self.a1 / (2.0 * np.tan(self.alpha_3))) ** 2)) < 1e-06  # testing if crr=1.0 leads to circle?
        return

    def test_regular_polygon_shape_factor(self):
        assert abs(
            shapes2D.RegularPolygon(self.n2, self.crr0, side_len=self.a1, ).shape_factor() - (2.0 / np.sqrt(np.pi))
        ) < 1e-06
        #
        assert abs(
            shapes2D.RegularPolygon(self.n2, self.crr4, side_len=self.a1, ).shape_factor() - 1.00
        ) < 1e-06  # testing if crr=1.0 leads to shf of 1.0
        return
