import numpy as np
from scipy.integrate import quad
from functools import lru_cache

from ..base import ConicCurve, _T_ClosedShape2D
from ..base import Point2D, PointArray2D
from ..core import get_type, float_type
from ..constants import PI, TWO_PI


class EllipticalArc(ConicCurve):
    def __init__(
            self,
            smj: float_type,
            smn: float_type,
            centre: tuple[float_type, float_type] = (0.0, 0.0),
            mjx_angle: float_type = 0.0,
            theta_1: float_type = 0.0,
            theta_2: float_type = TWO_PI,
    ):
        super().__init__()

        if smn <= 0.0:
            raise ValueError("Semi-minor axis must be > zero")
        if smj < smn:
            raise ValueError("Semi-major axis must be >= semi-minor axis")

        self.smj = smj
        self.smn = smn
        self.centre = Point2D(*centre)
        self.mjx_angle = mjx_angle
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.end_point_1 = self.point(self.theta_1)
        self.end_point_2 = self.point(self.theta_2)
        self.aspect_ratio = self.smj / self.smn
        self.eccentricity = np.sqrt(1 - (1 / (self.aspect_ratio ** 2)))

    def area(self):
        return 0.5 * self.smj * self.smn * (self.theta_2 - self.theta_1)

    def perimeter(self, closure: str = 'open'):
        arc_len, _ = quad(self._arc_len_integrand, self.theta_1, self.theta_2)
        if closure == 'open':
            return arc_len
        elif closure == 'e2e':
            chord_len = self.end_point_1.distance_to(self.end_point_2)
            return arc_len + chord_len
        elif closure == 'ece':
            chord_len_1 = self.end_point_1.distance_to(self.centre)
            chord_len_2 = self.end_point_2.distance_to(self.centre)
            return arc_len + chord_len_1 + chord_len_2
        else:
            raise ValueError("Unknown closure type")

    def _arc_len_integrand(self, t):
        return np.hypot(self.smj * np.sin(t), self.smn * np.cos(t))

    def sample_points(self, num_points=None):
        if num_points is None:
            num_points = max(16, int(self._point_density * self.perimeter()))

        t = np.linspace(self.theta_1, self.theta_2, num_points)
        return self.points_array(t)

    def point(self, t) -> Point2D:
        x = self.smj * np.cos(t)
        y = self.smn * np.sin(t)
        p = Point2D(x, y)
        p.transform(self.mjx_angle, self.centre.x, self.centre.y)
        return p

    def points_array(self, t) -> PointArray2D:
        ac, bs = self.smj * np.cos(t), self.smn * np.sin(t)
        points = np.column_stack((ac, bs))
        points = PointArray2D(points, dtype=get_type('float'))
        points.transform(self.mjx_angle, self.centre.x, self.centre.y)
        return points


class Ellipse(_T_ClosedShape2D):

    point_class = Point2D

    def __init__(
            self,
            smj: float_type,
            smn: float_type,
            centre: tuple[float_type, float_type] = (0.0, 0.0),
            mjx_angle: float_type = 0.0,
    ):
        super(Ellipse, self).__init__()

        self.boundary = EllipticalArc(smj, smn, centre, mjx_angle)

    @property
    @lru_cache(maxsize=1)
    def perimeter(self) -> float_type:
        self._perimeter = self.boundary.perimeter()
        return self._perimeter

    @property
    @lru_cache(maxsize=1)
    def area(self) -> float_type:
        self._area = self.boundary.area()
        return self._area

    def __getattr__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return getattr(self.boundary, name)

    def eval_boundary(
        self, num_points=100, theta_1=0.0, theta_2=TWO_PI, cycle=True, incl_theta_2=True
    ):

        t = np.linspace(theta_1, theta_2, num_points, endpoint=incl_theta_2)

        xy = np.empty((t.shape[0], 2))
        xy[:, 0] = self.smj * np.cos(t)
        xy[:, 1] = self.smn * np.sin(t)

        points = PointSet2D(xy)

        points.transform(self.mjx_angle, self.centre.x, self.centre.y)

        self.boundary_points = points
        self.boundary_points._cycle = cycle
        return self

    def contains_point(self, p: PointType, tol=1e-8) -> typing.Literal[-1, 0, 1]:
        # Rotating back to the standrd poistion where ell align with x-axis
        p = Point2D.from_seq((p[0] - self.centre.x, p[1] - self.centre.y)).transform(
            -self.mjx_angle
        )
        val = (p.x**2 / self.smj**2) + (p.y**2 / self.smn**2)
        if val > 1.0 + tol:
            return -1.0
        elif val < 1.0 - tol:
            return 1.0
        else:
            return 0

    def r_shortest(self, xi: float) -> float:
        """Evaluates the shortest distance to the ellipse locus from a point on the major axis
        located at a distance xi from the centre of the ellipse.
        """
        return self.smn * np.sqrt(
            1.0 - ((xi * xi) / (self.smj * self.smj - self.smn * self.smn))
        )

    def plot(
        self, axs, b_box=False, b_box_plt_opt=None, points_plt_opt=None, cycle=True
    ):
        if self.boundary_points is None:
            self.eval_boundary()
        return super().plot(axs, b_box, b_box_plt_opt, points_plt_opt, cycle)

    def uns(self, dh=0.0) -> CircleSet:
        if self.aspect_ratio == 1.0:
            return CircleSet(Circle(self.smj, self.centre))

        assert dh >= 0, "dh, ie., buffer, must be greater than or equal to zero"

        ell_outer = Ellipse(self.smj + dh, self.smn + dh,
                            self.mjx_angle, self.centre)
        e_i: float = self.eccentricity
        e_o: float = ell_outer.eccentricity
        m: float = 2.0 * e_o * e_o / (e_i * e_i)

        x_max, r_min = self.smj * e_i * e_i, self.smn / self.aspect_ratio
        last_circle: Circle = Circle(r_min, (x_max, 0.0))
        x_i = -1.0 * x_max
        circles: list[Circle] = []
        while True:
            if x_i > x_max:
                circles.append(last_circle)
                break
            r_i = self.r_shortest(x_i)
            circles.append(Circle(r_i, (x_i, 0.0)))

            r_o = ell_outer.r_shortest(x_i)

            x_i = (x_i * (m - 1.0)) + (m * e_i *
                                       np.sqrt(r_o * r_o - r_i * r_i))
        circles_set = CircleSet(*circles)
        circles_set.transform(self.centre.x, self.centre.y, self.mjx_angle)
        return circles_set
