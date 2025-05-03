from .. import (
    Point,
    TopologicalCurve,
    PointND,
    Point2D,
)
from ..base import PointType


class ConicCurve(TopologicalCurve):
    pass


class EllipticalArc(ConicCurve):
    pass



class Ellipse(TopologicalClosedShape2D):
    def __init__(self, smj, smn, mjx_angle=0.0, centre=(0.0, 0.0)):
        super(Ellipse, self).__init__()

        # Assertions
        assert smj >= smn, "Semi-major axis must be >= semi-minor axis"
        assert smn >= 0, "Semi-minor axis must be >= zero"

        self.smj = smj
        self.smn = smn
        self.mjx_angle = mjx_angle
        self.centre = Point2D(*centre)

    @property
    @lru_cache(maxsize=1)
    def perimeter(self) -> float:
        self._perimeter = PI * (
            (3.0 * (self.smj + self.smn))
            - np.sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
        )
        return self._perimeter

    @property
    @lru_cache(maxsize=1)
    def area(self) -> float:
        self._area = PI * self.smj * self.smn
        return self._area

    @property
    def aspect_ratio(self):
        return self.smj / self.smn

    @property
    def eccentricity(self) -> float:
        ratio = self.smn / self.smj
        return np.sqrt(1 - (ratio * ratio))

    def eval_boundary(
        self, num_points=100, theta_1=0.0, theta_2=TWO_PI, cycle=True, incl_theta_2=True
    ):

        t = np.linspace(theta_1, theta_2, num_points, endpoint=incl_theta_2)

        xy = np.empty((t.shape[0], 2))
        xy[:, 0] = self.smj * np.cos(t)
        xy[:, 1] = self.smn * np.sin(t)

        points = PointSet2D(xy)

        points.transform(self.mjx_angle, self.centre.x, self.centre.y)

        self.boundary = points
        self.boundary._cycle = cycle
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
        if self.boundary is None:
            self.eval_boundary()
        return super().plot(axs, b_box, b_box_plt_opt, points_plt_opt, cycle)

    def uns(self, dh=0.0) -> CircleSet:
        if self.aspect_ratio == 1.0:
            return CircleSet(Circle(self.smj, self.centre))

        assert dh >= 0, "dh, ie., buffer, must be greater than or equal to zero"

        ell_outer = Ellipse(self.smj + dh, self.smn + dh, self.mjx_angle, self.centre)
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

            x_i = (x_i * (m - 1.0)) + (m * e_i * np.sqrt(r_o * r_o - r_i * r_i))
        circles_set = CircleSet(*circles)
        circles_set.transform(self.centre.x, self.centre.y, self.mjx_angle)
        return circles_set


class ParabolicArc(ConicCurve):
    pass


class HyperbolicArc(ConicCurve):
    pass
