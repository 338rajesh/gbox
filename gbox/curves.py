from numpy import linspace, stack, zeros_like, pi, cos, sin
from .points import Points
from .gbox import Shape2D


class StraightLine(Shape2D):
    def __init__(
            self,
            length: float = 2.0,
            start_point: tuple[float, float] = (0.0, 0.0),
            angle: float = 0.0,
    ):
        super(StraightLine, self).__init__()
        self.length = length
        self.x0, self.y0 = start_point
        self.angle = angle

    @property
    def locus(self):
        xi = linspace(0.0, self.length, self.num_locus_points)
        self._locus = Points(stack((xi, zeros_like(xi)), axis=1))
        self._locus.transform(self.angle, self.x0, self.y0)
        return self._locus


class EllipticalArc(Shape2D):
    def __init__(
            self,
            smj: float = 2.0,
            smn: float = 1.0,
            theta_1: float = 0.0,
            theta_2: float = pi / 2,
            centre=(0.0, 0.0),
            smj_angle: float = 0.0,
    ):
        super(EllipticalArc, self).__init__()
        self.smj = smj
        self.smn = smn
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.xc, self.yc = self.centre = centre
        self.smj_angle = smj_angle
        # self.locus: Points = Points()

    @property
    def locus(self):
        theta = linspace(self.theta_1, self.theta_2, self.num_locus_points)
        self._locus = Points(stack((self.smj * cos(theta), self.smn * sin(theta)), axis=1))
        self._locus.transform(self.smj_angle, self.xc, self.yc)
        return self._locus


class CircularArc(EllipticalArc):
    def __init__(self, r=1.0, theta_1=0.0, theta_2=2.0 * pi, centre=(0.0, 0.0)):
        super(CircularArc, self).__init__(r, r, theta_1, theta_2, centre, 0.0)
