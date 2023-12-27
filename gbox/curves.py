from numpy import linspace, stack, zeros_like, pi, cos, sin, tan
from .points import Points
from .gbox import Shape2D


class StraightLine(Shape2D):
    """
            Line segment, defined by its length, starting point and orientation with respect to the positive x-axis.


        >>> line = StraightLine(5.0, (1.0, 1.0,), 0.25 * pi)
        >>> line.length
        5.0
        >>> line.slope
        0.9999999999999999
        >>> line.equation()
        (0.9999999999999999, -1.0, 1.1102230246251565e-16)
        >>> line.locus.points
        array([[1.        , 1.        ],
               [1.03571246, 1.03571246],
               [1.07142493, 1.07142493],
               .
               .
               [4.49982144, 4.49982144],
               [4.53553391, 4.53553391]])

        """
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

        self._slope = 0.0

    @property
    def slope(self):
        self._slope = tan(self.angle)
        return self._slope

    def equation(self):
        """ Returns a, b, c of the line equation in the form of ax + by + c = 0 """
        return self.slope, -1.0, (self.y0 - (self.slope * self.x0))

    @property
    def locus(self):
        xi = linspace(0.0, self.length, self.num_locus_points)
        self._locus = Points(stack((xi, zeros_like(xi)), axis=1))
        self._locus.transform(self.angle, self.x0, self.y0)
        return self._locus


class EllipticalArc(Shape2D):
    """

    >>> ellipse_arc = EllipticalArc(2.0, 1.0, 0.0, pi * 0.5, (2.0, 5.0), 0.4 * pi )
    >>> ellipse_arc.locus.points
    >>> ellipse_arc.locus.points  # returns locus with 100 points by default
    array([[2.61803399, 6.90211303],
           [2.60286677, 6.90677646],
           [2.58754778, 6.91095987],
           .
           .
           .
           [1.0588689 , 5.33915695],
           [1.04894348, 5.30901699]])
    One can also set the number o
    >>> ellipse_arc.num_locus_points = 6
    >>> ellipse_arc.locus.points
    array([[2.61803399, 6.90211303],
           [2.29389263, 6.9045085 ],
           [1.94098301, 6.7204774 ],
           [1.59385038, 6.36803399],
           [1.28647451, 5.88167788],
           [1.04894348, 5.30901699]])
    """
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
