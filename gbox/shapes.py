"""
Implicit Assumptions

All angles are supplied in radians

"""
from numpy import sin, cos, arcsin, tan, pi, sqrt, linspace, array, concatenate, stack, zeros_like, ndarray
from matplotlib.pyplot import subplots, show, savefig, close
from .points import Points, rotate
from .gbox import PLOT_OPTIONS


class Shape:
    pass


class Shape2D(Shape):
    pass


class StraightLine(Shape2D):
    def __init__(self, length: float = 2.0, ):
        super(StraightLine, self).__init__()
        self.length = length
        self.locus: Points = Points()

    def eval_locus(self, num_points: int = None, start_point: tuple[float, float] = None, angle: float = None):
        xi = linspace(0.0, self.length, num_points)
        self.locus = Points(stack((xi, zeros_like(xi)), axis=1))
        self.locus.transform(angle, start_point[0], start_point[1])
        return self


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
        self.centre = centre
        self.smj_angle = smj_angle
        self.locus: Points = Points()

    def eval_locus(
            self,
            num_points=None,
            centre=None,
            angle: float = None,
    ):
        if centre is None:
            centre = self.centre
        if angle is None:
            angle = self.smj_angle
        theta = linspace(self.theta_1, self.theta_2, num_points)
        self.locus = Points(stack((self.smj * cos(theta), self.smn * sin(theta)), axis=1))
        self.locus.transform(angle, centre[0], centre[1])
        return self


class CircularArc(EllipticalArc):
    def __init__(self, r=1.0, theta_1=0.0, theta_2=2.0 * pi, centre=(0.0, 0.0)):
        super(CircularArc, self).__init__(r, r, theta_1, theta_2, centre, 0.0)


class ClosedShape2D(Shape2D):
    """
        Closed Shape in the two-dimensional space or a plane is defined by
        the locus of points, pivot point (lying on or inside or outside) the locus and angle made by a pivot axis.
        The pivot point and axis are used for convenience and are set to `(0.0, 0.0)` and 0.0 degrees by default.
    """

    def __init__(
            self,
            pivot_point=(0.0, 0.0),
            pivot_angle=0.0,
            locus=None,
    ):
        super(ClosedShape2D, self).__init__()
        self.pivot_point = pivot_point
        self.pivot_angle = pivot_angle
        self._points: Points = Points()
        self.locus = Points() if locus is None else locus
        #
        self._area = None
        self._perimeter = None

    @property
    def locus(self):
        return self._points

    @locus.setter
    def locus(self, value):
        if isinstance(value, ndarray):
            self._points = Points(value)
        elif isinstance(value, Points):
            self._points = value
        else:
            raise TypeError(f"locus must be either 'numpy.ndarray' or 'Points' type but not {type(value)}")

    def shape_factor(self):
        assert self.area is not None and self.area > 0.0, (
            f"Area must be a positive real number but not {self.area}"
        )
        assert self.perimeter is not None and self.perimeter > 0.0, (
            f"Perimeter must be a positive real number but not {self.perimeter}"
        )
        return self.perimeter / sqrt(4.0 * pi * self.area)

    def plot(
            self,
            axis=None,
            f_path=None,
            closure=True,
            face_color=None,
            edge_color=None,
            linewidth=None,
            show_grid=None,
            hide_axes=None,
            **plt_opt
    ):
        """

        :param linewidth:
        :param hide_axes:
        :param axis: Shape is plotted on this axis and returns the same, If not provided, a figure will be created
         with default options which will be saved at `f_path` location if the `f_path` is specified.
         Otherwise, it will be displayed using matplotlib.pyplot.show() method.
        :param f_path:
        :param closure: Whether to make loop by connecting the last point with the first point.
        :param face_color: Color to fill the shape
        :param edge_color: Color
        :param show_grid:
        :param plt_opt: The plotting key-word arguments, that are taken by `matplotlib.patches.Polygon()`.
        :return:
        """
        if face_color is None:
            face_color = PLOT_OPTIONS.face_color
        if edge_color is None:
            edge_color = PLOT_OPTIONS.edge_color
        if show_grid is None:
            show_grid = PLOT_OPTIONS.show_grid
        if hide_axes is None:
            hide_axes = PLOT_OPTIONS.hide_axes
        if linewidth is None:
            linewidth = PLOT_OPTIONS.linewidth

        assert self.locus is not None, "Plotting a shape requires locus but it is set to `None` at present."
        if closure:
            self.locus.close_loop()

        def _plot(_axs):
            _axs.fill(
                self.locus.points[:, 0], self.locus.points[:, 1],
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                **plt_opt
            )
            _axs.axis('equal')
            if show_grid:
                _axs.grid()
            if hide_axes:
                _axs.axis('off')
            return _axs

        if axis is None:
            _, axis = subplots(1, 1)
            _plot(axis)
            if f_path is None:
                show()
            else:
                savefig(f_path)
                close('all')
        else:
            return _plot(axis)


class Ellipse(ClosedShape2D):
    def __init__(self,
                 smj: float = 2.0,
                 smn: float = 1.0,
                 theta_1=0.0,
                 theta_2=2.0 * pi,
                 centre=(0.0, 0.0),
                 smj_angle=0.0,
                 locus=None
                 ):
        assert smj >= smn, f"Requires semi major axis > semi minor axis but found {smj} < {smn}"
        self.smj = smj
        self.smn = smn
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        super(Ellipse, self).__init__(centre, smj_angle, locus=locus)
        #
        self.ellipse = EllipticalArc(smj, smn, theta_1, theta_2, centre, smj_angle)
        return

    @property
    def perimeter(self, method="Ramanujan"):
        if method == "Ramanujan":
            self._perimeter = pi * (
                    (3.0 * (self.smj + self.smn))
                    - sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
            )
        return self._perimeter

    @property
    def area(self):
        self._area = pi * self.smj * self.smn
        return self._area

    def eval_locus(
            self,
            centre: tuple[float, float] = None,
            smj_angle: float = None,
            num_points=100,
    ):
        """

        :param num_points: Number of points along the sector.
        :param centre: sector step length.
        :param smj_angle: sector step length.
        :return: xy
        :rtype ndarray:
        """
        #
        if centre is None:
            centre = self.pivot_point
        if smj_angle is None:
            smj_angle = self.pivot_angle
        self.locus = self.ellipse.eval_locus(num_points, centre, smj_angle).locus
        return self


class Circle(Ellipse):
    def __init__(self, radius=2.0, cent=(0.0, 0.0)):
        super().__init__(radius, radius, centre=cent)


class RegularPolygon(ClosedShape2D):
    def __init__(self,
                 num_sides: int = 3,
                 corner_radius: float = 0.15,
                 side_len: float = 1.0,
                 centre: tuple[float, float] = (0.0, 0.0),
                 pivot_angle: float = 0.0,
                 locus=None,
                 ):
        assert corner_radius >= 0.0, "Corner radius must be positive."
        assert num_sides > 2, "Number of sides should be integer and greater than 2"
        #
        self.num_sides = int(num_sides)
        self.side_len = side_len
        self.alpha = pi / self.num_sides
        self.corner_radius = corner_radius
        if centre is not None:
            self.centre = centre
        #
        if locus is None:
            locus = Points()
        super(RegularPolygon, self).__init__(centre, pivot_angle, locus, )
        # crr: corner radius ratio should lie between [0, 1]
        self.crr = (2.0 * self.corner_radius * tan(self.alpha)) / self.side_len
        self.cot_alpha = cos(self.alpha) / sin(self.alpha)
        return

    @property
    def perimeter(self):
        self._perimeter = self.num_sides * self.side_len * (1.0 - self.crr + (self.crr * self.alpha * self.cot_alpha))
        return self._perimeter

    @property
    def area(self):
        self._area = 0.25 * self.num_sides * self.side_len * self.side_len * self.cot_alpha * (
                1.0 - ((self.crr * self.crr) * (1.0 - (self.alpha * self.cot_alpha)))
        )
        return self._area

    def eval_locus(self, num_points=100, centre=None, pivot_angle=None):
        """
        Perimeter = (
            num_sides * (side_length - (2.0 * corner_radius * tan(alpha))) +
            2.0 * pi * corner_radius
        )
        :param centre:
        :param pivot_angle:
        :param num_points:
        :return:
        """
        # TODO find the optimal number of points for each line segment and circular arc
        h = self.side_len - (2.0 * self.corner_radius * tan(self.alpha))
        r_ins = 0.5 * self.side_len * self.cot_alpha
        r_cir = 0.5 * self.side_len / sin(self.alpha)
        k = r_cir - (self.corner_radius / cos(self.alpha))
        if pivot_angle is None:
            pivot_angle = self.pivot_angle
        if centre is None:
            centre = self.centre
        # For each side: a straight line + a circular arc
        loci = []
        for j in range(self.num_sides):
            theta_j = 2.0 * j * self.alpha
            edge_i = StraightLine(length=h).eval_locus(
                num_points, rotate(r_ins, -0.5 * h, theta_j, 0.0, 0.0), (0.5 * pi) + theta_j
            )
            arc_i = CircularArc(self.corner_radius, -self.alpha, self.alpha, (0.0, 0.0), ).eval_locus(num_points)
            arc_i.locus.transform(theta_j + self.alpha, k * cos(theta_j + self.alpha), k * sin(theta_j + self.alpha))
            loci.append(edge_i.locus.points[:-1, :])
            loci.append(arc_i.locus.points[:-1, :])
        self.locus = Points(concatenate(loci, axis=0))
        self.locus.transform(pivot_angle, centre[0], centre[1])
        return self


class Rectangle(ClosedShape2D):
    def __init__(
            self,
            smj=2.0,
            smn=1.0,
            rc: float = 0.0,
            centre=(0.0, 0.0),
            smj_angle=0.0,
            locus=None
    ):
        assert smj >= smn, f"Requires semi major axis > semi minor axis but found {smj} < {smn}"
        self.smj = smj
        self.smn = smn
        self.rc = rc
        super(Rectangle, self).__init__(centre, smj_angle, locus)
        return

    @property
    def perimeter(self):
        self._perimeter = 4 * (self.smj + self.smn) - (2.0 * (4.0 - pi) * self.rc)
        return self._perimeter

    @property
    def area(self):
        self._area = (4.0 * self.smj * self.smn) - ((4.0 - pi) * self.rc * self.rc)
        return self._area
        # return super(Ellipse, self).area

    def eval_locus(self, num_points: int = 10, centre=None, smj_angle=None):
        if centre is None:
            centre = self.pivot_point
        if smj_angle is None:
            smj_angle = self.pivot_angle
        a, b, r = self.smj, self.smn, self.rc
        l_1, l_2, arc = StraightLine(2.0 * (b - r)), StraightLine(2.0 * (a - r)), CircularArc(r, 0.0, pi / 2)
        loci = [
            l_1.eval_locus(num_points, (a, -b + r), pi / 2).locus.points[:-1, :],
            arc.eval_locus(num_points, (a - r, b - r), 0.0).locus.points[:-1, :],
            l_2.eval_locus(num_points, (a - r, b), pi).locus.points[:-1, :],
            arc.eval_locus(num_points, (r - a, b - r), pi / 2).locus.points[:-1, :],
            l_1.eval_locus(num_points, (-a, b - r), 1.5 * pi).locus.points[:-1, :],
            arc.eval_locus(num_points, (r - a, r - b), pi).locus.points[:-1, :],
            l_2.eval_locus(num_points, (-a + r, -b), 0.0).locus.points[:-1, :],
            arc.eval_locus(num_points, (a - r, r - b), 1.5 * pi).locus.points[:-1, :],
        ]
        self.locus = Points(concatenate(loci, axis=0))
        self.locus.transform(smj_angle, centre[0], centre[1])
        return self


class Capsule(Rectangle):
    def __init__(
            self,
            smj: float = 2.0,
            smn: float = 1.0,
            centre=(0.0, 0.0),
            smj_angle=0.0,
            locus=None
    ):
        super(Capsule, self).__init__(smj, smn, smn, centre, smj_angle, locus)


class CShape(ClosedShape2D):
    def __init__(
            self,
            r_out=2.0,
            r_in=1.0,
            theta_c: float = 0.5 * pi,
            centre=(0.0, 0.0),
            pivot_angle: float = 0.0,
            locus=None
    ):
        assert r_out >= r_in, f"Requires outer radius > inner radius but found {r_out} < {r_in}"
        self.r_in = r_in
        self.r_out = r_out
        self.r_tip = (r_out - r_in) * 0.5
        self.r_mean = (r_out + r_in) * 0.5
        self.theta_c = theta_c
        self.pivot_angle = pivot_angle
        if locus is None:
            locus = Points()
        self.centre = centre
        super(CShape, self).__init__(locus)
        return

    @property
    def perimeter(self):
        self._perimeter = (2.0 * pi * self.r_tip) + (2.0 * self.theta_c * self.r_mean)
        return self._perimeter

    @property
    def area(self):
        self._area = (pi * self.r_tip * self.r_tip) + (2.0 * self.theta_c * self.r_tip * self.r_mean)
        return self._area

    def eval_locus(
            self,
            num_points=10,
            centre=None,
            pivot_angle=None
    ):
        if centre is None:
            centre = self.centre
        if pivot_angle is None:
            pivot_angle = self.pivot_angle
        curves = [
            CircularArc(self.r_tip, pi, 2.0 * pi, ).eval_locus(num_points, (self.r_mean, 0.0), 0.0),
            CircularArc(self.r_out, 0.0, self.theta_c, ).eval_locus(num_points, (0.0, 0.0), 0.0),
            CircularArc(self.r_tip, self.theta_c, self.theta_c + pi, ).eval_locus(
                num_points, rotate(self.r_mean, 0.0, self.theta_c, 0.0, 0.0), 0.0),
            CircularArc(self.r_in, self.theta_c, 0.0).eval_locus(num_points, (0.0, 0.0), 0.0),
        ]
        self.locus = Points(concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0))
        self.locus.transform(pivot_angle, centre[0], centre[1])
        return self


class NLobeShape(ClosedShape2D):

    def __init__(self,
                 num_lobes: int = 2,
                 r_lobe: float = 1.0,
                 ld_factor: float = 0.5,
                 centre=(0.0, 0.0),
                 pivot_angle: float = 0.0,
                 locus=None
                 ):
        assert num_lobes > 1, "Number of lobes must be greater than 1"
        assert 0.0 < ld_factor < 1.0, (
            f"Invalid lobe distance factor {ld_factor} is encountered, it must be lie between 0.0 and 1.0."
        )
        super(NLobeShape, self).__init__(locus)
        #
        #
        self.num_lobes = int(num_lobes)
        self.r_lobe = r_lobe
        self.ld_factor = ld_factor
        self.alpha = pi / num_lobes
        self.pivot_angle = pivot_angle
        self.centre = centre
        #
        self.theta = arcsin(sin(self.alpha) * ((self.r_outer - r_lobe) / (2.0 * r_lobe)))
        #
        self._r_outer = None

    @property
    def r_outer(self):
        self._r_outer = self.r_lobe * (1.0 + ((1.0 + self.ld_factor) / sin(self.alpha)))
        return self._r_outer

    @property
    def perimeter(self):
        self._perimeter = 2.0 * self.num_lobes * self.r_lobe * (self.alpha + (2.0 * self.theta))
        return self._perimeter

    @property
    def area(self):
        self._area = self.num_lobes * self.r_lobe * self.r_lobe * (
                self.alpha + (2.0 * (1.0 + self.ld_factor) * sin(self.alpha + self.theta) / sin(self.alpha))
        )
        return self._area

    def eval_locus(self, num_points=100, centre=None, pivot_angle=None):
        if centre is None:
            centre = self.centre
        if pivot_angle is None:
            pivot_angle = self.pivot_angle
        r_l, r_o = self.r_lobe, self.r_outer
        beta = self.theta + self.alpha
        c_1 = (r_o - r_l, 0.0)
        c_2 = (r_o - r_l + (2.0 * r_l * cos(beta)), 2.0 * r_l * sin(beta))
        curves = []
        for j in range(self.num_lobes):
            curve_1 = CircularArc(r_l, -beta, beta).eval_locus(num_points, centre=c_1)
            curve_2 = CircularArc(r_l, -self.theta, self.theta).eval_locus(num_points)
            curve_2.locus.transform(pi + self.alpha, *c_2).reverse()
            beta_j = 2.0 * j * self.alpha
            curves.extend([curve_1.locus.transform(beta_j), curve_2.locus.transform(beta_j)])
        #
        self.locus = Points(concatenate([a_curve.points[:-1, :] for a_curve in curves], axis=0))
        self.locus.transform(pivot_angle, centre[0], centre[1])
        return self


class BoundingBox2D(ClosedShape2D):
    def __init__(self, xlb=-1.0, ylb=-1.0, xub=1.0, yub=1.0):
        super(BoundingBox2D, self).__init__()
        assert xub > xlb, f"x upper bound ({xub}) < ({xlb}) x lower bound"
        assert yub > ylb, f"x upper bound ({yub}) < ({ylb}) x lower bound"
        lx, ly = xub - xlb, yub - ylb
        self.xlb = xlb
        self.xub = xub
        self.ylb = ylb
        self.yub = yub
        self.lx = lx
        self.ly = ly
        #
        self.locus = Points(array(
            [[self.xlb, self.ylb], [self.xub, self.ylb], [self.xub, self.yub], [self.xlb, self.yub], ]
        ))

    def eval_locus(self):
        return self

    @property
    def perimeter(self):
        self._perimeter = 2.0 * (self.lx + self.ly)
        return self._perimeter

    @property
    def area(self):
        self._area = self.lx * self.ly
        return self._area


class ShapesList(list):
    def plot(self, *args, **kwargs):
        for i in range(self.__len__()):
            axs = self.__getitem__(i).eval_locus().plot(*args, **kwargs)


class ClosedShapesList(ShapesList):
    def __init__(self):
        super(ClosedShapesList, self).__init__()

    @staticmethod
    def validate_incl_data(a, n):
        assert isinstance(a, ndarray), "given inclusion data must be an numpy.ndarray"
        assert a.shape[1] == n, f"Incorrect number of columns, found {a.shape[1]} instead of {n}"
        return


class Circles(ClosedShapesList):
    def __init__(self, xyr: ndarray):
        self.validate_incl_data(xyr, 3)
        super(Circles, self).__init__()
        self.extend([Circle(r, (x, y)) for (x, y, r) in xyr])


class Capsules(ClosedShapesList):
    def __init__(self, xyt_ab):
        self.validate_incl_data(xyt_ab, 5)
        super(Capsules, self).__init__()
        self.extend([Capsule(a, b, (x, y), tht) for (x, y, tht, a, b) in xyt_ab])


class RegularPolygons(ClosedShapesList):
    def __init__(self, xyt_arn):
        self.validate_incl_data(xyt_arn, 6)
        super(RegularPolygons, self).__init__()
        self.extend([RegularPolygon(n, rc, a, (x, y), tht) for (x, y, tht, a, rc, n) in xyt_arn])


class Ellipses(ClosedShapesList):
    def __init__(self, xyt_ab):
        self.validate_incl_data(xyt_ab, 5)
        super(Ellipses, self).__init__()
        self.extend([Ellipse(a, b, centre=(x, y), smj_angle=tht) for (x, y, tht, a, b) in xyt_ab])


class Rectangles(ClosedShapesList):
    def __init__(self, xyt_abr):
        self.validate_incl_data(xyt_abr, 6)
        super(Rectangles, self).__init__()
        self.extend([Rectangle(a, b, r, (x, y), tht) for (x, y, tht, a, b, r) in xyt_abr])


class CShapes(ClosedShapesList):
    def __init__(self, xyt_ro_ri_ang):
        self.validate_incl_data(xyt_ro_ri_ang, 6)
        super(CShapes, self).__init__()
        self.extend([CShape(ro, ri, ang, (x, y), tht) for (x, y, tht, ro, ri, ang) in xyt_ro_ri_ang])


class NLobeShapes(ClosedShapesList):
    def __init__(self, xyt_abr):
        self.validate_incl_data(xyt_abr, 6)
        super(NLobeShapes, self).__init__()
        self.extend([NLobeShape(a, b, r, (x, y), tht) for (x, y, tht, a, b, r) in xyt_abr])
