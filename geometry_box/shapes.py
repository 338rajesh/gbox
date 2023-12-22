"""
Implicit Assumptions

All angles are supplied in radians

"""
from numpy import (
    sin, cos, arcsin, tan,
    pi, sqrt, linspace, array,
    concatenate, stack,
    zeros_like, ndarray
)
from matplotlib.pyplot import (
    subplots, show, savefig, close
)

from .points import (
    Points,
    rotate,
)


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
        self.a = None  # area of the enclosed shape
        self.p = None  # perimeter of the enclosed shape along the locus
        self._points: Points = Points()
        self.locus = Points() if locus is None else locus

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
        assert self.a is not None and self.a > 0.0, f"Area must be a positive real number but not {self.a}"
        assert self.p is not None and self.p > 0.0, f"Perimeter must be a positive real number but not {self.p}"
        return self.p / sqrt(4.0 * pi * self.a)

    def plot(
            self, axis=None, f_path=None,
            closure=True,
            face_color='w', edge_color='k',
            grid=False,
            **plt_opt
    ):
        """

        :param axis: Shape is plotted on this axis and returns the same, If not provided, a figure will be created
         with default options which will be saved at `f_path` location if the `f_path` is specified.
         Otherwise, it will be displayed using matplotlib.pyplot.show() method.
        :param f_path:
        :param closure: Whether to make loop by connecting the last point with the first point.
        :param face_color: Color to fill the shape
        :param edge_color: Color
        :param grid:
        :param plt_opt: The plotting key-word arguments, that are taken by `matplotlib.patches.Polygon()`.
        :return:
        """

        assert self.locus is not None, "Plotting a shape requires locus but it is set to `None` at present."
        if closure:
            self.locus.close_loop()

        def _plot(_axs):
            _axs.fill(
                self.locus.points[:, 0], self.locus.points[:, 1],
                facecolor=face_color, edgecolor=edge_color, **plt_opt
            )
            _axs.axis('equal')
            _axs.grid() if grid else None
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
        self.p = self.perimeter()
        self.a = self.area()
        return

    def perimeter(self, method="Ramanujan"):
        if method == "Ramanujan":
            self.p = pi * (
                    (3.0 * (self.smj + self.smn))
                    - sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
            )
        return self.p

    def area(self):
        self.a = pi * self.smj * self.smn
        return self.a

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
        self.locus = Points()
        super(RegularPolygon, self).__init__(centre, pivot_angle, locus, )
        # crr: corner radius ratio should lie between [0, 1]
        self.crr = (2.0 * self.corner_radius * tan(self.alpha)) / self.side_len
        self.cot_alpha = cos(self.alpha) / sin(self.alpha)
        #
        self.p = self.perimeter()
        self.a = self.area()
        return

    def perimeter(self):
        return self.num_sides * self.side_len * (1.0 - self.crr + (self.crr * self.alpha * self.cot_alpha))

    def area(self):
        return 0.25 * self.num_sides * self.side_len * self.side_len * self.cot_alpha * (
                1.0 - ((self.crr * self.crr) * (1.0 - (self.alpha * self.cot_alpha)))
        )

    def eval_locus(self, pivot_angle=None, centre=None, num_points=100):
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
            arc_i = EllipticalArc(
                self.corner_radius, self.corner_radius, -self.alpha, self.alpha, (0.0, 0.0), 0.0,
            ).eval_locus(num_points)
            arc_i.locus.transform(theta_j + self.alpha, k * cos(theta_j + self.alpha), k * sin(theta_j + self.alpha))
            loci.append(edge_i.locus.points[:-1, :])
            loci.append(arc_i.locus.points[:-1, :])
        self.locus = Points(concatenate(loci, axis=0))
        self.locus.transform(pivot_angle, centre[0], centre[1])
        return self


#
# theta_j + (0.5 * pi)

class Rectangle(ClosedShape2D):
    def __init__(self, smj=2.0, smn=1.0, centre=(0.0, 0.0), smj_angle=0.0, rc: float = 0.0, locus=None):
        assert smj >= smn, f"Requires semi major axis > semi minor axis but found {smj} < {smn}"
        self.smj = smj
        self.smn = smn
        self.rc = rc
        super(Rectangle, self).__init__(centre, smj_angle, locus)
        return

    def perimeter(self):
        return 4 * (self.smj + self.smn) - (2.0 * (4.0 - pi) * self.rc)

    def area(self):
        return (4.0 * self.smj * self.smn) - ((4.0 - pi) * self.rc * self.rc)
        # return super(Ellipse, self).area

    def eval_locus(self, num_points: int = 5, centre=None, smj_angle=None):
        if centre is None:
            centre = self.pivot_point
        if smj_angle is None:
            smj_angle = self.pivot_angle
        a, b, r = self.smj, self.smn, self.rc
        l_1, l_2, arc = StraightLine(b - (2.0 * r)), StraightLine(a - (2.0 * r)), EllipticalArc(smj=r, smn=r)
        loci = [
            l_1.eval_locus(num_points, (a, -b + r), pi / 2).locus.points[:-1, :],
            arc.eval_locus(num_points, (a - r, b - r), 0.0).locus.points[:-1, :],
            l_2.eval_locus(num_points, (a - r, b), pi).locus.points[:-1, :],
            arc.eval_locus(num_points, (r - a, b - r), pi / 2).locus.points[:-1, :],
            l_1.eval_locus(num_points, (-a, b - r), 1.5 * pi).locus.points[:-1, :],
            arc.eval_locus(num_points, (r - a, r - b), pi).locus.points[:-1, :],
            l_2.eval_locus(num_points, (-a + r, -b), 0.0).locus.points[:-1, :],
            arc.eval_locus(num_points, (a - r, r - b), 1.5 * pi).locus.points[:-1, :]
        ]
        self.locus = Points(concatenate(loci, axis=0))
        self.locus.transform(smj_angle, centre[0], centre[1])
        return self


class CShape(ClosedShape2D):  # FIXME
    def __init__(self, ri=2.0, ro=1.0, theta_c: float = 0.5 * pi, cent=(0.0, 0.0), locus=None):
        assert ro >= ri, f"Requires outer radius > inner radius but found {ro} < {ri}"
        self.ri = ri
        self.ro = ro
        self.r = (ro - ri) * 0.5
        self.rm = (ro + ri) * 0.5
        self.theta_c = theta_c
        self.locus = locus
        self.cent = cent
        super(CShape, self).__init__(locus)
        return

    def perimeter(self):
        return (2.0 * pi * self.r) + (2.0 * self.theta_c * self.rm)

    def area(self):
        return (pi * self.r * self.r) + (2.0 * self.theta_c * self.r * self.rm)

    def eval_locus(self):
        return


class NLobeShape(ClosedShape2D):  # FIXME

    def __init__(self,
                 num_lobes: int,
                 ldf: float,
                 eq_radius: float,
                 # lobe_radius: float = None,
                 # outer_radius: float = None,
                 locus=None
                 ):
        super(NLobeShape, self).__init__(locus)
        #
        assert 0.0 < ldf < 1.0, f"Invalid lobe distance factor {ldf} is encountered, it must be in (0.0, 1.0)"
        #
        self.num_lobes = num_lobes
        self.eq_radius = eq_radius
        self.ldf = ldf
        self.lobe_radius = None
        self.outer_radius = None
        self.alpha = pi / self.num_lobes
        self.theta = arcsin(0.5 * (1.0 + self.ldf))

    def set_lobe_radius(self, l_df: float = None, eq_radius: float = None):
        if l_df is None:
            l_df = self.ldf
        if eq_radius is None:
            eq_radius = self.eq_radius
        #
        k1 = self.alpha * sin(self.alpha)
        self.lobe_radius = eq_radius * sqrt(k1 / (k1 + (2.0 * (1.0 + l_df) * sin(self.alpha + self.theta))))
        return self

    def set_outer_radius(self):
        if self.lobe_radius is None:
            self.set_lobe_radius()
        self.outer_radius = ((self.ldf + 1.0 + sin(self.alpha)) / sin(self.alpha)) * self.lobe_radius
        return self
        # the present implementation assumes that ldf and eq_radius are known!

    def perimeter(self):
        if self.lobe_radius is None:
            self.set_lobe_radius()
        return 2.0 * self.num_lobes * self.lobe_radius * (self.alpha + (2.0 * self.theta))

    def area(self):
        if self.lobe_radius is None:
            self.set_lobe_radius()
        return self.num_lobes * self.lobe_radius * self.lobe_radius * (
                self.alpha + (2.0 * (1.0 + self.ldf) * sin(self.alpha + self.theta) / sin(self.alpha))
        )


class BoundingBox2D(ClosedShape2D):
    def __init__(self, xlb, ylb, xub, yub):
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
        self.p = self.perimeter()
        self.a = self.area()
        #
        self.locus = Points(array(
            [[self.xlb, self.ylb], [self.xub, self.ylb], [self.xub, self.yub], [self.xlb, self.yub], ]
        ))

    def perimeter(self):
        return 2.0 * (self.lx + self.ly)

    def area(self):
        return self.lx * self.ly

        # assert len(bbox) == 4, "Number of bounds must be exactly 4"
        # self.bbox = bbox
        # self.dim: int = 2
        # self.xlb: float = self.bbox[0]
        # self.ylb: float = self.bbox[1]
        # self.xub: float = self.bbox[2]
        # self.yub: float = self.bbox[3]
        # assert self.xub >= self.xlb, f"x upper bound ({self.xub}) < ({self.xlb}) x lower bound"
        # assert self.yub >= self.ylb, f"y upper bound ({self.yub}) < ({self.ylb}) y lower bound"
        # self.lx: float = self.xub - self.xlb
        # self.ly: float = self.yub - self.ylb
        # self.perimeter: float = 2.0 * (self.lx + self.ly)
        # self.area: float = self.lx * self.ly
        # self.a = self.area


class BoundingBox:
    def __init__(self, *bbox: float):
        assert len(bbox) in (4, 6), "Length of the bounding box must be either 4 (for 2D) or 6 (for 3D)"
        self.bbox = bbox
        if len(self.bbox) == 4:
            self.dim: int = 2
            self.xlb: float = self.bbox[0]
            self.ylb: float = self.bbox[1]
            self.xub: float = self.bbox[2]
            self.yub: float = self.bbox[3]
            assert self.xub > self.xlb, f"x upper bound ({self.xub}) > ({self.xlb}) x lower bound"
            assert self.yub > self.ylb, f"y upper bound ({self.yub}) > ({self.ylb}) y lower bound"
            self.lx: float = self.xub - self.xlb
            self.ly: float = self.yub - self.ylb
            self.perimeter: float = 2.0 * (self.lx + self.ly)
            self.area: float = self.lx * self.ly
            self.domain: float = self.area

        elif len(self.bbox) == 6:
            self.dim: int = 3
            self.xlb: float = self.bbox[0]
            self.ylb: float = self.bbox[1]
            self.zlb: float = self.bbox[2]
            self.xub: float = self.bbox[3]
            self.yub: float = self.bbox[4]
            self.zub: float = self.bbox[5]
            assert self.xub > self.xlb, f"x upper bound ({self.xub}) > ({self.xlb}) x lower bound"
            assert self.yub > self.ylb, f"y upper bound ({self.yub}) > ({self.ylb}) y lower bound"
            assert self.zub > self.zlb, f"z upper bound ({self.zub}) > ({self.zlb}) z lower bound"
            self.lx: float = self.xub - self.xlb
            self.ly: float = self.yub - self.ylb
            self.lz: float = self.zub - self.zlb
            self.surface_area: float = 2.0 * (self.lx * self.ly + self.ly * self.lz + self.lz * self.lx)
            self.volume: float = self.lx * self.ly * self.lz
            self.domain: float = self.volume
        else:
            raise ValueError(f"The length of the bounding box can be either 4 or 6 but not {len(self.bbox)}")


"""
 # def scale(self, *scaling_factors: float):
    #     num_scaling_factors: int = len(scaling_factors)
    #     assert num_scaling_factors in (1, 4, 6), "number of scaling factors must be 1 or 4 or 6"
    #     if num_scaling_factors == 1:
    #
    #     return

 def make_sector(
         self,
         num_sec_points=100,
         ds=None,
 ):
     xy = array([]).reshape(0, 2)
     alpha = pi / self.num_sides
     #
     num_sec_points = get_num_sector_points(num_sec_points, ds, self.p)
     r_inscribed = 0.5 * self.side_len / tan(alpha)
     h = (r_inscribed - self.cr) / cos(alpha)
     for i in range(int(self.num_sides)):
         theta_vertex = self.axs_angle + (2.0 * i * alpha)
         theta_sector = lin space(
             start=theta_vertex - alpha, stop=theta_vertex + alpha, num=num_sec_points)
         xx_yy = [self.centre[0] + (h * cos(theta_vertex)), self.centre[1] + (h * sin(theta_vertex))] + (
                 self.cr * column_stack([cos(theta_sector), sin(theta_sector)]))
         xy = concatenate((xy, xx_yy), axis=0)
     #
     self.locus = xy
     return self

        # if self.ldf is None and self.lobe_radius is None and self.outer_radius is None:
        #     raise ValueError("At least two of three must be supplied.")
        # else:
        #     if self.ldf is None:
        #         self.ldf = ((self.outer_radius / self.lobe_radius) - 1.0) * sin(self.alpha) - 1.0
        #     else:
        #         k = (self.ldf + 1.0 + sin(self.alpha)) / sin(self.alpha)
        #         if self.outer_radius is None:
        #             self.outer_radius = self.lobe_radius * k
        #         elif self.lobe_radius is None:
        #             self.lobe_radius = self.outer_radius / k
        #

"""
