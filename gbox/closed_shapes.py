"""
Implicit Assumptions

All angles are supplied in radians

"""
from numpy import arcsin, concatenate, ndarray
from numpy import sin, sqrt, cos, tan, pi

from .curves import StraightLine, EllipticalArc, CircularArc
from .gbox import ClosedShape2D, ClosedShapesList
from .points import Points, rotate
from .utils import is_ordered, assert_range, assert_positivity


class Ellipse(ClosedShape2D):
    """
    Ellipse defined its centre, orientation of semi-major axis with the positive x-axis, starting and ending points
    (defined by the parametric values theta_1 and theta_2), semi-major and semi-minor axis lengths. It has perimeter,
    area, shape factor, locus, bounding box and union of circles representation properties.

    >>> ellipse = Ellipse()
    >>> ellipse.smj  # prints semi-major axis length, a
    >>> ellipse.smn  # prints semi-minor axis length, b
    >>> ellipse.pivot_point  # prints centre of the ellipse
    >>> ellipse.pivot_angle  # prints orientation of the semi-major axis of the ellipse
    >>> ellipse.shape_factor  # prints shape factor of the ellipse

    """

    def __init__(self,
                 smj: float = 2.0,
                 smn: float = 1.0,
                 theta_1=0.0,
                 theta_2=2.0 * pi,
                 centre=(0.0, 0.0),
                 smj_angle=0.0,
                 ):
        is_ordered(smn, smj, 'Semi minor axis', 'Semi major axis')
        self.smj = smj
        self.smn = smn
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        super(Ellipse, self).__init__(centre, smj_angle)

    @property
    def perimeter(self):
        """
        Perimeter is approximated using the following Ramanujan formula

        .. math::
            p = \pi[3(a+b) - \sqrt{(3a + b)(a + 3b)}]

        """
        self._perimeter = pi * (
                (3.0 * (self.smj + self.smn))
                - sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
        )
        return self._perimeter

    @property
    def area(self):
        """
        Area is evaluated using the following formula,

        .. math::
            A = \pi a b

        """
        self._area = pi * self.smj * self.smn
        return self._area

    @property
    def locus(self):
        """
        Determines the points along the locus of the ellipse.

        .. math::
            x = a \\cos{ \\theta },  y = b \\sin{ \\theta }; \;\; \\theta \in [\\theta_1, \\theta_2]
        """
        #
        self._locus = EllipticalArc(
            self.smj,
            self.smn,
            self.theta_1,
            self.theta_2,
            self.pivot_point,
            self.pivot_angle,
        ).locus
        return self._locus

    @property
    def bounding_box(self):
        """
        Returns the coordinate-axis aligned bounds of the ellipse using the following formulae

        .. math::
            x = x_c \pm \sqrt{a^2 \cos^2 \\theta + b^2 \sin^2 \\theta}

            y = y_c \pm \sqrt{a^2 \sin^2 \\theta + b^2 \cos^2 \\theta}

        """
        k1 = sqrt((self.smj ** 2) * (cos(self.pivot_angle) ** 2) + (self.smj ** 2) * (sin(self.pivot_angle) ** 2))
        k2 = sqrt((self.smj ** 2) * (sin(self.pivot_angle) ** 2) + (self.smj ** 2) * (cos(self.pivot_angle) ** 2))
        self._b_box = self.pxc - k1, self.pyc - k2, self.pxc + k1, self.pyc + k2
        return self._b_box

    def union_of_circles(self):
        """
        Returns union of circles representation for the Ellipse
        """
        return


class Circle(Ellipse):
    """
    Inherits all the methods and properties from the `Ellipse()` using same semi-major and semi-minor axis lengths.
    """

    def __init__(self, radius=2.0, cent=(0.0, 0.0)):
        super().__init__(radius, radius, centre=cent)


class RegularPolygon(ClosedShape2D):
    """
    Regular Polygon with `n`-sides

    """

    def __init__(self,
                 num_sides: int = 3,
                 corner_radius: float = 0.15,
                 side_len: float = 1.0,
                 centre: tuple[float, float] = (0.0, 0.0),
                 pivot_angle: float = 0.0,
                 ):
        """

        :param num_sides:  int, number of sides which must be greater than 2
        :param corner_radius: float, corner radius to add fillets
        :param side_len: float, side length
        :param centre: tuple[float, float], centre
        :param pivot_angle: float, A reference angle in radians, measured from the positive x-axis with the normal
            to the first side of the polygon.

        """
        assert_positivity(corner_radius, 'Corner radius', absolute=False)
        assert_range(num_sides, 3)
        #
        self.num_sides = int(num_sides)
        self.side_len = side_len
        self.alpha = pi / self.num_sides
        self.corner_radius = corner_radius
        #
        super(RegularPolygon, self).__init__(centre, pivot_angle)
        # crr: corner radius ratio should lie between [0, 1]
        self.crr = (2.0 * self.corner_radius * tan(self.alpha)) / self.side_len
        self.cot_alpha = cos(self.alpha) / sin(self.alpha)
        return

    @property
    def perimeter(self):
        """

        :rtype: float
        """
        self._perimeter = self.num_sides * self.side_len * (1.0 - self.crr + (self.crr * self.alpha * self.cot_alpha))
        return self._perimeter

    @property
    def area(self):
        self._area = 0.25 * self.num_sides * self.side_len * self.side_len * self.cot_alpha * (
                1.0 - ((self.crr * self.crr) * (1.0 - (self.alpha * self.cot_alpha)))
        )
        return self._area

    @property
    def locus(self):
        # TODO find the optimal number of points for each line segment and circular arc
        h = self.side_len - (2.0 * self.corner_radius * tan(self.alpha))
        r_ins = 0.5 * self.side_len * self.cot_alpha
        r_cir = 0.5 * self.side_len / sin(self.alpha)
        k = r_cir - (self.corner_radius / cos(self.alpha))
        # For each side: a straight line + a circular arc
        loci = []
        for j in range(self.num_sides):
            theta_j = 2.0 * j * self.alpha
            edge_i = StraightLine(h, rotate(r_ins, -0.5 * h, theta_j, 0.0, 0.0), (0.5 * pi) + theta_j).locus
            arc_i = CircularArc(self.corner_radius, -self.alpha, self.alpha, (0.0, 0.0), ).locus.transform(
                theta_j + self.alpha, k * cos(theta_j + self.alpha), k * sin(theta_j + self.alpha)
            )
            loci.extend([edge_i, arc_i])
        self._locus = Points(concatenate([a_loci.points[:-1, :] for a_loci in loci], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class Rectangle(ClosedShape2D):
    def __init__(
            self,
            smj=2.0,
            smn=1.0,
            rc: float = 0.0,
            centre=(0.0, 0.0),
            smj_angle=0.0
    ):
        is_ordered(smn, smj, 'Semi minor axis', 'Semi major axis')
        self.smj = smj
        self.smn = smn
        self.rc = rc
        super(Rectangle, self).__init__(centre, smj_angle)
        return

    @property
    def perimeter(self):
        self._perimeter = 4 * (self.smj + self.smn) - (2.0 * (4.0 - pi) * self.rc)
        return self._perimeter

    @property
    def area(self):
        self._area = (4.0 * self.smj * self.smn) - ((4.0 - pi) * self.rc * self.rc)
        return self._area

    @property
    def locus(self):
        a, b, r = self.smj, self.smn, self.rc
        aa, bb = 2.0 * (a - r), 2.0 * (b - r)
        curves = [
            StraightLine(bb, (a, -b + r), 0.5 * pi),
            CircularArc(r, 0.0 * pi, 0.5 * pi, (a - r, b - r)),
            StraightLine(aa, (a - r, b), 1.0 * pi),
            CircularArc(r, 0.5 * pi, 1.0 * pi, (r - a, b - r)),
            StraightLine(bb, (-a, b - r), 1.5 * pi),
            CircularArc(r, 1.0 * pi, 1.5 * pi, (r - a, r - b)),
            StraightLine(aa, (-a + r, -b), 2.0 * pi),
            CircularArc(r, 1.5 * pi, 2.0 * pi, (a - r, r - b)),
        ]
        self._locus = Points(concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class Capsule(Rectangle):
    def __init__(
            self,
            smj: float = 2.0,
            smn: float = 1.0,
            centre=(0.0, 0.0),
            smj_angle=0.0,
    ):
        super(Capsule, self).__init__(smj, smn, smn, centre, smj_angle)


class CShape(ClosedShape2D):
    def __init__(
            self,
            r_out=2.0,
            r_in=1.0,
            theta_c: float = 0.5 * pi,
            centre=(0.0, 0.0),
            pivot_angle: float = 0.0,
    ):
        is_ordered(r_in, r_out, 'Inner radius', 'Outer radius')
        self.r_in = r_in
        self.r_out = r_out
        self.r_tip = (r_out - r_in) * 0.5
        self.r_mean = (r_out + r_in) * 0.5
        self.theta_c = theta_c
        self.pivot_point = centre
        self.pivot_angle = pivot_angle
        super(CShape, self).__init__()
        return

    @property
    def perimeter(self):
        self._perimeter = (2.0 * pi * self.r_tip) + (2.0 * self.theta_c * self.r_mean)
        return self._perimeter

    @property
    def area(self):
        self._area = (pi * self.r_tip * self.r_tip) + (2.0 * self.theta_c * self.r_tip * self.r_mean)
        return self._area

    @property
    def locus(self):
        c_1 = rotate(self.r_mean, 0.0, self.theta_c, 0.0, 0.0)
        curves = [
            CircularArc(self.r_tip, pi, 2.0 * pi, (self.r_mean, 0.0), ),
            CircularArc(self.r_out, 0.0, self.theta_c, (0.0, 0.0), ),
            CircularArc(self.r_tip, self.theta_c, self.theta_c + pi, c_1, ),
            CircularArc(self.r_in, self.theta_c, 0.0, (0.0, 0.0)),
        ]
        self._locus = Points(concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class NLobeShape(ClosedShape2D):

    def __init__(self,
                 num_lobes: int = 2,
                 r_lobe: float = 1.0,
                 ld_factor: float = 0.5,
                 centre=(0.0, 0.0),
                 pivot_angle: float = 0.0,
                 ):
        assert_range(num_lobes, 2, tag='Number of lobes')
        assert_range(ld_factor, 0.0, 1.0, False, 'lobe distance factor')
        self.pivot_point = centre
        self.pivot_angle = pivot_angle
        super(NLobeShape, self).__init__()
        #
        #
        self.num_lobes = int(num_lobes)
        self.r_lobe = r_lobe
        self.ld_factor = ld_factor
        self.alpha = pi / num_lobes
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

    @property
    def locus(self):
        r_l, r_o = self.r_lobe, self.r_outer
        beta = self.theta + self.alpha
        c_1 = (r_o - r_l, 0.0)
        c_2 = (r_o - r_l + (2.0 * r_l * cos(beta)), 2.0 * r_l * sin(beta))
        curves = []
        for j in range(self.num_lobes):
            # Making in a lobe along the positive x-axis
            curve_1 = CircularArc(r_l, -beta, beta, c_1).locus
            curve_2 = CircularArc(r_l, -self.theta, self.theta).locus.transform(pi + self.alpha, *c_2).reverse()
            # Rotating to the respective lobe direction
            beta_j = 2.0 * j * self.alpha
            curves.extend([curve_1.transform(beta_j), curve_2.transform(beta_j)])
        #
        self._locus = Points(concatenate([a_curve.points[:-1, :] for a_curve in curves], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class BoundingBox2D(Rectangle):
    def __init__(self, xlb=-1.0, ylb=-1.0, xub=1.0, yub=1.0):
        is_ordered(xlb, xub, "x lower bound", "x upper bound")
        is_ordered(ylb, yub, "y lower bound", "y upper bound")
        lx, ly = xub - xlb, yub - ylb
        self.xlb = xlb
        self.xub = xub
        self.ylb = ylb
        self.yub = yub
        self.xc = 0.5 * (xlb + xub)
        self.yc = 0.5 * (ylb + yub)
        self.lx = lx
        self.ly = ly
        super(BoundingBox2D, self).__init__(smj=0.5 * lx, smn=0.5 * ly, centre=(self.xc, self.yc))


# ==========================================
#           SHAPES LIST
# ==========================================


class Circles(ClosedShapesList):
    def __init__(self, xyr: ndarray):
        self.validate_incl_data(xyr, 3)
        super(Circles, self).__init__()
        self.xc, self.yc, self.r = xyr.T
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
