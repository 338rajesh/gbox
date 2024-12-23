"""

Geometry Box
============


Assumptions:
------------

- All the angular units are in the radians

Author: Rajesh Nakka
License: MIT


"""

from dataclasses import dataclass
from io import BytesIO
from math import inf
from itertools import product
from typing import Sequence, Union


import matplotlib.pyplot as plt
import numpy as np

from .utilities import Assert, rotation_matrix_2d

PointType = Union[list, tuple]

# =======================================================
#          BOUNDING BOX
# =======================================================

# create a class Point, subclassing tuple
@dataclass
class BoundingBox:
    lower_bound: list | tuple
    upper_bound: list | tuple

    @property
    def dim(self):
        return len(self.lower_bound)

    def __post_init__(self):
        Assert(self.lower_bound, self.upper_bound).have_equal_lenths(
            "Lower Bound and Upper Bound must have same length"
        )
        Assert(self.lower_bound).lt(
            self.upper_bound, "Lower Bound must be less than Upper Bound"
        )

    def __eq__(self, other) -> bool:
        Assert(other).of_type(BoundingBox, "Bounding Box must be of same type")
        return (
            self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )

    @property
    def vertices(self) -> "Points":
        return Points(list(product(*zip(self.lower_bound, self.upper_bound))))

    def __repr__(self) -> str:
        return f"Bounding Box:  {self.lower_bound}, {self.upper_bound}"

    @property
    def x(self):
        return self.vertices.coordinates[:, 0]

    @property
    def y(self):
        return self.vertices.coordinates[:, 1]

    @property
    def volume(self):
        return np.prod(np.array(self.upper_bound) - np.array(self.lower_bound))

    def has_point(self, point: list | tuple) -> bool:
        Assert(len(point), self.dim).equal(
            "Point must have same dimension as Bounding Box"
        )
        return all(
            [l <= p <= u for l, p, u in zip(self.lower_bound, point, self.upper_bound)]
        )

    def plot(self, axs, cycle=True, **plt_opt):
        Assert(self.dim, 2).equal("Bounding Box can only be plotted in 2D")
        (xl, yl), (xu, yu) = self.lower_bound, self.upper_bound
        x = np.array([xl, xu, xu, xl])
        y = np.array([yl, yl, yu, yu])
        if cycle:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        axs.plot(x, y, **plt_opt)


class Point(tuple):

    def _assert_compatibility(self, other):
        Assert(other).of_type(Point, "Points must be of same type")
        Assert(len(other), len(self)).equal("Points must have same length")

    def __new__(cls, *coords) -> "Point":
        Assert(*coords).of_type(float, "Coordinates must be of float type")
        return super().__new__(cls, coords)

    def __repr__(self) -> str:
        return f"Point:  {tuple(self)}"

    @property
    def dim(self):
        return len(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return ValueError("Can only compare points")
        return all([a == b for a, b in zip(self, other)])

    def __add__(self, other) -> "Point":
        self._assert_compatibility(other)
        return Point(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other) -> "Point":
        self._assert_compatibility(other)
        return Point(*(a - b for a, b in zip(self, other)))

    def distance_to(self, other) -> float:
        """Evaluates the distance between the current point and (x_2, y_2)

        Attributes
        ----------
        x_2: float
            x-coordinate
        y_2: float
            y-coordinate

        >>> Point(0.0, 0.0).distance_to(3.0, 4.0)
        5.0

        """
        self._assert_compatibility(other)
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(self, other)))

    def in_bounds(self, bounds: BoundingBox | tuple, include_bounds=False):
        if isinstance(bounds, BoundingBox):
            lb, ub = bounds.lower_bound, bounds.upper_bound
        else:
            Assert(len(bounds), 2).equal(
                "Bounds must be a tuple of lower & upper bounds"
            )
            lb, ub = bounds
            Assert(lb, self, ub).have_equal_lenths(
                "Lower, Upper and Point must have same dimension"
            )
        if include_bounds:
            return all([l <= p <= u for l, p, u in zip(lb, self, ub)])
        else:
            return all([l < p < u for l, p, u in zip(lb, self, ub)])

    def as_array(self):
        return np.array(self)

    def line_eqn(self, q: PointType):
        Assert(q).of_type(Point, "other point must be of type Point")
        p, q = np.array(self), np.array(q)
        direction = q - p

        def _line_eqn(t):
            return p + t * direction

        return _line_eqn

    def reflection(self, q: "Point", p1: PointType, p2: PointType):
        """Reflects the current point about a line connecting p1 and p2"""
        raise NotImplementedError("reflect is not implemented")
        # Assert(q).of_type(Point, "other point must be of type Point")
        # p1, p2, q = np.array(p1), np.array(p2), q.as_array()
        # d = p2 - p1
        # u = d / np.linalg.norm(d)
        # projections = p1 + np.outer((q - p1) @ u, u)
        # reflected_point = 2 * projections - q
        # return Point(*reflected_point)


class Point2D(Point):
    def __new__(cls, x: float, y: float) -> "Point2D":
        return super().__new__(cls, x, y)

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def slope(self, q, eps=1e-16):
        """ """
        Assert(q).of_type(Point2D, "other point must be of type Point2D")
        eps = eps if q.x == self.x else 0.0
        return (q.y - self.y) / (q.x - self.x + eps)


class Points:
    """Collection of **ordered** points"""

    def __init__(self, points: list | tuple | np.ndarray):

        # Validate inputs and convert to numpy array
        if not isinstance(points, np.ndarray):
            Assert(points).are_seq_of_seq(
                float,
                "Points must be given as NumpyArray or sequence of sequences of floats",
            )
            points = np.array(points)
        Assert(points.ndim, 2).equal("Points must form a two-dimensional array")

        self.coordinates = points
        self.dim = self.coordinates.shape[1]
        self._cycle = False

    @classmethod
    def from_dimension_data(cls, *data):
        dat = np.array(data).T
        return cls(dat)

    def __len__(self):
        return self.coordinates.shape[0]

    def __repr__(self):
        return f"Points:\n{self.coordinates}"

    def __eq__(self, value):
        Assert(value).of_type(Points, "Points must be of same type")
        return np.array_equal(self.coordinates, value.coordinates)

    def copy(self):
        return Points(self.coordinates.copy())

    @property
    def bounding_box(self):
        return BoundingBox(
            [self.coordinates[:, i].min() for i in range(self.dim)],
            [self.coordinates[:, i].max() for i in range(self.dim)],
        )

    def reflection(self, p1: tuple[float, float], p2: tuple[float, float]):
        """Reflects the current points about a line connecting p1 and p2"""
        return NotImplementedError("reflect is not implemented")

    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        Assert(value).of_type(bool, "cycle must be of type bool")
        self._cycle = value


class Points2D(Points):

    def __init__(self, points, **kwargs):
        super(Points2D, self).__init__(points, **kwargs)
        Assert(self.dim, 2).equal("Points must be in two dimensions")

    @property
    def x(self):
        return self.coordinates[:, 0]

    @property
    def y(self):
        return self.coordinates[:, 1]

    def transform(
        self,
        angle=0.0,
        dx=0.0,
        dy=0.0,
    ):
        """In-place transformation of the points cluster by rotation and translation"""
        if angle != 0.0 or dx != 0.0 or dy != 0.0:
            self.coordinates[:] = (
                self.coordinates @ rotation_matrix_2d(angle)
            ) + np.array([dx, dy])
        return self

    def reverse(self):
        """Reverses the order of points **in-place**"""
        self.coordinates[:] = np.flip(self.coordinates, axis=0)
        return self

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def plot(self, axs, b_box=False, b_box_plt_opt=None, points_plt_opt=None):
        """Plots the points"""

        Assert(self.dim, 2).equal("Plotting is supported only for 2D points")
        _plt_opt = {"color": "blue", "marker": "o", "linestyle": "None"}
        _b_box_line_opt = {"color": "red", "linewidth": 2}

        # Plot points
        if points_plt_opt is not None:
            _plt_opt.update(points_plt_opt)

        axs.plot(
            self.x if not self.cycle else np.append(self.x, self.x[0]),
            self.y if not self.cycle else np.append(self.y, self.y[0]),
            **_plt_opt,
        )

        # Add bounding box, if required
        if b_box:
            if b_box_plt_opt is not None:
                _b_box_line_opt.update(b_box_plt_opt)
            self.bounding_box.plot(axs, **_b_box_line_opt)

        axs.axis("equal")


class Points3D(Points):
    def __init__(self, points, **kwargs):
        super(Points3D, self).__init__(points, **kwargs)
        Assert(self.dim, 3).equal("For Point3D, Points must be in three dimensions")

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def z(self):
        return self.points[:, 2]

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """ """
        raise NotImplementedError("make_periodic_tiles is not implemented")


class TopologicalCurve:
    """Base class for all topological curves"""

    def __init__(self):
        self.points: Points = None

    def plot(
        self, axs, b_box=False, cycle=False, b_box_plt_opt=None, points_plt_opt=None
    ):
        self.points.cycle = cycle
        self.points.plot(axs, b_box, b_box_plt_opt, points_plt_opt)


class TopologicalShape:
    """Base class for all topological shapes"""

    def __init__(self):
        pass

    @property
    def bounding_box(self):
        return NotImplementedError("bounding_box is not implemented")


class TopologicalShape2D(TopologicalShape):
    """Base class for the two-dimensional topological shapes"""

    def __init__(self):
        super(TopologicalShape2D, self).__init__()

    def curve_length(self):
        return NotImplementedError("curve_length is not implemented")


class StraightLine(TopologicalCurve):
    """Base class for all straight lines"""

    def __init__(self):
        super(StraightLine, self).__init__()


class EllipticalArc(TopologicalCurve):
    def __init__(
        self,
        smj: float,
        smn: float,
        theta_1: float = 0.0,
        theta_2: float = np.pi / 2,
        mjx_angle: float = 0.0,
        centre=(0.0, 0.0),
    ):
        super(EllipticalArc, self).__init__()

        # Assertions
        Assert(smj).ge(smn, "Semi-major axis must be greater than semi-minor axis")
        Assert(smn).ge(0, "Semi-minor axis must be greater than zero")
        Assert(theta_1).lt(theta_2, "Theta 1 must be less than theta 2")
        Assert(theta_1, theta_2).between(
            -np.pi, np.pi, "Theta 1 and theta 2 must be between -pi and pi"
        )

        self.smj = smj
        self.smn = smn
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.centre = Point2D(*centre)
        self.mjx_angle = mjx_angle

    def eval_boundary(self, num_points=100):
        theta = np.linspace(self.theta_1, self.theta_2, num_points)
        x = self.smj * np.cos(theta)
        y = self.smn * np.sin(theta)
        self.points = Points2D.from_dimension_data(x, y).transform(
            self.mjx_angle, self.centre.x, self.centre.y
        )
        return self

    @property
    def aspect_ratio(self):
        return self.smj / self.smn

    @property
    def eccentricity(self):
        return np.sqrt(1 - (self.smn / self.smj) ** 2)


class CircularArc(EllipticalArc):
    def __init__(self, radius, theta_1, theta_2, centre=(0.0, 0.0)):
        super(CircularArc, self).__init__(
            radius, radius, theta_1, theta_2, centre=centre
        )
        self.radius = radius


# # ======================================================================================================================
# #                                             CLOSED SHAPES
# # ======================================================================================================================


# class ShapePlotter:
#     """
#     Plotter for various shapes
#     """

#     def __init__(
#         self,
#         locus: Points,
#         axs=None,
#         f_path=None,
#         closure=True,
#         linewidth=None,
#         show_grid=None,
#         hide_axes=None,
#         face_color=None,
#         edge_color=None,
#     ):
#         """

#         :param axs: Shape is plotted on this axs and returns the same, If not provided, a figure will be created
#          with default options which will be saved at `f_path` location if the `f_path` is specified.
#          Otherwise, it will be displayed using matplotlib.pyplot.show() method.
#         :param f_path: str, file path to save the figure
#         :param closure: Whether to make loop by connecting the last point with the first point.
#         :param face_color: str, Color to fill the shape
#         :param edge_color: str, Color of the edge
#         :param linewidth: float,
#         :param show_grid: bool, enable/disable the grid on figure
#         :param hide_axes: bool, enable/disable the axs on figure
#         :return: None

#         """
#         if show_grid is None:
#             show_grid = PLOT_OPTIONS.show_grid
#         if hide_axes is None:
#             hide_axes = PLOT_OPTIONS.hide_axes
#         if linewidth is None:
#             linewidth = PLOT_OPTIONS.linewidth
#         if face_color is None:
#             face_color = PLOT_OPTIONS.face_color
#         if edge_color is None:
#             edge_color = PLOT_OPTIONS.edge_color
#         #
#         assert (
#             locus is not None
#         ), "Plotting a shape requires locus but it is set to `None` at present."
#         if closure:
#             locus.close_loop()
#         self.locus: Points = locus
#         self.axs = axs
#         self.f_path = f_path
#         self.closure = closure
#         self.linewidth = linewidth
#         self.show_grid = show_grid
#         self.hide_axes = hide_axes
#         self.face_color = face_color
#         self.edge_color = edge_color

#     def _plot_on_axis(self, _axs, fill=True, title=None, **plt_opt):
#         if fill:
#             _axs.fill(
#                 self.locus.points[:, 0],
#                 self.locus.points[:, 1],
#                 facecolor=self.face_color,
#                 edgecolor=self.edge_color,
#                 linewidth=self.linewidth,
#                 **plt_opt,
#             )
#         else:
#             _axs.plot(
#                 self.locus.points[:, 0],
#                 self.locus.points[:, 1],
#                 color=self.edge_color,
#                 linewidth=self.linewidth,
#                 **plt_opt,
#             )

#         axis("equal")

#         if title is not None:
#             _axs.set_title(title)
#         if self.show_grid:
#             _axs.grid()
#         if self.hide_axes:
#             axis("off")
#         return _axs

#     def _plot(self, fill_plot=True, title=None, **plt_opt):

#         def _plt():
#             if fill_plot:
#                 self._plot_on_axis(self.axs, fill=True, title=title, **plt_opt)
#             else:
#                 self._plot_on_axis(self.axs, fill=False, title=title, **plt_opt)

#         if self.axs is None:
#             _, self.axs = subplots(1, 1)
#             _plt()
#             if self.f_path is None:
#                 try:
#                     show()
#                 except ValueError as e:
#                     print(f"Tried to display the figure but not working due to {e}")
#             else:
#                 savefig(self.f_path)
#                 close("all")
#         else:
#             return _plt()

#     def line_plot(self, title=None, **plt_opt):
#         """
#         Line plot of the shapes
#         """
#         self._plot(fill_plot=False, title=title, **plt_opt)

#     def fill_plot(self, title=None, **plt_opt):
#         self._plot(fill_plot=True, title=title, **plt_opt)


# class Curve2D(Shape2D):
#     """Curve in tw-dimensional space"""

#     def plot(
#         self,
#         axs=None,
#         f_path=None,
#         closure=True,
#         linewidth=None,
#         show_grid=None,
#         hide_axes=None,
#         edge_color="b",
#         title=None,
#         **plt_opt,
#     ):
#         ShapePlotter(
#             self.locus,
#             axs,
#             f_path,
#             closure,
#             linewidth,
#             show_grid,
#             hide_axes,
#             edge_color=edge_color,
#         ).line_plot(title=title, **plt_opt)


# class ClosedShape2D(Shape2D):
#     """
#     Closed Shape in the two-dimensional space or a plane is defined by
#     the locus of points, pivot point (lying on or inside or outside) the locus and angle made by a pivot axs.
#     The pivot point and axs are used for convenience and are set to `(0.0, 0.0)` and 0.0 degrees by default.
#     """

#     def __init__(
#         self,
#         pivot_point=(0.0, 0.0),
#         pivot_angle=0.0,
#     ):
#         super(ClosedShape2D, self).__init__()
#         self.pxc, self.pyc = self.pivot_point = pivot_point
#         self.pivot_angle = pivot_angle
#         #
#         self._area = 0.0
#         self._perimeter = 0.0
#         self._sf = 1.0

#     @property
#     def area(self):
#         """

#         :rtype: float

#         """
#         return self._area

#     @property
#     def perimeter(self):
#         """

#         :rtype: float

#         """
#         return self._perimeter

#     @property
#     def shape_factor(self):
#         """

#         :rtype: float

#         """
#         assert_positivity(self.area, "Area")
#         assert_positivity(self.perimeter, "Perimeter")
#         self._sf = self.perimeter / sqrt(4.0 * pi * self.area)
#         return self._sf

#     def plot(
#         self,
#         axs=None,
#         f_path=None,
#         closure=True,
#         linewidth=None,
#         show_grid=None,
#         hide_axes=None,
#         face_color=None,
#         edge_color=None,
#         title=None,
#         **plt_opt,
#     ):
#         """

#         :rtype: None

#         """
#         ShapePlotter(
#             self.locus,
#             axs,
#             f_path,
#             closure,
#             linewidth,
#             show_grid,
#             hide_axes,
#             face_color=face_color,
#             edge_color=edge_color,
#         ).fill_plot(title=title, **plt_opt)


# class ShapesList(list):
#     """
#     List of multiple shapes
#     """

#     def __init__(self):
#         super(ShapesList, self).__init__()
#         self._loci = Points()
#         self._perimeters = ()
#         self._areas = ()
#         self._shape_factors = ()

#     def plot(self, **kwargs):
#         """
#         A convenient method for plotting multiple shapes, and it takes same arguments and key-word arguments as
#         the ClosedShapes2D.plot()

#         :rtype: None

#         """
#         for i in range(self.__len__()):
#             self.__getitem__(i).plot(**kwargs)

#     @property
#     def loci(self):
#         """
#         Evaluates locus of all the shapes in the list. The first dimension of the loci refers to shapes

#         :rtype: Points

#         """
#         self._loci = Points(
#             stack(
#                 [self.__getitem__(i).locus.points for i in range(self.__len__())],
#                 axis=0,
#             )
#         )
#         return self._loci

#     @property
#     def perimeters(self):
#         """
#             Evaluates perimeters of all the shapes in the list.

#         :rtype: list[float]

#         """
#         self._perimeters = [
#             self.__getitem__(i).perimeter for i in range(self.__len__())
#         ]
#         return self._perimeters

#     @property
#     def areas(self):
#         """
#             Evaluates area of all the shapes in the list.

#         :rtype: list[float]

#         """
#         self._areas = [self.__getitem__(i).area for i in range(self.__len__())]
#         return self._areas

#     @property
#     def shape_factors(self):
#         """
#             Evaluates shape factors of all the shapes in the list.

#         :rtype: list[float]

#         """
#         self._shape_factors = [
#             self.__getitem__(i).shape_factor for i in range(self.__len__())
#         ]
#         return self._shape_factors


# class ClosedShapesList(ShapesList):
#     """List of multiple closed shapes"""

#     def __init__(self):
#         super(ClosedShapesList, self).__init__()

#     @staticmethod
#     def validate_incl_data(a, n):
#         assert isinstance(a, ndarray), "given inclusion data must be an numpy.ndarray"
#         assert (
#             a.shape[1] == n
#         ), f"Incorrect number of columns, found {a.shape[1]} instead of {n}"
#         return


# class Ellipse(ClosedShape2D):
#     """
#     Ellipse defined its centre, orientation of semi-major axs with the positive x-axs, starting and ending points
#     (defined by the parametric values theta_1 and theta_2), semi-major and semi-minor axs lengths. It has perimeter,
#     area, shape factor, locus, bounding box and union of circles representation properties.

#     >>> ellipse = Ellipse()
#     >>> ellipse.smj  # prints semi-major axs length, a
#     >>> ellipse.smn  # prints semi-minor axs length, b
#     >>> ellipse.pivot_point  # prints centre of the ellipse
#     >>> ellipse.pivot_angle  # prints orientation of the semi-major axs of the ellipse
#     >>> ellipse.shape_factor  # prints shape factor of the ellipse

#     """

#     def __init__(
#         self,
#         smj: float = 2.0,
#         smn: float = 1.0,
#         theta_1=0.0,
#         theta_2=2.0 * pi,
#         centre=(0.0, 0.0),
#         smj_angle=0.0,
#     ):
#         is_ordered(smn, smj, "Semi minor axs", "Semi major axs")
#         self.smj = smj
#         self.smn = smn
#         self.theta_1 = theta_1
#         self.theta_2 = theta_2
#         super(Ellipse, self).__init__(centre, smj_angle)
#         self._ecc = 1.0

#     @property
#     def perimeter(self):
#         """
#         Perimeter is approximated using the following Ramanujan formula

#         .. math::
#             p = \\pi[3(a+b) - \\sqrt{(3a + b)(a + 3b)}]

#         """
#         self._perimeter = pi * (
#             (3.0 * (self.smj + self.smn))
#             - sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
#         )
#         return self._perimeter

#     @property
#     def area(self):
#         """
#         Area is evaluated using the following formula,

#         .. math::
#             A = \\pi a b

#         """
#         self._area = pi * self.smj * self.smn
#         return self._area

#     @staticmethod
#     def _eval_eccentricity(_a: float, _b: float) -> float:
#         return sqrt(1 - (_b * _b) / (_a * _a))

#     @property
#     def eccentricity(self):
#         """
#         Eccentricity of the ellipse, evaluated using

#         .. math::
#             e = \\sqrt{1 - \\frac{b^2}{a^2}}

#         """
#         self._ecc = self._eval_eccentricity(self.smj, self.smn)
#         return self._ecc

#     @property
#     def locus(self):
#         """
#         Determines the points along the locus of the ellipse.

#         .. math::
#             x = a \\cos{ \\theta },  y = b \\sin{ \\theta }; \\;\\; \\theta \\in [\\theta_1, \\theta_2]
#         """
#         #
#         self._locus = EllipticalArc(
#             self.smj,
#             self.smn,
#             self.theta_1,
#             self.theta_2,
#             self.pivot_point,
#             self.pivot_angle,
#         ).locus
#         return self._locus

#     @property
#     def bounding_box(self):
#         """
#         Returns the coordinate-axs aligned bounds of the ellipse using the following formulae

#         .. math::
#             x = x_c \\pm \\sqrt{a^2 \\cos^2 \\theta + b^2 \\sin^2 \\theta}

#             y = y_c \\pm \\sqrt{a^2 \\sin^2 \\theta + b^2 \\cos^2 \\theta}

#         """
#         k1 = sqrt(
#             (self.smj**2) * (cos(self.pivot_angle) ** 2)
#             + (self.smj**2) * (sin(self.pivot_angle) ** 2)
#         )
#         k2 = sqrt(
#             (self.smj**2) * (sin(self.pivot_angle) ** 2)
#             + (self.smj**2) * (cos(self.pivot_angle) ** 2)
#         )
#         self._b_box = self.pxc - k1, self.pyc - k2, self.pxc + k1, self.pyc + k2
#         return self._b_box

#     def union_of_circles(self, buffer: float = 0.01) -> ClosedShapesList:
#         """
#         Returns union of circles representation for the Ellipse

#         :param buffer: A small thickness around the shape to indicate the buffer region.

#         :rtype: ClosedShapesList
#         """
#         assert buffer > 0.0, f"buffer must be a positive real number, but not {buffer}"
#         e, e_outer = self.eccentricity, self._eval_eccentricity(
#             self.smj + buffer, self.smn + buffer
#         )
#         zeta = e_outer / e
#         k = self.smj * e * e
#         xi = -k  # starting at -ae^2

#         def r_shortest(_xi, _a, _b):
#             return _b * sqrt(1.0 - ((_xi**2) / (_a**2 - _b**2)))

#         circles = ClosedShapesList()
#         while True:
#             if xi > k:
#                 circles.append(Circle(self.smn * self.smn / self.smj, cent=(k, 0.0)))
#                 break
#             ri = r_shortest(xi, self.smj, self.smn)
#             circles.append(Circle(ri, cent=(xi, 0.0)))
#             r_ip1 = r_shortest(xi, self.smj + buffer, self.smn + buffer)
#             xi = (xi * ((2.0 * zeta * zeta) - 1.0)) + (
#                 2.0 * e_outer * zeta * sqrt((r_ip1 * r_ip1) - (ri * ri))
#             )

#         return circles


# class Circle(Ellipse):
#     """
#     Inherits all the methods and properties from the `Ellipse()` using same semi-major and semi-minor axs lengths.
#     """

#     def __init__(self, radius=2.0, cent=(0.0, 0.0)):
#         super().__init__(radius, radius, centre=cent)


# class Polygon(ClosedShape2D):
#     def __init__(self, vert: ndarray = None):
#         super(Polygon, self).__init__()
#         self.vertices = vert
#         self._side_lengths = ()

#     @property
#     def area(self):
#         """
#         Evaluates the area of a polygon using the following formula

#         """
#         a = sum(self.vertices * roll(roll(self.vertices, 1, 0), 1, 1), axis=0)
#         self._area = 0.5 * abs(a[0] - a[1])
#         return self._area

#     @property
#     def side_lengths(self):
#         self._side_lengths = sqrt(
#             sum((self.vertices - roll(self.vertices, 1, axis=0)) ** 2, axis=1)
#         )
#         return self._side_lengths

#     @property
#     def perimeter(self):
#         self._perimeter = sum(self.side_lengths)
#         return self._perimeter

#     @property
#     def locus(self):
#         self._locus = Points(self.vertices)
#         return self._locus

#     @property
#     def bounding_box(self):
#         self._b_box = (
#             self.vertices.min(axis=0).tolist() + self.vertices.max(axis=0).tolist()
#         )
#         return self._b_box


# class RegularPolygon(ClosedShape2D):
#     """
#     Regular Polygon with `n`-sides

#     """

#     def __init__(
#         self,
#         num_sides: int = 3,
#         corner_radius: float = 0.15,
#         side_len: float = 1.0,
#         centre: tuple[float, float] = (0.0, 0.0),
#         pivot_angle: float = 0.0,
#     ):
#         """

#         :param num_sides:  int, number of sides which must be greater than 2
#         :param corner_radius: float, corner radius to add fillets
#         :param side_len: float, side length
#         :param centre: tuple[float, float], centre
#         :param pivot_angle: float, A reference angle in radians, measured from the positive x-axs with the normal
#             to the first side of the polygon.

#         """
#         assert_positivity(corner_radius, "Corner radius", absolute=False)
#         assert_range(num_sides, 3)
#         #
#         self.num_sides = int(num_sides)
#         self.side_len = side_len
#         self.alpha = pi / self.num_sides
#         self.corner_radius = corner_radius
#         #
#         super(RegularPolygon, self).__init__(centre, pivot_angle)
#         # crr: corner radius ratio should lie between [0, 1]
#         self.crr = (2.0 * self.corner_radius * tan(self.alpha)) / self.side_len
#         self.cot_alpha = cos(self.alpha) / sin(self.alpha)
#         return

#     @property
#     def perimeter(self):
#         """

#         :rtype: float
#         """
#         self._perimeter = (
#             self.num_sides
#             * self.side_len
#             * (1.0 - self.crr + (self.crr * self.alpha * self.cot_alpha))
#         )
#         return self._perimeter

#     @property
#     def area(self):
#         self._area = (
#             0.25
#             * self.num_sides
#             * self.side_len
#             * self.side_len
#             * self.cot_alpha
#             * (1.0 - ((self.crr * self.crr) * (1.0 - (self.alpha * self.cot_alpha))))
#         )
#         return self._area

#     @property
#     def locus(self):
#         # TODO find the optimal number of points for each line segment and circular arc
#         h = self.side_len - (2.0 * self.corner_radius * tan(self.alpha))
#         r_ins = 0.5 * self.side_len * self.cot_alpha
#         r_cir = 0.5 * self.side_len / sin(self.alpha)
#         k = r_cir - (self.corner_radius / cos(self.alpha))
#         # For each side: a straight line + a circular arc
#         loci = []
#         for j in range(self.num_sides):
#             theta_j = 2.0 * j * self.alpha
#             edge_i = StraightLine(
#                 h, rotate(r_ins, -0.5 * h, theta_j, 0.0, 0.0), (0.5 * pi) + theta_j
#             ).locus
#             arc_i = CircularArc(
#                 self.corner_radius,
#                 -self.alpha,
#                 self.alpha,
#                 (0.0, 0.0),
#             ).locus.transform(
#                 theta_j + self.alpha,
#                 k * cos(theta_j + self.alpha),
#                 k * sin(theta_j + self.alpha),
#             )
#             loci.extend([edge_i, arc_i])
#         self._locus = Points(
#             concatenate([a_loci.points[:-1, :] for a_loci in loci], axis=0)
#         )
#         self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
#         return self._locus


# class Rectangle(ClosedShape2D):
#     def __init__(
#         self, smj=2.0, smn=1.0, rc: float = 0.0, centre=(0.0, 0.0), smj_angle=0.0
#     ):
#         is_ordered(smn, smj, "Semi minor axs", "Semi major axs")
#         self.smj = smj
#         self.smn = smn
#         self.rc = rc
#         super(Rectangle, self).__init__(centre, smj_angle)
#         return

#     @property
#     def perimeter(self):
#         self._perimeter = 4 * (self.smj + self.smn) - (2.0 * (4.0 - pi) * self.rc)
#         return self._perimeter

#     @property
#     def area(self):
#         self._area = (4.0 * self.smj * self.smn) - ((4.0 - pi) * self.rc * self.rc)
#         return self._area

#     @property
#     def locus(self):
#         a, b, r = self.smj, self.smn, self.rc
#         aa, bb = 2.0 * (a - r), 2.0 * (b - r)
#         curves = [
#             StraightLine(bb, (a, -b + r), 0.5 * pi),
#             CircularArc(r, 0.0 * pi, 0.5 * pi, (a - r, b - r)),
#             StraightLine(aa, (a - r, b), 1.0 * pi),
#             CircularArc(r, 0.5 * pi, 1.0 * pi, (r - a, b - r)),
#             StraightLine(bb, (-a, b - r), 1.5 * pi),
#             CircularArc(r, 1.0 * pi, 1.5 * pi, (r - a, r - b)),
#             StraightLine(aa, (-a + r, -b), 2.0 * pi),
#             CircularArc(r, 1.5 * pi, 2.0 * pi, (a - r, r - b)),
#         ]
#         self._locus = Points(
#             concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0)
#         )
#         self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
#         return self._locus


# class Capsule(Rectangle):
#     def __init__(
#         self,
#         smj: float = 2.0,
#         smn: float = 1.0,
#         centre=(0.0, 0.0),
#         smj_angle=0.0,
#     ):
#         super(Capsule, self).__init__(smj, smn, smn, centre, smj_angle)


# class CShape(ClosedShape2D):
#     def __init__(
#         self,
#         r_out=2.0,
#         r_in=1.0,
#         theta_c: float = 0.5 * pi,
#         centre=(0.0, 0.0),
#         pivot_angle: float = 0.0,
#     ):
#         is_ordered(r_in, r_out, "Inner radius", "Outer radius")
#         self.r_in = r_in
#         self.r_out = r_out
#         self.r_tip = (r_out - r_in) * 0.5
#         self.r_mean = (r_out + r_in) * 0.5
#         self.theta_c = theta_c
#         self.pivot_point = centre
#         self.pivot_angle = pivot_angle
#         super(CShape, self).__init__()
#         return

#     @property
#     def perimeter(self):
#         self._perimeter = (2.0 * pi * self.r_tip) + (2.0 * self.theta_c * self.r_mean)
#         return self._perimeter

#     @property
#     def area(self):
#         self._area = (pi * self.r_tip * self.r_tip) + (
#             2.0 * self.theta_c * self.r_tip * self.r_mean
#         )
#         return self._area

#     @property
#     def locus(self):
#         c_1 = rotate(self.r_mean, 0.0, self.theta_c, 0.0, 0.0)
#         curves = [
#             CircularArc(
#                 self.r_tip,
#                 pi,
#                 2.0 * pi,
#                 (self.r_mean, 0.0),
#             ),
#             CircularArc(
#                 self.r_out,
#                 0.0,
#                 self.theta_c,
#                 (0.0, 0.0),
#             ),
#             CircularArc(
#                 self.r_tip,
#                 self.theta_c,
#                 self.theta_c + pi,
#                 c_1,
#             ),
#             CircularArc(self.r_in, self.theta_c, 0.0, (0.0, 0.0)),
#         ]
#         self._locus = Points(
#             concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0)
#         )
#         self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
#         return self._locus


# class NLobeShape(ClosedShape2D):

#     def __init__(
#         self,
#         num_lobes: int = 2,
#         r_lobe: float = 1.0,
#         ld_factor: float = 0.5,
#         centre=(0.0, 0.0),
#         pivot_angle: float = 0.0,
#     ):
#         assert_range(num_lobes, 2, tag="Number of lobes")
#         assert_range(ld_factor, 0.0, 1.0, False, "lobe distance factor")
#         self.pivot_point = centre
#         self.pivot_angle = pivot_angle
#         super(NLobeShape, self).__init__()
#         #
#         #
#         self.num_lobes = int(num_lobes)
#         self.r_lobe = r_lobe
#         self.ld_factor = ld_factor
#         self.alpha = pi / num_lobes
#         #
#         self.theta = arcsin(
#             sin(self.alpha) * ((self.r_outer - r_lobe) / (2.0 * r_lobe))
#         )
#         #
#         self._r_outer = None

#     @property
#     def r_outer(self):
#         self._r_outer = self.r_lobe * (1.0 + ((1.0 + self.ld_factor) / sin(self.alpha)))
#         return self._r_outer

#     @property
#     def perimeter(self):
#         self._perimeter = (
#             2.0 * self.num_lobes * self.r_lobe * (self.alpha + (2.0 * self.theta))
#         )
#         return self._perimeter

#     @property
#     def area(self):
#         self._area = (
#             self.num_lobes
#             * self.r_lobe
#             * self.r_lobe
#             * (
#                 self.alpha
#                 + (
#                     2.0
#                     * (1.0 + self.ld_factor)
#                     * sin(self.alpha + self.theta)
#                     / sin(self.alpha)
#                 )
#             )
#         )
#         return self._area

#     @property
#     def locus(self):
#         r_l, r_o = self.r_lobe, self.r_outer
#         beta = self.theta + self.alpha
#         c_1 = (r_o - r_l, 0.0)
#         c_2 = (r_o - r_l + (2.0 * r_l * cos(beta)), 2.0 * r_l * sin(beta))
#         curves = []
#         for j in range(self.num_lobes):
#             # Making in a lobe along the positive x-axs
#             curve_1 = CircularArc(r_l, -beta, beta, c_1).locus
#             curve_2 = CircularArc(r_l, -self.theta, self.theta).locus.transform(
#                 pi + self.alpha, *c_2
#             )
#             curve_2.reverse()
#             # Rotating to the respective lobe direction
#             beta_j = 2.0 * j * self.alpha
#             curves.extend([curve_1.transform(beta_j), curve_2.transform(beta_j)])
#         #
#         self._locus = Points(
#             concatenate([a_curve.points[:-1, :] for a_curve in curves], axis=0)
#         )
#         self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
#         return self._locus


# class Circles(ClosedShapesList):
#     def __init__(self, xyr: ndarray):
#         self.validate_incl_data(xyr, 3)
#         super(Circles, self).__init__()
#         self.xc, self.yc, self.r = xyr.T
#         self.extend([Circle(r, (x, y)) for (x, y, r) in xyr])


# class Capsules(ClosedShapesList):
#     def __init__(self, xyt_ab):
#         self.validate_incl_data(xyt_ab, 5)
#         super(Capsules, self).__init__()
#         self.extend([Capsule(a, b, (x, y), tht) for (x, y, tht, a, b) in xyt_ab])


# class RegularPolygons(ClosedShapesList):
#     def __init__(self, xyt_arn):
#         self.validate_incl_data(xyt_arn, 6)
#         super(RegularPolygons, self).__init__()
#         self.extend(
#             [RegularPolygon(n, rc, a, (x, y), tht) for (x, y, tht, a, rc, n) in xyt_arn]
#         )


# class Ellipses(ClosedShapesList):
#     def __init__(self, xyt_ab):
#         self.validate_incl_data(xyt_ab, 5)
#         super(Ellipses, self).__init__()
#         self.extend(
#             [
#                 Ellipse(a, b, centre=(x, y), smj_angle=tht)
#                 for (x, y, tht, a, b) in xyt_ab
#             ]
#         )


# class Rectangles(ClosedShapesList):
#     def __init__(self, xyt_abr):
#         self.validate_incl_data(xyt_abr, 6)
#         super(Rectangles, self).__init__()
#         self.extend(
#             [Rectangle(a, b, r, (x, y), tht) for (x, y, tht, a, b, r) in xyt_abr]
#         )


# class CShapes(ClosedShapesList):
#     def __init__(self, xyt_ro_ri_ang):
#         self.validate_incl_data(xyt_ro_ri_ang, 6)
#         super(CShapes, self).__init__()
#         self.extend(
#             [
#                 CShape(ro, ri, ang, (x, y), tht)
#                 for (x, y, tht, ro, ri, ang) in xyt_ro_ri_ang
#             ]
#         )


# class NLobeShapes(ClosedShapesList):
#     def __init__(self, xyt_abr):
#         self.validate_incl_data(xyt_abr, 6)
#         super(NLobeShapes, self).__init__()
#         self.extend(
#             [NLobeShape(a, b, r, (x, y), tht) for (x, y, tht, a, b, r) in xyt_abr]
#         )
