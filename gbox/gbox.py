"""

Geometry Box
============


Assumptions:
------------

- All the angular units are in the radians

Author: Rajesh Nakka
License: MIT


"""

from io import BytesIO
from math import inf
from itertools import product
from typing import Sequence, Union, Literal, Iterable
from functools import lru_cache
import math
import warnings


import matplotlib.pyplot as plt
import numpy as np

from .utilities import (
    rotation_matrix_2d,
    REAL_NUMBER,
    operators,
)

PI = np.pi
TWO_PI = 2 * PI
NDArray = np.ndarray
PointType = Union[list, tuple, "Point", NDArray]

# =======================================================
#          BOUNDING BOX
# =======================================================


# create a class Point, subclassing tuple
class BoundingBox:
    def __init__(self, lower_bound: list | tuple, upper_bound: list | tuple):

        assert len(lower_bound) == len(
            upper_bound
        ), "lower bound and upper bound must have same length"

        for i, j in zip(lower_bound, upper_bound):
            assert isinstance(
                i, REAL_NUMBER
            ), f"Expecting real number, got {i} in lower bound"
            assert isinstance(
                j, REAL_NUMBER
            ), f"Expecting real number, got {j} in upper bound"
            assert (
                i < j
            ), f"Expecting lower bounds to be less than upper bounds. But, {i} of lower bound is greater than {j} of upper bound"

        self.lb = np.array(lower_bound)
        self.ub = np.array(upper_bound)

    @property
    def dim(self):
        return self.lb.size

    def __eq__(self, bb_2: "BoundingBox") -> bool:

        assert isinstance(bb_2, BoundingBox), f"bounding boxes must be of same type"
        for i, j in zip(self.lb, bb_2.lb):
            assert i == j, f"Elements {i} and {j} are not equal"
        return True

    @property
    def vertices(self) -> "Points":
        return Points(list(product(*zip(self.lb, self.ub))))

    def __repr__(self) -> str:
        return f"Bounding Box:  {self.lb}, {self.ub}"

    @property
    def x(self) -> NDArray:
        return self.vertices.coordinates[:, 0]

    @property
    def y(self) -> NDArray:
        return self.vertices.coordinates[:, 1]

    @property
    def volume(self) -> float:
        return np.prod(self.side_lengths())

    def side_lengths(self) -> NDArray:
        return self.ub - self.lb

    def has_point(self, p: list | tuple) -> bool:
        assert self.dim == len(p), f"point 'p' must have dimension has bounding box"
        return all([l <= p <= u for l, p, u in zip(self.lb, p, self.ub)])

    def overlaps_with(self, bb: "BoundingBox", incl_bounds=False) -> bool:
        return all(
            lb1 <= ub2 and ub1 >= lb2 if incl_bounds else lb1 < ub2 and ub1 > lb2
            for lb1, ub1, lb2, ub2 in zip(self.lb, self.ub, bb.lb, bb.ub)
        )

    def plot(self, axs, cycle=True, **plt_opt) -> None:

        assert (
            self.dim == 2
        ), "Bounding Box can only be plotted in 2D, but {self.dim} found"

        (xl, yl), (xu, yu) = self.lb, self.ub

        x = np.array([xl, xu, xu, xl])
        y = np.array([yl, yl, yu, yu])

        if cycle:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        axs.plot(x, y, **plt_opt)


class Point(tuple):

    def __new__(cls, *coords) -> "Point":
        for coord in coords:
            assert isinstance(coord, REAL_NUMBER), "Coordinates must be real numbers"
        return super().__new__(cls, coords)

    def __repr__(self) -> str:
        return f"Point:  {tuple(self)}"

    @property
    def dim(self):
        return len(self)

    @classmethod
    def from_seq(cls, seq):
        if isinstance(seq, Point):
            return seq

        if isinstance(seq, (list, tuple, NDArray)):
            if hasattr(seq, "ndim") and seq.ndim > 1:
                warnings.warn(f"unravelling {seq.ndim}D array to a 1D array")
                seq = seq.ravel()
        else:
            raise TypeError(
                "Points must be given as NumpyArray or sequence of sequences of floats"
            )
        return cls(*seq)

    def operation(self, p, op: str) -> "Point":

        assert isinstance(
            p, (Point, REAL_NUMBER)
        ), "Only points and real numbers are supported"
        assert op in operators, f"Operation {op} is not supported"

        op = operators.get(op)

        if isinstance(p, Point):
            assert (
                p.dim == self.dim
            ), "Operations can be performed only only between a point and a point of same dimension or a float."
            return Point(*(op(i, j) for i, j in zip(self, p)))
        else:
            return Point(*(op(i, p) for i in self))

    def __add__(self, p: Union["Point", float]) -> "Point":
        return self.operation(p, "add")

    def __sub__(self, p: Union["Point", float]) -> "Point":
        return self.operation(p, "sub")

    def __mul__(self, p: Union["Point", float]) -> "Point":
        return self.operation(p, "mul")

    def __eq__(self, p: "Point") -> bool:
        return all([a == b for a, b in zip(self, p)])

    def distance_to(self, p: "Point") -> float:
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
        assert isinstance(p, Point), "other point 'p' must be of type 'Point'"
        assert self.dim == p.dim, "Points must have same dimension"
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(self, p)))

    def in_bounds(
        self,
        bounds: BoundingBox | list | tuple,
        include_bounds=False,
    ) -> bool:
        """Checks if the current point is within the bounds."""
        if isinstance(bounds, (list, tuple)):
            assert (
                len(bounds) == 2
            ), "Bounds must have length 2 if supplied as list or tuple"
            bounds = BoundingBox(*bounds)

        assert isinstance(bounds, BoundingBox), "bounds are expected to be BoundingBox"
        assert self.dim == bounds.dim, "Points and Bounds must have same dimension"

        if include_bounds:
            for l, p, u in zip(bounds.lb, self, bounds.ub):
                if not (l <= p <= u):
                    return False
        else:
            for l, p, u in zip(bounds.lb, self, bounds.ub):
                if not (l < p < u):
                    return False
        return True

    def as_array(self):
        return np.array(self)

    def as_list(self):
        return list(self)

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

    def is_close_to(self, p: "Point", eps: float = 1e-16) -> bool:
        """Checks if the current point is close to other point 'p'"""
        assert isinstance(p, Point), "other point 'p' must be of type 'Point'"
        for a, b in zip(self, p):
            if abs(a - b) > eps:
                return False
        return True


class Point2D(Point):
    def __new__(cls, x: float, y: float) -> "Point2D":
        return super().__new__(cls, x, y)

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def slope(self, q: "Point2D", eps: float = 1e-16) -> float:
        """ """
        assert isinstance(q, Point2D), "other point 'q' must be of type 'Point2D'"
        eps = eps if q.x == self.x else 0.0
        return (q.y - self.y) / (q.x - self.x + eps)

    def angle(self, p2: "Point2D", rad=True) -> float:
        assert isinstance(p2, Point2D), "other point must be of type Point2D"
        dy: float = p2.y - self.y
        dx: float = p2.x - self.x
        ang: float = np.arctan2(dy, dx)
        if ang < 0:
            ang += TWO_PI
        return ang if rad else np.rad2deg(ang)

    def transform(self, angle: float = 0.0, dx: float = 0.0, dy: float = 0.0):
        """Returns a new point transformed by rotation and translation"""
        new_x = (self.x * np.cos(angle) - self.y * np.sin(angle)) + dx
        new_y = (self.x * np.sin(angle) + self.y * np.cos(angle)) + dy
        return Point2D(new_x, new_y)


class Points:
    """Collection of **ordered** points"""

    def __init__(self, points: list | tuple | NDArray):

        # Validate inputs and convert to numpy array
        if not isinstance(points, NDArray):
            if not isinstance(points, (list, tuple)):
                raise TypeError(
                    "Points must be given as NumpyArray or sequence of sequences of floats"
                )
            points = np.array(points)
        assert (
            points.ndim == 2
        ), f"Points must form a two-dimensional array, but {points.ndim} found"
        assert isinstance(
            points, np.ndarray
        ), f"Points must be of type NumpyArray, but {type(points)} found"

        self.coordinates = points
        self.dim = self.coordinates.shape[1]
        self._cycle = False

    @classmethod
    def from_dimension_data(cls, *data: Iterable[float]) -> "Points":
        dat = np.array(data).T
        return cls(dat)

    @classmethod
    def from_points(cls, *points: Point) -> "Points":
        for a_p in points:
            assert isinstance(
                a_p, Point
            ), f"Supplied points must be of type Point, but {type(a_p)} found"

        return cls(tuple(p.as_array() for p in points))

    def __len__(self):
        return self.coordinates.shape[0]

    def __repr__(self):
        return f"Points:\n{self.coordinates}"

    def __eq__(self, p: "Points"):
        assert isinstance(p, Points), "Equality can be performed only between Points"
        return np.array_equal(self.coordinates, p.coordinates)

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
    def cycle(self, val: bool):
        assert isinstance(val, bool), "cycle must be of type bool"
        self._cycle = val


class PointsCollection:
    def __init__(self, *points: Points):
        self._data = np.array([p.coordinates for p in points])


class Points1D(Points):

    def __init__(self, points: list | tuple | NDArray, **kwargs):
        assert len(points) > 0, "Points1D must have at least one point"
        if len(points) == 1:
            points = [[i] for i in points]
        super(Points1D, self).__init__(points, **kwargs)
        assert self.dim == 1, "Constructing 'Points1D' requires one dimensional points"

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    def transform(self, dx: float = 0.0) -> "Points2D":
        """In-place transformation of the points cluster by rotation and translation"""
        if dx != 0.0:
            self.coordinates[:] = self.coordinates[:] + dx
        return self

    def reverse(self) -> "Points1D":
        """Reverses the order of points **in-place**"""
        self.coordinates[:] = np.flip(self.coordinates, axis=0)
        return self

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def plot(
        self,
        axs,
        points_plt_opt: dict = None,
    ):
        """Plots the points"""

        assert self.dim == 1, "Points Plotting is supported only for 1D and 2D points"
        _plt_opt = {"color": "blue", "marker": "o", "linestyle": "None"}

        # Plot points
        if points_plt_opt is not None:
            _plt_opt.update(points_plt_opt)

        axs.plot(
            self.x if not self.cycle else np.append(self.x, self.x[0]),
            **_plt_opt,
        )

        axs.axis("equal")


class Points2D(Points):

    def __init__(self, points: list | tuple | NDArray, **kwargs):
        super(Points2D, self).__init__(points, **kwargs)
        assert self.dim == 2, "Constructing 'Points2D' requires two dimensional points"

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> NDArray:
        return self.coordinates[:, 1]

    def transform(
        self,
        angle: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> "Points2D":
        """In-place transformation of the points cluster by rotation and translation"""
        x_ = (self.x * np.cos(angle) - self.y * np.sin(angle)) + dx
        y_ = (self.x * np.sin(angle) + self.y * np.cos(angle)) + dy
        self.coordinates[:, 0] = x_
        self.coordinates[:, 1] = y_
        return self

    def reverse(self) -> "Points2D":
        """Reverses the order of points **in-place**"""
        self.coordinates[:] = np.flip(self.coordinates, axis=0)
        return self

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def plot(
        self,
        axs,
        b_box: bool = False,
        b_box_plt_opt: dict = None,
        points_plt_opt: dict = None,
    ):
        """Plots the points"""

        assert self.dim <= 2, "Points Plotting is supported only for 1D and 2D points"
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
    # TODO implement tests
    def __init__(self, points, **kwargs):
        super(Points3D, self).__init__(points, **kwargs)
        assert (
            self.dim == 3
        ), "Constructing 'Points3D' requires three dimensional points"

    @property
    def x(self) -> NDArray:
        return self.points[:, 0]

    @property
    def y(self) -> NDArray:
        return self.points[:, 1]

    @property
    def z(self) -> NDArray:
        return self.points[:, 2]

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """ """
        raise NotImplementedError("make_periodic_tiles is not implemented")


# =============================================================================
#                           TOPOLOGICAL CURVES
# =============================================================================


class TopologicalCurve:
    """Base class for all topological curves"""

    def __init__(self):
        self.points: Points = None

    def plot(
        self, axs, b_box=False, cycle=False, b_box_plt_opt=None, points_plt_opt=None
    ):
        self.points.cycle = cycle
        self.points.plot(axs, b_box, b_box_plt_opt, points_plt_opt)


class StraightLine(TopologicalCurve):
    """Base class for all straight lines"""

    def __init__(self, p1: PointType, p2: PointType):
        super(StraightLine, self).__init__()

        self.p1 = Point.from_seq(p1)
        self.p2 = Point.from_seq(p2)

    def length(self) -> float:
        return self.p1.distance_to(self.p2)

    def equation(self):
        p, q = self.p1.as_array(), self.p2.as_array()
        direction = q - p

        def _line_eqn(t):
            return p + t * direction

        return _line_eqn


class StraightLine2D(StraightLine):
    def __init__(self, p1: PointType, p2: PointType):
        assert len(p1) == 2 and len(p2) == 2, "Expecting 2D points"
        super(StraightLine2D, self).__init__(p1, p2)

        self.p1 = Point2D.from_seq(p1)
        self.p2 = Point2D.from_seq(p2)

    def angle(self, rad=True) -> float:
        """Returns the angle of the line w.r.t positive x-axis in [0, 2 * pi]"""
        return self.p1.angle(self.p2, rad)


# =============================================================================
#                       TOPOLOGICAL SHAPES
# =============================================================================


class TopologicalClosedShape:
    """Base class for all topological shapes in n-dimensions and closed"""

    def __init__(self):
        self.boundary: Points = None

    @property
    def bounding_box(self):
        return NotImplementedError("bounding_box is not implemented")


class TopologicalClosedShape2D(TopologicalClosedShape):
    """Base class for the two-dimensional topological shapes"""

    def __init__(self):
        super(TopologicalClosedShape2D, self).__init__()
        self._area = None
        self._perimeter = None

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, a: float):
        assert a > 0, "Area must be greater than zero"
        self._area = a

    @property
    def perimeter(self):
        return self._perimeter

    @perimeter.setter
    def perimeter(self, p: float):
        assert p > 0, "Perimeter must be greater than zero"
        self._perimeter = p

    @property
    def shape_factor(self):
        return self.perimeter / math.sqrt(4.0 * math.pi * self.area)

    @property
    def eq_radius(self):
        return np.sqrt(self.area / math.pi)

    def plot(
        self,
        axs,
        b_box=False,
        b_box_plt_opt=None,
        points_plt_opt=None,
        cycle=True,
    ):
        assert (
            self.boundary.dim == 2
        ), f"Plot is supported for boundary in 2D only, but {self.boundary.dim}D points were provided"
        self.boundary.cycle = cycle
        self.boundary.plot(axs, b_box, b_box_plt_opt, points_plt_opt)


# -------------------------------------------------------------------


class Circle(TopologicalClosedShape2D):
    def __init__(
        self,
        radius,
        centre=(0.0, 0.0),
        theta_1: float = 0.0,
        theta_2: float = 2.0 * math.pi,
    ):
        assert radius > 0, "Radius must be greater than zero"
        assert theta_1 < theta_2, "Theta 1 must be less than theta 2"
        assert (
            theta_1 >= 0.0 and theta_2 <= 2.0 * np.pi
        ), "Theta 1 and theta 2 must be between 0 and 2.0 * pi"

        super(Circle, self).__init__()

        self.radius = radius
        self.centre = Point2D(*centre)
        self.boundary: Points2D = None
        self.theta_1 = theta_1
        self.theta_2 = theta_2

        self.area = math.pi * radius * radius
        self.perimeter = 2 * math.pi * radius

    def eval_boundary(self, num_points=None, arc_length=0.1, min_points=100):
        if num_points is None:
            num_points = max(
                int(math.ceil(2.0 * np.pi * self.radius / arc_length)), min_points
            )

        theta = np.linspace(self.theta_1, self.theta_2, num_points)

        xy = np.empty((num_points, 2))
        xy[:, 0] = self.radius * np.cos(theta)
        xy[:, 1] = self.radius * np.sin(theta)

        xy[:, 0] += self.centre.x
        xy[:, 1] += self.centre.y

        self.boundary = Points2D(xy)

        return self

    def contains_point(self, p: PointType, tol=1e-8) -> Literal[-1, 0, 1]:
        p = Point2D.from_seq(p)
        assert p.dim == 2, "Expecting 2D points"
        dist: float = self.centre.distance_to(p)
        if dist > self.radius + tol:
            return -1
        elif dist < self.radius - tol:
            return 1
        else:
            return 0

    def distance_to(self, c: "Circle") -> float:
        assert isinstance(c, Circle), "'c' must be of Circle type"
        return self.centre.distance_to(c.centre)

    def plot(
        self, axs, b_box=False, b_box_plt_opt=None, points_plt_opt=None, cycle=True
    ) -> None:
        if self.boundary is None:
            self.eval_boundary()
        return super().plot(axs, b_box, b_box_plt_opt, points_plt_opt, cycle)


class Circles:
    def __init__(self, *circles: Circle, initial_capacity: int = 100):
        if len(circles) == 0:
            raise ValueError("Must have at least one circle")
        for a_c in circles:
            if not isinstance(a_c, Circle):
                raise TypeError("All elements must be of type Circle")

        # Pre-allocating memory
        self.capacity: int = initial_capacity
        self._data: NDArray = np.empty((self.capacity, 3))
        self.size: int = 0
        self.boundaries: list[Points2D] = []

        self.add_circles(*circles)

    def add_circles(self, *c: Circle) -> None:

        # Check, if the pre-allocated memory is sufficient
        req_size = len(c) + self.size
        if req_size > self.capacity:
            self._grow_to(req_size)

        new_data = np.array([c.centre.as_list() + [c.radius] for c in c])

        # Adding the new circles
        self._data[self.size : len(c) + self.size] = new_data

        self.size += len(c)

    def _grow_to(self, new_size: int) -> None:
        while self.capacity < new_size:
            self.capacity = int(self.capacity * 1.5)

        new_data = np.empty((self.capacity, 3))
        new_data[: self.size] = self._data[: self.size]
        self._data = new_data

    @property
    def data(self) -> NDArray:
        return self._data[: self.size]

    @property
    def centres(self) -> NDArray:
        return self._data[: self.size, :2]

    @property
    def xc(self) -> NDArray:
        return self._data[: self.size, 0]

    @property
    def yc(self) -> NDArray:
        return self._data[: self.size, 1]

    @property
    def radii(self) -> NDArray:
        return self._data[: self.size, 2]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Circles:\n{self.size} circles"

    def __iter__(self):
        return iter(self.data)

    def clip(self, r_min, r_max):
        raise NotImplementedError("clip is not implemented")

    def evaluate_boundaries(self, num_points=None, arc_length=0.1, min_points=100):
        if num_points is None:
            num_points = max(
                int(np.ceil(TWO_PI * np.max(self.radii) / arc_length)), min_points
            )

        theta = np.linspace(0, TWO_PI, num_points)
        xy = np.empty((self.size, num_points, 2))
        xy[:, :, 0] = self.radii[:, None] * np.cos(theta)
        xy[:, :, 1] = self.radii[:, None] * np.sin(theta)

        xy[:, :, 0] = xy[:, :, 0] + self.xc[:, None]
        xy[:, :, 1] = xy[:, :, 1] + self.yc[:, None]

        self.boundaries = [Points2D(a_xy) for a_xy in xy]
        return self

    def bounding_box(self) -> BoundingBox:
        xlb = np.min(self.xc - self.radii)
        xub = np.max(self.xc + self.radii)
        ylb = np.min(self.yc - self.radii)
        yub = np.max(self.yc + self.radii)
        return BoundingBox([xlb, ylb], [xub, yub])

    def perimeters(self):
        return TWO_PI * self.radii

    def areas(self):
        return PI * self.radii * self.radii

    def distances_to(self, p: PointType) -> NDArray:
        p = Point2D.from_seq(p)
        assert p.dim == 2, "other point must be of dimension 2"
        return np.linalg.norm(self.centres - p.as_array(), axis=1)

    # TODO: If it can be implemented using __contains__ special method
    def contains_point(self, p: PointType, tol=1e-8) -> Literal[-1, 0, 1]:
        p = Point2D.from_seq(p)
        assert p.dim == 2, "other point must be of dimension 2"
        distances = self.distances_to(p)
        if np.any(distances < self.radii - tol):
            return 1
        elif np.all(distances > self.radii + tol):
            return -1
        return 0

    def plot(
        self, axs, b_box=False, b_box_plt_opt=None, points_plt_opt=None, cycle=True
    ) -> None:
        self.evaluate_boundaries()
        if b_box:
            self.bounding_box().plot(axs, **b_box_plt_opt)
        for a_boundary in self.boundaries:
            a_boundary.plot(axs, points_plt_opt=points_plt_opt)


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
        self._perimeter = math.pi * (
            (3.0 * (self.smj + self.smn))
            - math.sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
        )
        return self._perimeter

    @property
    @lru_cache(maxsize=1)
    def area(self) -> float:
        self._area = math.pi * self.smj * self.smn
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
        # TODO finding the optimal number of points based on ellipse properties

        t = np.linspace(theta_1, theta_2, num_points, endpoint=incl_theta_2)

        xy = np.empty((t.shape[0], 2))
        xy[:, 0] = self.smj * np.cos(t)
        xy[:, 1] = self.smn * np.sin(t)

        points = Points2D(xy)

        points.transform(self.mjx_angle, self.centre.x, self.centre.y)

        self.boundary = points
        self.boundary._cycle = cycle
        return self

    def contains_point(self, p: PointType, tol=1e-8) -> Literal[-1, 0, 1]:
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
        return self.smn * math.sqrt(
            1.0 - ((xi * xi) / (self.smj * self.smj - self.smn * self.smn))
        )

    def uns(self, dh=0.0) -> Circles:
        if self.aspect_ratio == 1.0:
            return Circles(Circle(self.smj, self.centre))

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

            x_i = (x_i * (m - 1.0)) + (m * e_i * math.sqrt(r_o * r_o - r_i * r_i))
        return Circles(*circles)


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
#             StraightLine(aa, (-a + r, -b), TWO_PI),
#             CircularArc(r, 1.5 * pi, TWO_PI, (a - r, r - b)),
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
#         self._perimeter = (TWO_PI * self.r_tip) + (2.0 * self.theta_c * self.r_mean)
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
#                 TWO_PI,
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
