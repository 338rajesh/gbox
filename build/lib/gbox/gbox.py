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
    def vertices(self) -> "PointSet":
        return PointSet(list(product(*zip(self.lb, self.ub))))

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
        """Returns a Point from a sequence or NumpyArray of floats"""
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
        ), "Only point of 'Point' and real number type are supported"
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
        assert self.dim == p.dim, "The other point must have same dimension"
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
        assert self.dim == bounds.dim, "Mismatch in the dimension of Point and Bounds"

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


class PointSet:
    """Collection of **ordered** points"""

    def __init__(self, points: list | tuple | NDArray):

        # Validate inputs and convert to numpy array
        if not isinstance(points, NDArray):
            if not isinstance(points, (list, tuple)):
                raise TypeError(
                    "PointSet is construction requred a NumpyArray or sequence of sequences of floats"
                )
            points = np.array(points)
        assert (
            points.ndim == 2
        ), f"PointSet must form a two-dimensional array, but {points.ndim} found"
        assert isinstance(
            points, np.ndarray
        ), f"PointSet must be of type NumpyArray, but {type(points)} found"

        self.coordinates = points
        self.dim = self.coordinates.shape[1]
        self._cycle = False

    @classmethod
    def from_dimension_data(cls, *data: Iterable[float]) -> "PointSet":
        dat = np.array(data).T
        return cls(dat)

    @classmethod
    def from_points(cls, *points: Point) -> "PointSet":
        for a_p in points:
            assert isinstance(
                a_p, Point
            ), f"Supplied points must be of type Point, but {type(a_p)} found"

        return cls(tuple(p.as_array() for p in points))

    def __len__(self):
        return self.coordinates.shape[0]

    def __repr__(self):
        return f"PointSet:\n{self.coordinates}"

    def __eq__(self, p: "PointSet"):
        assert isinstance(p, PointSet), "Equality check requires two PointSets"
        return np.array_equal(self.coordinates, p.coordinates)

    def copy(self):
        return PointSet(self.coordinates.copy())

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
    def __init__(self, *points: PointSet):
        self._data = np.array([p.coordinates for p in points])


class PointSet1D(PointSet):

    def __init__(self, points: list | tuple | NDArray, **kwargs):
        assert len(points) > 0, "PointSet1D must have at least one point"
        if len(points) == 1:
            points = [[i] for i in points]
        super(PointSet1D, self).__init__(points, **kwargs)
        assert (
            self.dim == 1
        ), "Constructing 'PointSet1D' requires one dimensional points"

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    def transform(self, dx: float = 0.0) -> "PointSet2D":
        """In-place transformation of the points cluster by rotation and translation"""
        if dx != 0.0:
            self.coordinates[:] = self.coordinates[:] + dx
        return self

    def reverse(self) -> "PointSet1D":
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

        assert self.dim == 1, "PointSet Plotting is supported only for 1D and 2D points"
        _plt_opt = {"color": "blue", "marker": "o", "linestyle": "None"}

        # Plot points
        if points_plt_opt is not None:
            _plt_opt.update(points_plt_opt)

        axs.plot(
            self.x if not self.cycle else np.append(self.x, self.x[0]),
            **_plt_opt,
        )

        axs.axis("equal")


class PointSet2D(PointSet):

    def __init__(self, points: list | tuple | NDArray, **kwargs):
        super(PointSet2D, self).__init__(points, **kwargs)
        assert (
            self.dim == 2
        ), "Constructing 'PointSet2D' requires two dimensional points"

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
    ) -> "PointSet2D":
        """In-place transformation of the points cluster by rotation and translation"""
        x_ = (self.x * np.cos(angle) - self.y * np.sin(angle)) + dx
        y_ = (self.x * np.sin(angle) + self.y * np.cos(angle)) + dy
        self.coordinates[:, 0] = x_
        self.coordinates[:, 1] = y_
        return self

    def reverse(self) -> "PointSet2D":
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

        assert self.dim <= 2, "PointSet Plotting is supported only for 1D and 2D points"
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


class Points3D(PointSet):
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
        self.points: PointSet = None

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
        self.boundary: PointSet = None

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
        assert self.boundary is not None, "Boundary is not defined"
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
        self.boundary: PointSet2D = None
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

        self.boundary = PointSet2D(xy)

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


class CircleSet:
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
        self.boundaries: list[PointSet2D] = []

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

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle: float = 0.0,
        scale: float | NDArray = 1.0,
        pivot: tuple[float, float] = (0.0, 0.0),
    ) -> "CircleSet":
        """Updates the current circle set by transformation"""

        # Scaling the Radius
        if not isinstance(scale, (float, list, tuple, NDArray)):
            raise TypeError("Scale must be of type: float, list, tuple or NDArray")
        scale = np.atleast_1d(scale)
        assert scale.ndim == 1, "Scale must be a 1D array"
        if scale.size not in (1, self.size):
            raise ValueError(
                "Scale must be a float or have same length as the number of circles"
            )
        self._data[: self.size, 2] = self._data[: self.size, 2] * scale

        # Applying Rotation and Translation, if required
        if angle != 0.0:
            x = self._data[: self.size, 0] - pivot[0]
            y = self._data[: self.size, 1] - pivot[1]
            x_ = x * np.cos(angle) - y * np.sin(angle) + pivot[0]
            y_ = x * np.sin(angle) + y * np.cos(angle) + pivot[1]
            self._data[: self.size, 0] = x_
            self._data[: self.size, 1] = y_

        if dx != 0.0:
            self._data[: self.size, 0] += dx

        if dy != 0.0:
            self._data[: self.size, 1] += dy

        return self

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
        return f"CircleSet:\n{self.size} circles"

    def __iter__(self):
        return iter(self.data)

    def clip(self, r_min, r_max):
        raise NotImplementedError("clip is not implemented")

    def evaluate_boundaries(self, num_points=None, arc_length=0.1, min_points=100):
        if num_points is None:
            num_points = max(
                int(np.ceil(TWO_PI * np.max(self.radii) / arc_length)), min_points
            )

        t = np.linspace(0.0, TWO_PI, num_points)
        xy = np.empty((self.size, num_points, 2))
        xy[:, :, 0] = self.radii[:, None] * np.cos(t)
        xy[:, :, 1] = self.radii[:, None] * np.sin(t)

        xy[:, :, 0] = xy[:, :, 0] + self.xc[:, None]
        xy[:, :, 1] = xy[:, :, 1] + self.yc[:, None]

        self.boundaries = [PointSet2D(a_xy) for a_xy in xy]
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
        for (idx, a_boundary) in enumerate(self.boundaries):
            if "label" in points_plt_opt and idx > 0:
                del points_plt_opt["label"]
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

        t = np.linspace(theta_1, theta_2, num_points, endpoint=incl_theta_2)

        xy = np.empty((t.shape[0], 2))
        xy[:, 0] = self.smj * np.cos(t)
        xy[:, 1] = self.smn * np.sin(t)

        points = PointSet2D(xy)

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

            x_i = (x_i * (m - 1.0)) + (m * e_i * math.sqrt(r_o * r_o - r_i * r_i))
        circles_set = CircleSet(*circles)
        circles_set.transform(self.centre.x, self.centre.y, self.mjx_angle)
        return circles_set


class Polygon(TopologicalClosedShape2D):
    def __init__(self, vertices: NDArray = None):
        super(Polygon, self).__init__()
        self.vertices = vertices


class RegularPolygon(Polygon):
    pass


class Rectangle(TopologicalClosedShape2D):
    pass


class Capsule(TopologicalClosedShape2D):
    pass


class CShape(TopologicalClosedShape2D):
    pass


class NLobeShape(TopologicalClosedShape2D):
    pass
