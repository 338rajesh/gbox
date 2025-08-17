from collections.abc import Iterable, Sequence
from itertools import product
from typing import Union, Tuple, Literal

import numpy as np
from matplotlib.patches import Patch, Rectangle

from .utils import PlotMixin

DEFAULT_FLOAT = np.float32
FloatType = Union[float, DEFAULT_FLOAT]

PI = DEFAULT_FLOAT(np.pi)
TOLERANCE = np.finfo(DEFAULT_FLOAT).eps

_DEFAULT_POINT_PLOT_OPTIONS = {
    "color": "blue",
    "linestyle": "None",
    "marker": "o",
    "markersize": 5,
}
_DEFAULT_LINE_PLOT_OPTIONS = {
    "color": "red",
    "linewidth": 2,
    "linestyle": "-",
    "marker": "None",
}


# class GboxBaseModel()

# ============================================================================
#                           POINT ND CLASS
# ============================================================================
# region PointND


class PointND:
    """Point class for representing a point in N-dimensional space"""

    __slots__ = ("coor",)

    def __init__(self, *coords: FloatType) -> None:
        """Constructs a point from the given coordinates"""
        self._validate_coords(coords)
        self.coor = np.array(coords, dtype=DEFAULT_FLOAT)

    # ============================
    #       CORE METHODS
    # ============================
    @staticmethod
    def _validate_coords(coords: Sequence[FloatType]) -> None:
        """Validates the coordinates of the point"""
        if not coords:
            raise ValueError("Point must have at least one coordinate")

        allowable_types = (int, float, type(DEFAULT_FLOAT(0.0)))
        if not all(isinstance(c, allowable_types) for c in coords):
            raise TypeError(
                f"All coordinates must be one of {allowable_types}, "
                f"but got {[type(c) for c in coords]}",
            )

    @classmethod
    def _from_(cls, p: Union["PointND", Sequence[float]]):
        if isinstance(p, cls):
            return p
        if isinstance(p, Sequence):
            return cls(*p)
        raise TypeError(
            f"Cannot convert {p.__class__.__name__} to {cls.__name__}. "
            f"Expected {cls.__name__} or sequence of coordinates.",
        )

    # ============================
    #       MAGIC METHODS
    # ============================
    def __len__(self) -> int:
        """Returns the number of coordinates in the point"""
        return len(self.coor)

    def __getitem__(self, idx: int) -> DEFAULT_FLOAT:
        """Returns the coordinate at the given index"""
        return self.coor[idx]

    def __iter__(self) -> Iterable[float]:
        """Returns an iterator over the coordinates of the point"""
        return iter(self.coor)

    def __array__(self, dtype=None, copy=True):
        """Returns the coordinates of the point as a numpy array"""
        arr = np.array(self.coor, dtype=dtype, copy=copy)
        return arr

    def __eq__(self, q) -> bool:
        """Checks if two points are equal"""
        try:
            q = self._from_(q)
        except TypeError:
            raise TypeError(
                f"Cannot compare {type(q)} with {self.__class__.__name__}",
            )
        return np.array_equal(self.coor, q.coor)

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}({', '.join(map(str, self.coor))})"

    # =================================
    #       POINT PROPERTIES
    # =================================
    @property
    def dim(self) -> int:
        """Returns the dimension of the point"""
        return len(self.coor)

    # =================================
    #       GEOMETRIC PROPERTIES
    # =================================
    def distance_to(self, q: Union["PointND", Sequence]) -> DEFAULT_FLOAT:
        q = self._from_(q)
        return np.linalg.norm(self.coor - q.coor).astype(DEFAULT_FLOAT)

    def in_bounds(
        self,
        lower_bound: Union["PointND", Sequence[float]],
        upper_bound: Union["PointND", Sequence[float]],
    ) -> bool:
        lower_bound = self._from_(lower_bound)
        upper_bound = self._from_(upper_bound)

        if not (self.dim == lower_bound.dim == upper_bound.dim):
            raise ValueError(
                "Mismatch in the dimension of Point and Bounds; "
                f"point {self.dim}, "
                f"Lower bound: {lower_bound.dim}, "
                f"Upper bound: {upper_bound.dim}",
            )

        return bool(
            np.all(
                (self.coor >= lower_bound.coor)
                & (self.coor <= upper_bound.coor),
            ),
        )

    def reflection(self, q: "PointND", p1: "PointND", p2: "PointND"):
        """Reflects the current point about a line connecting p1 and p2"""
        raise NotImplementedError("reflect is not implemented")
        # Assert(q).of_type(Point, "other point must be of type Point")
        # p1, p2, q = np.array(p1), np.array(p2), q.as_array()
        # d = p2 - p1
        # u = d / np.linalg.norm(d)
        # projections = p1 + np.outer((q - p1) @ u, u)
        # reflected_point = 2 * projections - q
        # return Point(*reflected_point)

    def is_close_to(
        self,
        q: Union["PointND", Sequence],
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Checks if the current point is close to other point 'p'

        Parameters
        ----------
        q : "PointND"
            The other point to be compared
        rtol : float
            Relative tolerance, defaults to 1e-5
        atol : float
            Absolute tolerance, defaults to 1e-8

        Returns
        -------
        bool
            True if the current point is close to other point 'p',
            False otherwise.

        Examples
        --------
        >>> PointND(1.0, 2.0).is_close_to((1.0 + 1e-08, 2.0 + 1e-07))
        True
        >>> PointND(1.0, 2.0).is_close_to((3.0 + 1e-08, 4.0 + 1e-07))
        False

        """
        q = self._from_(q)
        return np.allclose(self.coor, q.coor, rtol=rtol, atol=atol)


# endregion PointND
# region Point1D


class Point1D(PointND):
    __slots__ = ()

    def __init__(self, x: float) -> None:
        super().__init__(x)

    @property
    def x(self) -> DEFAULT_FLOAT:
        return self.coor[0]

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}(x={self.x})"


# region Point2D


class Point2D(PointND):
    __slots__ = ()  # No additional attributes for 2D points

    def __init__(self, x: FloatType, y: FloatType) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}(x={self.x}, y={self.y})"

    @property
    def x(self) -> DEFAULT_FLOAT:
        return self.coor[0]

    @property
    def y(self) -> DEFAULT_FLOAT:
        return self.coor[1]

    def slope(
        self,
        q: Union["Point2D", Sequence[float]],
        eps: float = 1e-06,
    ) -> DEFAULT_FLOAT:
        """Returns the slope of the line joining the current point and other
        point 'q'.
        """
        q = self._from_(q)
        dx = q.x - self.x
        if abs(dx) < eps:
            return DEFAULT_FLOAT("inf")
        return ((q.y - self.y) / dx).astype(DEFAULT_FLOAT)

    def angle(
        self,
        p2: Union["Point2D", Tuple[FloatType, FloatType]],
        rad=True,
    ) -> DEFAULT_FLOAT:
        """Returns the angle between the current point and other point 'p2'

        Parameters
        ----------
        p2 : Union[list[float], "PointND"]
            Point or sequence of float_type
        rad : bool
            If True, returns angle in radians, otherwise in degrees

        Returns
        -------
        float_type
            Angle

        Examples
        --------
        >>> p = Point.from_([1.0, 2.0])
        >>> q = Point.from_([3.0, 4.0])
        >>> p.angle(q)
        0.7853981633974483

        """
        assert isinstance(p2, (Point2D, tuple)) and len(p2) == 2, (
            f"Expecting 2D points, but got {p2}"
        )
        ang = np.arctan2(p2[1] - self.y, p2[0] - self.x)
        ang = ang if ang >= 0 else ang + (2 * PI)
        ang = ang if rad else np.rad2deg(ang)
        return ang.astype(DEFAULT_FLOAT)

    def transform(
        self,
        angle: FloatType = 0.0,
        dx: FloatType = 0.0,
        dy: FloatType = 0.0,
        *,
        order: Literal["RT", "TR"] = "RT",  # Rotate > Trnsl | Trnsl > Rotate
    ) -> "Point2D":
        """Returns a new point transformed by rotation and translation
        around the origin
        """

        def rotate_2d(angle, x, y):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            return (x * cos_a - y * sin_a), (x * sin_a + y * cos_a)

        if order == "RT":
            new_x, new_y = rotate_2d(angle, self.x, self.y)
            new_x += dx
            new_y += dy
        elif order == "TR":
            new_x, new_y = rotate_2d(angle, self.x + dx, self.y + dy)
        else:
            raise ValueError(f"Invalid order: {order}")
        return self.__class__(new_x, new_y)


# endregion Point2D
# region Point3D


class Point3D(PointND):
    __slots__ = ()

    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"

    @property
    def x(self) -> DEFAULT_FLOAT:
        return self.coor[0]

    @property
    def y(self) -> DEFAULT_FLOAT:
        return self.coor[1]

    @property
    def z(self) -> DEFAULT_FLOAT:
        return self.coor[2]

    def transform(self, matrix: np.ndarray) -> "Point3D":
        """
        Transform using 4x4 transformation matrix (homogeneous coordinates)
        """
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        if matrix.dtype != DEFAULT_FLOAT:
            matrix = matrix.astype(DEFAULT_FLOAT)
        point = np.append(self.coor, 1.0)
        transformed = matrix @ point
        transformed = transformed[:3].astype(DEFAULT_FLOAT)
        return Point3D(*transformed)


# endregion Point3D
# region PointArrayND


class PointArrayND:
    """PointArrayND, a base class for representing a collection of points
    in N-dimensional space
    """

    __slots__ = ("_cycle", "coor")

    def __init__(self, points: np.ndarray):
        """Constructs a PointArray from a NumpyArray of points"""
        self._validate_points(points)
        self.coor = np.ascontiguousarray(points, dtype=DEFAULT_FLOAT)
        self._cycle = False  # For open curves

    # ============================
    # Private methods
    # ============================
    @staticmethod
    def _validate_points(points: np.ndarray) -> bool:
        """Validates the points in the array"""
        if not isinstance(points, np.ndarray):
            raise TypeError("PointArray construction requires a NumpyArray")
        if points.ndim != 2:
            raise ValueError("Points must be 2D array (n_points x n_dims)")
        if points.size == 0:
            raise ValueError("PointArray must have at least one point")
        return True

    @classmethod
    def from_points(
        cls,
        points: Sequence[PointND] | Sequence[Sequence[float]],
    ) -> "PointArrayND":
        """Constructs a PointArray from a sequence of Point objects or
        sequences of sequences of coordinates
        """
        if not points:
            raise ValueError("PointArray must have at least one point")
        if not isinstance(points, Sequence):
            raise TypeError(
                "Points must be a sequence of Point objects or "
                "sequences of sequence of coordinates",
            )
        _dim_ = len(points[0])
        if any(len(p) != _dim_ for p in points):
            raise ValueError("All points must have same dimension")
        return cls(np.array(points, dtype=DEFAULT_FLOAT))

    @classmethod
    def from_dims(cls, dims: Sequence[Sequence[FloatType]]):
        """
        Constructs a PointArray from a sequence of sequences of coordinates
        """
        if not dims:
            raise ValueError(
                "PointArray must have coordinates along at least one dimension"
            )
        if not all(len(d) == len(dims[0]) for d in dims):
            raise ValueError("All dimensions must have same length")
        return cls(np.array(dims, dtype=DEFAULT_FLOAT).T)

    # ============================
    #       MAGIC METHODS
    # ============================
    def __len__(self):
        """Returns the number of points in the PointArray"""
        return len(self.coor)

    def __getitem__(self, idx: int | slice | tuple) -> np.ndarray:
        """Returns the point(s) at the given index or slice"""
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError("Incorrect number of indices for PointArray")
            return self.coor[idx[0], idx[1]]
        return self.coor[idx]

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter(self.coor)

    def __array__(self, dtype=None, copy=True) -> np.ndarray:
        """Returns the coordinates of the point array as a numpy array"""
        arr = np.array(self.coor, dtype=dtype, copy=copy)
        return arr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({len(self)} points; dim={self.dim};"
            f" dtype={self.dtype})"
        )

    # ============================
    #       POINT PROPERTIES
    # ============================
    @property
    def dim(self):
        return self.coor.shape[1]

    @property
    def dtype(self):
        return self.coor.dtype

    @property
    def coordinates(self) -> np.ndarray:
        """Returns the coordinates of the point array"""
        return self.coor.copy()

    @property
    def cycle(self) -> bool:
        """Returns True if the points are cyclic"""
        return self._cycle

    @cycle.setter
    def cycle(self, val: bool):
        """Sets the cyclic property of the points"""
        if not isinstance(val, bool):
            raise TypeError("cycle take a boolean value")
        self._cycle = val

    # ============================
    #       GEOMETRY OPERATIONS
    # ============================
    def bounding_box(self) -> "BoundingBox":
        """Returns the bounding box of the current PointArray"""
        return BoundingBox(
            np.min(self.coor, axis=0),
            np.max(self.coor, axis=0),
        )

    def transform(
        self,
        matrix: np.ndarray,
        in_place: bool = False,
    ) -> Union["PointArrayND", None]:
        """Transform using a transformation matrix"""
        if matrix.shape != (self.dim + 1, self.dim + 1):
            raise ValueError(
                f"Transformation matrix must be {self.dim + 1}x{self.dim + 1}",
            )
        if matrix.dtype != self.dtype:
            matrix = matrix.astype(self.dtype)
        points = np.column_stack([self.coor, np.ones(len(self))])
        if in_place:
            transformed = points @ matrix.T
            if not transformed.dtype == self.dtype:
                transformed = transformed.astype(self.dtype)
            self.coor[:, :] = transformed[:, : self.dim]
            return None
        transformed = points @ matrix.T
        transformed = transformed[:, : self.dim].astype(self.dtype)
        return self.__class__(transformed)

    def reflection(
        self,
        p1: Union[list[float], "PointND"],
        p2: Union[list[float], "PointND"],
    ):
        """Reflects the current points about a line connecting p1 and p2"""
        raise NotImplementedError("Point Array reflection is not implemented")

    def reverse(self, in_place: bool = False) -> Union["PointArrayND", None]:
        """Reverses the order of the points"""
        rev_coor = np.flip(self.coor, axis=0)
        if in_place:
            self.coor = rev_coor
            return None
        return self.__class__(rev_coor)

    # ============================
    #       UTILITY METHODS
    # ============================
    def copy(self):
        """Returns a copy of the current PointArray"""
        return self.__class__(self.coordinates)

    def to_points_list(self) -> list[PointND]:
        """Returns a list of Point objects from the current PointArray"""
        return [PointND(*row) for row in self.coor]


# endregion PointArrayND
# region PointArray1D


class PointArray1D(PointArrayND):
    """PointArray1D, a subclass of PointArray, with one dimension"""

    __slots__ = ()

    def __init__(self, points: np.ndarray):
        """Constructs a PointArray1D from a NumpyArray"""
        if points.ndim == 1:
            points = np.atleast_2d(points).T
        super().__init__(points)
        if self.dim != 1:
            raise ValueError("PointArray1D must have one dimension")

    @property
    def x(self) -> np.ndarray:
        return self.coor[:, 0]

    def transform(
        self,
        dx: float = 0.0,
        in_place: bool = False,
    ) -> Union["PointArray1D", None]:
        """Transformation of the points cluster by rotation and translation"""
        if in_place:
            if dx != 0.0:
                self.coor[:] = self.coor[:] + dx
            return None
        return self.__class__(self.coor + dx)

# endregion PointArray1D
# region PointArray2D


class PointArray2D(PointArrayND):
    """PointArray2D, a subclass of PointArray, with two dimensions.

    Attributes
    ----------
    coordinates : np.ndarray
        Array of point coordinates
    dim : int
        Dimension of the points
    dtype : np.dtype
        Data type of the points
    x : NDArray
        Array of x coordinates
    y : NDArray
        Array of y coordinates

    """

    __slots__ = ()

    def __init__(self, points: np.ndarray) -> None:
        """Construct a PointArray2D from a NumpyArray."""
        super().__init__(points)
        if self.dim != 2:
            raise ValueError("PointArray2D must have 2 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]

    def transform(
        self,
        angle: FloatType = 0.0,
        dx: FloatType = 0.0,
        dy: FloatType = 0.0,
        pivot: Point2D | Tuple[FloatType, FloatType] = (0.0, 0.0),
        in_place: bool = False,
        order: str = "RT",
    ) -> Union["PointArray2D", None]:
        """Transformation of the points cluster by rotation and translation,
        either in-place or returning a new PointArray2D

        Parameters
        ----------
        angle : float
            Angle of rotation in radians, default: 0.0
        dx : float
            Translation along x axis, default: 0.0
        dy : float
            Translation along y axis, default: 0.0

        Returns
        -------
        PointArray2D

        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        temp_x = self.x - pivot[0]
        temp_y = self.y - pivot[1]
        if order == "RT":
            x = temp_x * cos_a - temp_y * sin_a + dx + pivot[0]
            y = temp_x * sin_a + temp_y * cos_a + dy + pivot[1]
        elif order == "TR":
            temp_x += dx
            temp_y += dy
            x = temp_x * cos_a - temp_y * sin_a + pivot[0]
            y = temp_x * sin_a + temp_y * cos_a + pivot[1]
        else:
            raise ValueError(f"Invalid order: {order}, should be 'RT' or 'TR'")

        if in_place:
            self.coor[:, 0] = x
            self.coor[:, 1] = y
            return None
        return PointArray2D(np.column_stack([x, y]))

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def sort(self) -> "PointArray2D":
        raise NotImplementedError("sort is not implemented")

    def plot(
        self,
        axs,
        points_plt_opt: dict | None = None,
        b_box: bool = False,
        box_plt_opt: dict | None = None,
    ):
        """Plots the points"""

        points_plt_options = {
            **_DEFAULT_POINT_PLOT_OPTIONS,
            **(points_plt_opt or {}),
        }
        axs.plot(
            np.append(self.x, self.x[0]) if self.cycle else self.x,
            np.append(self.y, self.y[0]) if self.cycle else self.y,
            **points_plt_options,
        )

        if b_box:
            bbox_plt_options = {
                **_DEFAULT_LINE_PLOT_OPTIONS,
                **(box_plt_opt or {}),
            }
            self.bounding_box.plot(axs, **bbox_plt_options)

        return axs


# endregion PointArray2D
# region PointArray3D


class PointArray3D(PointArrayND):
    """PointArray3D, a subclass of PointArray, with three dimensions"""

    __slots__ = ()

    def __init__(self, points):
        super().__init__(points)
        if self.dim != 3:
            raise ValueError("PointArray3D must have 3 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.coordinates[:, 2]

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """ """
        raise NotImplementedError("make_periodic_tiles is not implemented")


# endregion PointArray3D
# region BoundingBox


class BoundingBox(PlotMixin):  # TODO: Review this class
    """A class for performing n-dimensional bounding box operations

    It is expexcted that the number of elements in the lower and
    upper bound are same. Also, the lower bound must be less
    than the upper bound.

    Parameters
    ----------
    lower_bound : list or tuple
        Lower bounds of the box
    upper_bound : list or tuple
        Upper bounds of the box

    Raises
    ------
    ValueError
        If lower bound and upper bound have different length
        If lower bounds are greater than upper bounds

    Examples
    --------
    >>> import numpy as np
    >>> lower_bound = [0, 0]
    >>> upper_bound = [10, 10]
    >>> bounding_box = BoundingBox(lower_bound, upper_bound)
    >>> bounding_box.lb
    array([ 0.,  0.])
    >>> bounding_box.ub
    array([10., 10.])

    """

    __slots__ = ("_vertices", "_volume", "p_max", "p_min")

    def __init__(
        self,
        lower_bound: np.ndarray | Sequence[float],
        upper_bound: np.ndarray | Sequence[float],
    ):
        self.p_min = PointND(*lower_bound)
        self.p_max = PointND(*upper_bound)
        self._validate_bounds()

    def _validate_bounds(self) -> bool:
        if self.p_min.dim != self.p_max.dim:
            raise ValueError("lower and upper bounds must have same dimension")

        if np.any(self.p_min.coor > self.p_max.coor):
            raise ValueError(
                "lower bounds must be less than upper bounds"
                f"lower bounds: {self.p_min}, upper bounds: {self.p_max}",
            )
        return True

    def __eq__(self, bb_2: "BoundingBox") -> bool:
        """Checks if two bounding boxes are equal

        Parameters
        ----------
        bb_2 : BoundingBox
            Bounding box to be compared

        Returns
        -------
        bool
            True if bounding boxes are equal

        Raises
        ------
        TypeError
            If bb_2 is not of type BoundingBox

        """
        if not isinstance(bb_2, BoundingBox):
            raise TypeError("bb_2 must be of type BoundingBox")

        lb_equality = np.array_equal(self.p_min.coor, bb_2.p_min.coor)
        ub_equality = np.array_equal(self.p_max.coor, bb_2.p_max.coor)
        return lb_equality and ub_equality

    def __repr__(self) -> str:
        return (
            f"BoundingBox:\n\tlower_bound: {self.p_min.coor}"
            f"\n\tupper_bound: {self.p_max.coor}"
        )

    def has_point(self, p: PointND | Sequence[float]) -> bool:
        """Checks if the point 'p' is within the bounding box

        Returns
        -------
        bool
            True if 'p' is within the bounding box

        """
        p = PointND._from_(p)
        if not (self.p_min.dim == self.p_max.dim == p.dim):
            raise ValueError(
                f"point 'p' dimension {p.dim} does not match with "
                f"bounding box dimension {self.p_min.dim}",
            )
        return bool(
            np.all((self.p_min.coor <= p.coor) & (p.coor <= self.p_max.coor)),
        )

    def overlaps(self, bb: "BoundingBox", include_bounds=False) -> bool:
        """Returns True, if two bounding boxes overlap

        Returns
        -------
        bool
            True if two bounding boxes overlap

        """
        bb1_p1, bb1_p2 = self.p_min.coor, self.p_max.coor
        bb2_p1, bb2_p2 = bb.p_min.coor, bb.p_max.coor
        return bool(
            np.all(
                (bb1_p2 >= bb2_p1 if include_bounds else bb1_p2 > bb2_p1)
                & (bb1_p1 <= bb2_p2 if include_bounds else bb1_p1 < bb2_p2),
            ),
        )

    @property
    def volume(self) -> DEFAULT_FLOAT:
        """Returns the volume of the bounding box"""
        if not hasattr(self, "_volume"):
            self._volume = np.prod(self.p_max.coor - self.p_min.coor).astype(
                DEFAULT_FLOAT,
            )
        return self._volume

    @property
    def vertices(self) -> PointArrayND:
        if not hasattr(self, "_vertices"):
            vertices = list(
                product(*zip(self.p_min.coor, self.p_max.coor, strict=True))
            )
            self._vertices = PointArrayND(
                np.asarray(vertices, dtype=DEFAULT_FLOAT)
            )
        return self._vertices

    def get_patch(self, **rect_patch_kwargs) -> Patch:
        if self.p_min.dim != 2:
            raise ValueError("For plotting, bounding box must be 2D")
        (xl, yl), (xu, yu) = self.p_min.coor, self.p_max.coor
        return Rectangle((xl, yl), xu - xl, yu - yl, **rect_patch_kwargs)
