import numpy as np
import itertools
from typing import Union, TypeVar, Type, List, Generic, cast

from .core import (
    TypeConfig,
    FloatType,
    float_type,
    get_type,
    cast_to,
)

from .constants import PI
NDArray = np.ndarray
NDArrayType = np.typing.NDArray

PointNDType = TypeVar("PointNDType", bound="PointND")

# ============================================================================
#                           POINT ND CLASS
# ============================================================================


class PointND(tuple):
    """Point class for representing a point in N-dimensional space"""

    def __new__(cls, *coords):
        coords = [cast_to(c, "float") for c in coords]
        return super().__new__(cls, coords)

    def __init__(self, *coords) -> None:
        self.coords = coords

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} class; dim {len(self)}>"

    def __eq__(self, p: "PointND") -> bool:
        for a, b in zip(self, p):
            if a != b:
                return False
        return True

    @property
    def dim(self) -> int:
        """Returns the dimension of the point

        Returns
        -------
        IntType
            Dimension of the point

        """
        return self.__len__()

    @classmethod
    def from_(
        cls: Type[PointNDType], seq: Union[list[float_type], "PointND"]
    ) -> PointNDType:
        """
        Returns a Point from a list/tuple of floats.

        Parameters
        ----------
        seq : Union[list[float], tuple[float], "PointND"]
            Point or list/tuple of float_type

        Returns
        -------
        Point
            Point

        Raises
        ------
        TypeError
            If seq is not of type list, tuple or PointND
            If seq contains elements that are not of float type, see
            `FloatType._types_` for supported float types.
        ValueError
            If seq is empty

        Example
        -------
        >>> Point.from_([1.0, 2.0])
        Point:  (1.0, 2.0)
        >>> Point.from_((1.0, 2.0))
        Point:  (1.0, 2.0)

        """
        if not isinstance(seq, (list, tuple, PointND)):
            raise TypeError("seq must be of type list, tuple or Point")

        if isinstance(seq, cls):
            return seq

        # i.e., seq is either list or tuple
        if len(seq) == 0:
            raise ValueError("Point must have at least one element")
        for c in seq:
            if not isinstance(c, FloatType._types_):
                raise TypeError(
                    "All elements must be of type float",
                    f"but, element {c} found to be of type {type(c)}",
                )
        return cls(*seq)

    @staticmethod
    def _assert_points_compatibility_(p, q) -> None:
        """
        Checks if the given arguments are of Point type and have same
        dimension
        """
        if not isinstance(p, PointND):
            raise TypeError("p must be of type Point")
        if not isinstance(q, PointND):
            raise TypeError("q must be of type Point")
        if p.dim != q.dim:
            raise ValueError("p and q must be of same dimension")

    def distance_to(
            self, p: Union[list[float_type], "PointND"]
    ) -> float_type:
        """Evaluates the distance between the current point and point 'p'

        Parameters
        ----------
        p : Union[list[float], "PointND"]
            Point or sequence of float_type

        Returns
        -------
        float_type
            Distance

        Examples
        --------
        >>> p = Point.from_([1.0, 2.0])
        >>> q = Point.from_([3.0, 4.0])
        >>> p.distance_to(q)
        2.8284271247461903
        """
        p = self.__class__.from_(p)
        self.__class__._assert_points_compatibility_(self, p)
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(self, p)))
        return cast_to(distance, "float")

    def in_bounds(
        self,
        bounds: Union["BoundingBox", list, tuple],
        include_bounds: bool = False,
    ) -> bool:
        """Checks if the current point is within the bounds.

        Parameters
        ----------
        bounds : Union["BoundingBox", list, tuple]
            Bounding box or list/tuple of bounds
        include_bounds : bool
            If True, the point is considered within and on the bounds.
            Default: False

        Raises
        ------
        TypeError
            If bounds is not of type BoundingBox, or list or tuple
        ValueError
            If bounds are of type list or tuple and has length other than 2
            If dimension of Point and Bounds are not same

        Examples
        --------
        >>> import numpy as np
        >>> lower_bound = [0, 0]
        >>> upper_bound = [10, 10]
        >>> bounding_box = BoundingBox(lower_bound, upper_bound)
        >>> Point.from_([1.0, 2.0]).in_bounds(bounding_box)
        True
        >>> Point.from_([11.0, 12.0]).in_bounds((lower_bound, upper_bound))
        False
        """

        # converting list/tuple of bounds to BoundingBox
        if isinstance(bounds, (list, tuple)):
            if len(bounds) == 2:
                bounds = BoundingBox(*bounds)
            else:
                raise ValueError(
                    "Bounds must have length 2 if supplied as list or tuple"
                )

        # validating bounds type
        if not isinstance(bounds, BoundingBox):
            raise TypeError("bounds are expected to be BoundingBox")

        # validating dimension compatibility
        if self.dim != bounds.dim:
            raise ValueError("Mismatch in the dimension of Point and Bounds")

        for lb, p, ub in zip(bounds.lb, self, bounds.ub):
            if (p < lb or p > ub) if include_bounds else (p <= lb or p >= ub):
                return False
        return True

    def as_array(self) -> NDArray:
        return np.asarray(self, dtype=get_type("float"))

    def as_list(self) -> list:
        return list(self)

    def reflection(
            self, q: Union[list[float], "PointND"], p1: tuple, p2: tuple
    ):
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
            p: Union[list[float_type], "PointND"],
            eps: float_type = 1e-16,
    ) -> bool:
        """Checks if the current point is close to other point 'p'

        Parameters
        ----------
        p : Union[list[float], "PointND"]
            Point or sequence of float_type
        eps : float_type
            Tolerance, default: 1e-16

        Returns
        -------
        bool
            True if the current point is close to other point 'p',
            False otherwise.

        Examples
        --------
        >>> p = Point.from_([1.0, 2.0])
        >>> p.is_close_to((1.0 + 1e-08, 2.0 + 1e-07), eps=1e-6)
        True
        >>> q = Point.from_([3.0, 4.0])
        >>> p.is_close_to(q)
        False
        """
        p = self.__class__.from_(p)
        self.__class__._assert_points_compatibility_(self, p)

        for a, b in zip(self, p):
            if abs(a - b) > eps:
                return False
        return True


class Point1D(PointND):
    pass


class Point2D(PointND):
    def __init__(self, x: float_type, y: float_type):
        super(Point2D, self).__init__(x, y)
        self.x = cast_to(x, "float")
        self.y = cast_to(y, "float")

    def slope(
            self,
            q: Union[list[float_type], "PointND"],
            eps: float_type | None = None,
    ) -> float_type:
        """Returns the slope of the line joining the current point and other
        point 'q'.

        Parameters
        ----------
        q : Union[list[float], "PointND"]
            Point or sequence of float_type
        eps : float_type
            Tolerance, defaults to precision of float_type

        Returns
        -------
        float_type
            Slope

        Examples
        --------
        >>> p = Point.from_([1.0, 2.0])
        >>> q = Point.from_([3.0, 4.0])
        >>> p.slope(q)
        1.0
        """
        if eps is None:
            eps = cast_to(TypeConfig.float_precision(), "float")

        q = self.from_(q)
        assert isinstance(q, self.__class__), (
            f"other point must be of type '{self.__class__.__name__}'"
        )

        eps = eps if q.x == self.x else cast_to(0.0, "float")
        slope = (q.y - self.y) / (q.x - self.x + eps)
        return cast_to(slope, "float")

    def angle(
        self,
        p2: Union[list[float_type], "PointND"],
        rad=True
    ) -> float_type:
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
        p2 = self.from_(p2)
        assert isinstance(p2, self.__class__), (
            f"other point must be of type '{self.__class__.__name__}'"
        )
        ang = np.arctan2(p2.y - self.y, p2.x - self.x)
        if ang < 0:
            ang += 2.0 * PI
        ang = ang if rad else np.rad2deg(ang)
        return cast_to(ang, "float")

    def transform(
        self,
        angle: float_type = 0.0,
        dx: float_type = 0.0,
        dy: float_type = 0.0
    ) -> "Point2D":
        """Returns a new point transformed by rotation and translation
         around the origin

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
        Point2D
            Transformed point

        Examples
        --------
        >>> p = Point.from_([1.0, 0.0])
        >>> p.transform(angle=np.pi / 2)
        Point2D(x=0.0, y=1.0)
        """
        new_x = (self.x * np.cos(angle) - self.y * np.sin(angle)) + dx
        new_y = (self.x * np.sin(angle) + self.y * np.cos(angle)) + dy
        return type(self)(new_x, new_y)


class Point3D(PointND):
    pass


class PointArrayND(Generic[PointNDType]):
    """
    Constructs a PointArray from a NumpyArray, representing a
    collection of **ordered** points

    Attributes
    ----------
    coordinates : NDArray
        Two dimensional Numpy array of point coordinates
    dim : int
        Dimension of the points
    dtype : np.dtype
        Data type of the points

    Parameters
    ----------
    points : NDArray
        Two dimensional Numpy array of point coordinates, with
            one point per row.
    dtype : np.dtype
        Data type of the points. Defaults to float_type

    Raises
    ------
    TypeError
        If points is not a NumpyArray
    NotImplementedError
        If points is not two-dimensional

    Examples
    --------
    >>> p = PointArray([[1.0, 2.0], [3.0, 4.0]])
    >>> p
    PointArray:
    [[ 1.  2.]
     [ 3.  4.]]
    >>> import numpy as np
    >>> from gbox import PointArray, TypeConfig
    >>> TypeConfig.set_float_type(np.float64)
    >>> p = PointArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> p.coordinates
    array([[1., 2.],
            [3., 4.]])
    >>> p.dim
    2
    >>> p.dtype
    <class 'numpy.float64'>
    """

    point_cls: Type[PointNDType] = cast(Type[PointNDType], PointND)

    def _dim(self):
        _dim = self.__class__.__name__.split("Array")[1][:-1]
        return int(_dim) if _dim.isnumeric() else None

    def __init__(self, points: np.ndarray, dtype=None):
        if dtype is None:
            dtype = TypeConfig.float_type().dtype
        assert self._check_points_validity(points)
        self.coordinates = np.asarray(points, dtype=dtype)
        self.dim = self.coordinates.shape[1]
        self.dtype = dtype
        self._cycle = False

    def _check_points_validity(self, points):
        if not isinstance(points, np.ndarray):
            raise TypeError("PointArray is construction requred a NumpyArray")

        if points.ndim != 2:
            raise NotImplementedError(
                "At present PointArray is only implemented for 2D arrays."
                " I.e., only ONE group of points in n-dimensional plane is"
                " supported, and not the multiple/nested groups of points."
                " But, {points.ndim} point groups found"
            )

        if self._dim() is not None and points.shape[1] != self._dim():
            raise ValueError(
                f"Expected {self._dim()}D points represented by "
                f"{self._dim()} columns in a 2D array, but found "
                f"{points.shape[1]} columns"
            )
        return True

    @classmethod
    def from_sequences(cls, *data, dtype=None) -> "PointArrayND":
        """
        Constructs a PointArray from a sequence of points

        Parameters
        ----------
        data : Sequence
            Sequence of points. Each point is a sequence of coordinates
        dtype : np.dtype
            Data type of the points. Defaults to float_type

        Raises
        ------
        ValueError
            If data is empty
            If each point in the array does not have the same number
            of dimensions

        Examples
        --------
        >>> p = PointArray.from_sequences([1.0, 2.0], [3.0, 4.0])
        >>> isinstance(p, PointArray)
        True
        >>> p.coordinates
        array([[1., 2.], [3., 4.]])
        """
        if len(data) == 0:
            raise ValueError("PointArray must have at least one point")

        for a_point in data:
            if len(a_point) != len(data[0]):
                raise ValueError(
                    "Each point in the array must have same dimension"
                )

        return cls(np.array(data), dtype=dtype)

    @classmethod
    def from_dimensions_data(cls, *data, dtype=None) -> "PointArrayND":
        """Constructs a PointArray from a sequence of dimensional data

        Parameters
        ----------
        data : Sequence
            Sequence of dimensional data where sequence contains
             a dimension coordinates
        dtype : np.dtype
            Data type of the points. Defaults to float_type

        Raises
        ------
        ValueError
            If data is empty
            If all dimensions doesn't have the same number of points

        Examples
        --------
        >>> p = PointArray.from_dimensions_data(
        [1.0, 3.0, 5.0, 10.0], [2.0, 4.0, 6.0, 8.0]
        )
        >>> isinstance(p, PointArray)
        True
        >>> p.coordinates
        array([[ 1.,  2.],
               [ 3.,  4.],
               [ 5.,  6.],
               [10.,  8.]])
        """
        if len(data) == 0:
            raise ValueError("PointArray must have at least one dimension")

        for a_dim_data in data:
            if len(a_dim_data) != len(data[0]):
                raise ValueError("Unequal number of points along dimensions")

        return cls(np.array(data).T, dtype=dtype)

    @classmethod
    def from_points(cls, *points, dtype=None) -> "PointArrayND":
        """Constructs a PointArray from a sequence of Point type objects

        Parameters
        ----------
        points : Sequence
            Sequence of Point type objects
        dtype : np.dtype
            Data type of the points. Defaults to float_type

        Raises
        ------
        TypeError
            If points contains a non-Point object

        Examples
        --------
        >>> p = PointArray.from_points(Point(1.0, 2.0), Point(3.0, 4.0))
        >>> isinstance(p, PointArray)
        True
        >>> p.coordinates
        array([[1., 2.],
               [3., 4.]])
        """
        for a_p in points:
            a_p = cls.point_cls.from_(a_p)
            if not isinstance(a_p, cls.point_cls):
                raise TypeError(
                    "Supplied points must be of type Point,"
                    f"but {type(a_p)} found"
                )
        data = np.array([p.as_list() for p in points])
        return cls(data, dtype=dtype)

    def __len__(self):
        return self.coordinates.shape[0]

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}; dim::{self.dim};"
            f" {self.coordinates.shape[0]} points>"
        )

    def __eq__(self, p: "PointArrayND"):
        if not isinstance(p, self.__class__):
            raise TypeError("Equality check requires two PointSets")
        return np.array_equal(self.coordinates, p.coordinates)

    def copy(self):
        """Returns a copy of the current PointArray"""
        return self.__class__(self.coordinates.copy())

    @property
    def bounding_box(self) -> "BoundingBox":
        """Returns the bounding box of the current PointArray"""
        lb: list = [self.coordinates[:, i].min() for i in range(self.dim)]
        ub: list = [self.coordinates[:, i].max() for i in range(self.dim)]
        return BoundingBox(lb, ub)

    def trasnform(self, angle, dx, dy):
        raise NotImplementedError(
            "N-dime point array transformation is not implemented"
        )

    def reflection(
            self,
            p1: Union[list[float], "PointND"],
            p2: Union[list[float], "PointND"]
    ):
        """Reflects the current points about a line connecting p1 and p2"""
        raise NotImplementedError("Point Array reflection is not implemented")

    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError("cycle take a boolean value")
        self._cycle = val


class PointArray1D(PointArrayND["Point1D"]):
    """
    PointArray1D, a subclass of PointArray, with one dimension

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

    """
    point_cls: Type["Point1D"] = Point1D

    def __init__(self, points: np.ndarray, dtype=None, **kwargs):
        """Constructs a PointArray1D from a NumpyArray

        Parameters
        ----------
        points : NDArray
            One dimensional Numpy array or 2D Numpy array with one column
        dtype : np.dtype
            Data type of the points

        Raises
        ------
        TypeError
            If points is not a NumpyArray
        NotImplementedError
            If points is a numpy array with more than 2 dimensions
        ValueError
            If points is a numpy array with more than one column

        """
        super(PointArray1D, self).__init__(points, dtype=dtype, **kwargs)

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    def transform(self, dx: float = 0.0) -> "PointArray1D":
        """In-place transformation of the points cluster
         by rotation and translation

        Parameters
        ----------
        dx : float
            T ranslation along x axis

        Returns
        -------
        PointArray1D

        """
        if dx != 0.0:
            self.coordinates[:] = self.coordinates[:] + dx
        return self

    def reverse(self) -> "PointArray1D":
        """Reverses the order of points **in-place**"""
        self.coordinates[:] = np.flip(self.coordinates, axis=0)
        return self

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def plot(self, axs, points_plt_opt: dict | None = None):
        """Plots the points"""

        assert self.dim == 1, (
            "PointArray Plotting is supported only for 1D and 2D points"
        )
        _plt_opt = {"color": "blue", "marker": "o", "linestyle": "None"}

        # Plot points
        if points_plt_opt is not None:
            _plt_opt.update(points_plt_opt)

        axs.plot(
            self.x if not self.cycle else np.append(self.x, self.x[0]),
            **_plt_opt,
        )

        axs.axis("equal")


class PointArray2D(PointArrayND):
    """
    PointArray2D, a subclass of PointArray, with two dimensions

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

    point_cls: Type["Point2D"] = Point2D

    def __init__(self, points: np.ndarray, dtype=None, **kwargs):
        """Constructs a PointArray2D from a NumpyArray

        Parameters
        ----------
        points : NDArray
            Two dimensional Numpy array of point coordinates, with
             one point per row
        dtype : np.dtype
            Data type of the points, defaults to float_type

        Raises
        ------
        TypeError
            If points is not a NumpyArray
        NotImplementedError
            If points is not two-dimensional

        """
        super(PointArray2D, self).__init__(points, dtype=dtype, **kwargs)

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> NDArray:
        return self.coordinates[:, 1]

    def transform(
        self,
        angle: float_type = 0.0,
        dx: float_type = 0.0,
        dy: float_type = 0.0,
    ) -> "PointArray2D":
        """
        In-place transformation of the points cluster
         by rotation and translation

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
        x_ = (self.x * np.cos(angle) - self.y * np.sin(angle)) + dx
        y_ = (self.x * np.sin(angle) + self.y * np.cos(angle)) + dy
        self.coordinates[:, 0] = x_
        self.coordinates[:, 1] = y_
        return self

    def reverse(self) -> "PointArray2D":
        """Reverses the order of points **in-place**"""
        self.coordinates[:] = np.flip(self.coordinates, axis=0)
        return self

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def sort(self) -> "PointArray2D":
        raise NotImplementedError("sort is not implemented")

    def plot(
        self,
        axs,
        b_box: bool = False,
        b_box_plt_opt: dict | None = None,
        points_plt_opt: dict | None = None,
    ):
        """Plots the points"""
        if self.dim > 2:
            raise NotImplementedError(
                "PointArray Plotting is supported only for 1D and 2D points"
            )

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


class PointArray3D(PointArrayND):
    point_cls: Type["Point3D"] = Point3D

    def __init__(self, points, **kwargs):
        super(PointArray3D, self).__init__(points, **kwargs)

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> NDArray:
        return self.coordinates[:, 1]

    @property
    def z(self) -> NDArray:
        return self.coordinates[:, 2]

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """ """
        raise NotImplementedError("make_periodic_tiles is not implemented")


class BoundingBox:
    """
    A class for performing n-dimensional bounding box operations

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

    def __init__(
            self,
            lower_bound: List[float_type],
            upper_bound: List[float_type],
    ):
        self.lb = np.asarray(lower_bound, dtype=get_type("float"))
        self.ub = np.asarray(upper_bound, dtype=get_type("float"))
        self._check_bounds_validity()
        self.centre = (self.lb + self.ub) * 0.5
        self.dim = len(self.lb)
        self.vertices = PointArrayND(
            np.asarray(list(itertools.product(*zip(self.lb, self.ub)))),
            dtype=get_type("float"),
        )
        self.side_lengths = self.ub - self.lb
        self.volume = cast_to(np.prod(self.side_lengths), "float")

    def _check_bounds_validity(self):
        if len(self.lb) != len(self.ub):
            raise ValueError(
                "lower bound and upper bound must have same length"
            )

        for i, j in zip(self.lb, self.ub):
            if i >= j:
                raise ValueError(
                    f"Expecting lower bounds to be < upper bounds. "
                    f"But, {i} of lower bound is > {j} of upper bound"
                )

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

        lb_equality = np.array_equal(self.lb, bb_2.lb)
        ub_equality = np.array_equal(self.ub, bb_2.ub)
        return lb_equality and ub_equality

    def __repr__(self) -> str:
        """Returns the string representation of the bounding box"""
        return f"BoundingBox:\nLower Bound: {self.lb},\nUpper Bound: {self.ub}"

    def has_point(self, p: list | tuple) -> bool:
        """Checks if the point 'p' is within the bounding box

        Returns
        -------
        bool
            True if 'p' is within the bounding box

        """
        if self.dim != len(p):
            raise ValueError(
                "point 'p' dimension {len(p)} does not match"
                f"bounding box dimension {self.dim}"
            )
        return all([lb <= p <= ub for lb, p, ub in zip(self.lb, p, self.ub)])

    def overlaps_with(self, bb: "BoundingBox", incl_bounds=False) -> bool:
        """Returns True, if two bounding boxes overlap

        Returns
        -------
        bool
            True if two bounding boxes overlap

        """
        return all(
            lb1 <= ub2 and ub1 >= lb2
            if incl_bounds
            else lb1 < ub2 and ub1 > lb2
            for lb1, ub1, lb2, ub2 in zip(self.lb, self.ub, bb.lb, bb.ub)
        )

    def plot(self, axs, cycle=True, **plt_opt) -> None:
        """Plots the bounding box for two dimensional bounding box

        Raises
        ------
        ValueError
            If the dimension of the bounding box is not 2

        """

        if self.dim != 2:
            raise ValueError(
                f"Bounding Box can only be plotted in 2D, but {self.dim} found"
            )
        (xl, yl), (xu, yu) = self.lb, self.ub

        x = np.array([xl, xu, xu, xl])
        y = np.array([yl, yl, yu, yu])

        if cycle:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        axs.plot(x, y, **plt_opt)


class _T_CurveND(Generic[PointNDType]):
    """Base class for n-dimensional topological curves"""

    points_class: Type[PointNDType] = cast(Type[PointNDType], PointND)

    def __init__(self, points: PointArrayND):
        self.points: PointArrayND = points

    @property
    def dim(self):
        return self.points.dim

    def __repr__(self):
        return f"<{self.__class__.__name__} class; dim{self.points.dim}>"


class _T_ClosedShapeND:
    """Base class for all topological shapes in n-dimensions and closed"""

    def __init__(self):
        self.boundary_points: PointArrayND | None = None

    @property
    def bounding_box(self):
        return NotImplementedError("bounding_box is not implemented")


class _T_ClosedShape2D(_T_ClosedShapeND):
    """Base class for the two-dimensional topological shapes"""

    points_class: Type[Point2D] = Point2D

    def __init__(self, *args, **kwargs):
        super(_T_ClosedShape2D, self).__init__(*args, **kwargs)
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
        return self.perimeter / np.sqrt(4.0 * PI * self.area)

    @property
    def eq_radius(self):
        return np.sqrt(self.area / PI)

    def plot(
        self,
        axs,
        b_box=False,
        b_box_plt_opt=None,
        points_plt_opt=None,
        cycle=True,
    ):
        assert self.boundary_points is not None, "Boundary is not defined"
        assert self.boundary_points.dim == 2, (
            "Plot is supported for boundary in 2D only,"
            f"but {self.boundary_points.dim}D points were provided"
        )
        self.boundary_points.cycle = cycle
        # self.boundary.plot(axs, b_box, b_box_plt_opt, points_plt_opt)


class ConicCurve(_T_CurveND):
    _point_density: int = 100

    def __init__(self):
        points = PointArrayND(np.array([]))
        super(ConicCurve, self).__init__(points)
