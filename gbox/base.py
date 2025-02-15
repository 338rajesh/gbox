import numpy as np
import itertools
from typing import Union, Iterable

from .core import (
    TypeConfig,
    FloatType,
    IntType,
    get_type,
    cast_to,
)

PI = cast_to(np.pi, "float")
NDArray = np.ndarray

PointType = Union["PointND", Iterable[FloatType]]

NDArrayType = np.typing.NDArray


class PointND(tuple):
    def __new__(cls, *coords) -> "PointND":
        coords = [cast_to(c, "float") for c in coords]
        return super().__new__(cls, coords)

    def __repr__(self) -> str:
        return f"Point:\n\t{tuple(self)}"

    def __eq__(self, p: "PointND") -> bool:
        return all([a == b for a, b in zip(self, p)])

    @property
    def dim(self) -> IntType:
        """Returns the dimension of the point

        Returns
        -------
        IntType
            Dimension of the point

        """
        return len(self)

    @classmethod
    def _make_with_(cls, seq: PointType) -> "PointND":
        """Returns a Point from a sequence or NumpyArray of floats

        Supported types: list, tuple, numpy.ndarray of one dimension

        Parameters
        ----------
        seq : PointType
            Point or sequence of FloatType

        Returns
        -------
        Point
            Point

        Raises
        ------
        TypeError
            If seq is not of type list, tuple or NumpyArray
        ValueError
            If seq is numpy array with more than one dimension

        Example
        -------
        >>> Point._make_with_([1.0, 2.0])
        Point:  (1.0, 2.0)
        >>> Point._make_with_((1.0, 2.0))
        Point:  (1.0, 2.0)
        >>> Point._make_with_(np.array([1.0, 2.0]))
        Point:  (1.0, 2.0)

        """
        if isinstance(seq, PointND):
            return seq

        if isinstance(seq, (list, tuple, np.ndarray)):
            if hasattr(seq, "ndim") and seq.ndim > 1:
                raise ValueError(
                    "Only one-dimensional sequences are supported for Point construction"
                )
            return cls(*seq)
        else:
            raise TypeError(
                "Points must be given as NumpyArray or sequence of sequences of floats"
            )

    @staticmethod
    def _assert_points_compatibility_(p: "PointND", q: "PointND") -> None:
        """Checks if the given arguments are of Point type and have same dimension

        Parameters
        ----------
        p : Point
            First Point
        q : Point
            Second Point

        Raises
        ------
        TypeError
            If p or q are not of type Point
        ValueError
            If p and q are not of same dimension
        """
        if not isinstance(p, PointND):
            raise TypeError("p must be of type Point")
        if not isinstance(q, PointND):
            raise TypeError("q must be of type Point")
        if p.dim != q.dim:
            raise ValueError("p and q must be of same dimension")

    def distance_to(self, p: PointType) -> FloatType:
        """Evaluates the distance between the current point and point 'p'

        Parameters
        ----------
        p : PointType
            Point or sequence of FloatType

        Returns
        -------
        FloatType
            Distance

        Examples
        --------
        >>> p = Point._make_with_([1.0, 2.0])
        >>> q = Point._make_with_([3.0, 4.0])
        >>> p.distance_to(q)
        2.8284271247461903
        """
        p = PointND._make_with_(p)
        PointND._assert_points_compatibility_(self, p)
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
        >>> Point._make_with_([1.0, 2.0]).in_bounds(bounding_box)
        True
        >>> Point._make_with_([11.0, 12.0]).in_bounds((lower_bound, upper_bound))
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

    def reflection(self, q: PointType, p1: tuple, p2: tuple):
        """Reflects the current point about a line connecting p1 and p2"""
        raise NotImplementedError("reflect is not implemented")
        # Assert(q).of_type(Point, "other point must be of type Point")
        # p1, p2, q = np.array(p1), np.array(p2), q.as_array()
        # d = p2 - p1
        # u = d / np.linalg.norm(d)
        # projections = p1 + np.outer((q - p1) @ u, u)
        # reflected_point = 2 * projections - q
        # return Point(*reflected_point)

    def is_close_to(self, p: PointType, eps: FloatType = 1e-16) -> bool:
        """Checks if the current point is close to other point 'p'

        Parameters
        ----------
        p : PointType
            Point or sequence of FloatType
        eps : FloatType
            Tolerance, default: 1e-16

        Returns
        -------
        bool
            True if the current point is close to other point 'p', False otherwise

        Examples
        --------
        >>> p = Point._make_with_([1.0, 2.0])
        >>> p.is_close_to((1.0 + 1e-08, 2.0 + 1e-07), eps=1e-6)
        True
        >>> q = Point._make_with_([3.0, 4.0])
        >>> p.is_close_to(q)
        False
        """
        p = PointND._make_with_(p)
        PointND._assert_points_compatibility_(self, p)

        for a, b in zip(self, p):
            if abs(a - b) > eps:
                return False
        return True


class Point2D(PointND):
    def __new__(cls, x: float, y: float) -> "Point2D":
        return super().__new__(cls, x, y)

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def slope(self, q: PointType, eps: FloatType = None) -> FloatType:
        """Returns the slope of the line joining the current point and other point 'q'.

        Parameters
        ----------
        q : PointType
            Point or sequence of FloatType
        eps : FloatType
            Tolerance, defaults to precision of FloatType

        Returns
        -------
        FloatType
            Slope

        Examples
        --------
        >>> p = Point._make_with_([1.0, 2.0])
        >>> q = Point._make_with_([3.0, 4.0])
        >>> p.slope(q)
        1.0
        """
        if eps is None:
            eps = TypeConfig.float_precision()

        q = self._make_with_(q)
        assert isinstance(q, Point2D), "other point must be of type 'Point2D'"

        eps = eps if q.x == self.x else 0.0
        slope = (q.y - self.y) / (q.x - self.x + eps)
        return cast_to(slope, "float")

    def angle(self, p2: PointType, rad=True) -> FloatType:
        """Returns the angle between the current point and other point 'p2'

        Parameters
        ----------
        p2 : PointType
            Point or sequence of FloatType
        rad : bool
            If True, returns angle in radians, otherwise in degrees

        Returns
        -------
        FloatType
            Angle

        Examples
        --------
        >>> p = Point._make_with_([1.0, 2.0])
        >>> q = Point._make_with_([3.0, 4.0])
        >>> p.angle(q)
        0.7853981633974483
        """
        p2 = self._make_with_(p2)
        assert isinstance(p2, Point2D), "other point must be of type Point2D"
        ang = np.arctan2(p2.y - self.y, p2.x - self.x)
        if ang < 0:
            ang += 2.0 * PI
        ang = ang if rad else np.rad2deg(ang)
        return cast_to(ang, "float")

    def transform(
        self, angle: float = 0.0, dx: float = 0.0, dy: float = 0.0
    ) -> "Point2D":
        """Returns a new point transformed by rotation and translation around the origin

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
        >>> p = Point._make_with_([1.0, 0.0])
        >>> p.transform(angle=np.pi / 2)
        Point2D(x=0.0, y=1.0)
        """
        new_x = (self.x * np.cos(angle) - self.y * np.sin(angle)) + dx
        new_y = (self.x * np.sin(angle) + self.y * np.cos(angle)) + dy
        return Point2D(new_x, new_y)


class PointArrayND:
    """
    Collection of **ordered** points

    Attributes
    ----------
    coordinates : NDArray
        Two dimensional Numpy array of point coordinates
    dim : int
        Dimension of the points
    dtype : np.dtype
        Data type of the points

    Examples
    --------
    >>> p = PointArray([[1.0, 2.0], [3.0, 4.0]])
    >>> p
    PointArray:
    [[ 1.  2.]
     [ 3.  4.]]

    """

    def __init__(self, points: np.ndarray, dtype=None):
        """Constructs a PointArray from a NumpyArray

        Parameters
        ----------
        points : NDArray
            Two dimensional Numpy array of point coordinates, with one point per row.
        dtype : np.dtype
            Data type of the points. Defaults to FloatType

        Raises
        ------
        TypeError
            If points is not a NumpyArray
        NotImplementedError
            If points is not two-dimensional

        Examples
        --------
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
        if dtype is None:
            dtype = TypeConfig.float_type().dtype

        if not isinstance(points, np.ndarray):
            raise TypeError("PointArray is construction requred a NumpyArray")

        if points.ndim > 2:
            raise NotImplementedError(
                f"At present PointArray is only implemented for two-dimensional arrays. {points.ndim} found"
            )

        self.coordinates = np.asarray(points, dtype=dtype)
        self.dim = self.coordinates.shape[1]
        self.dtype = dtype
        self._cycle = False

    @classmethod
    def from_sequences(cls, *data, dtype=None) -> "PointArrayND":
        """Constructs a PointArray from a sequence of points

        Parameters
        ----------
        data : Sequence
            Sequence of points. Each point is a sequence of coordinates
        dtype : np.dtype
            Data type of the points. Defaults to FloatType

        Raises
        ------
        ValueError
            If data is empty
            If each point in the array does not have the same number of dimensions

        Examples
        --------
        >>> p = PointArray.from_sequences([1.0, 2.0], [3.0, 4.0])
        >>> isinstance(p, PointArray)
        True
        >>> p.coordinates
        array([[1., 2.],
               [3., 4.]])
        """
        if len(data) == 0:
            raise ValueError("PointArray must have at least one point")

        for a_point in data:
            if len(a_point) != len(data[0]):
                raise ValueError(
                    "Each point in the array must have the same number of dimensions"
                )

        return cls(np.array(data), dtype=dtype)

    @classmethod
    def from_dimensions_data(cls, *data, dtype=None) -> "PointArrayND":
        """Constructs a PointArray from a sequence of dimensional data

        Parameters
        ----------
        data : Sequence
            Sequence of dimensional data. Each dimension is a sequence of coordinates
        dtype : np.dtype
            Data type of the points. Defaults to FloatType

        Raises
        ------
        ValueError
            If data is empty
            If each dimension in the array does not have the same number of points

        Examples
        --------
        >>> p = PointArray.from_dimensions_data([1.0, 3.0, 5.0, 10.0], [2.0, 4.0, 6.0, 8.0])
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
                raise ValueError(
                    "Each dimension in the array must have the same number of points"
                )

        return cls(np.array(data).T, dtype=dtype)

    @classmethod
    def from_points(cls, *points, dtype=None) -> "PointArrayND":
        """Constructs a PointArray from a sequence of Point type objects

        Parameters
        ----------
        points : Sequence
            Sequence of Point type objects
        dtype : np.dtype
            Data type of the points. Defaults to FloatType

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
            a_p = PointND._make_with_(a_p)
            if not isinstance(a_p, PointND):
                raise TypeError(
                    f"Supplied points must be of type Point, but {type(a_p)} found"
                )
        data = np.array([p.as_list() for p in points])
        return cls(data, dtype=dtype)

    def __len__(self):
        return self.coordinates.shape[0]

    def __repr__(self):
        return f"PointArray:\n{self.coordinates}"

    def __eq__(self, p: "PointArrayND"):
        if not isinstance(p, PointArrayND):
            raise TypeError("Equality check requires two PointSets")

        return np.array_equal(self.coordinates, p.coordinates)

    def copy(self):
        """Returns a copy of the current PointArray"""
        return PointArrayND(self.coordinates.copy())

    @property
    def bounding_box(self) -> "BoundingBox":
        """Returns the bounding box of the current PointArray"""
        lb: list = [self.coordinates[:, i].min() for i in range(self.dim)]
        ub: list = [self.coordinates[:, i].max() for i in range(self.dim)]
        return BoundingBox(lb, ub)

    def reflection(self, p1: PointType, p2: PointType):
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


class PointArray1D(PointArrayND):
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

        if not isinstance(points, np.ndarray):
            raise TypeError("PointArray is construction requred a NumpyArray")

        if points.ndim > 2:
            raise NotImplementedError(
                "PointArray is construction requred a 2D NumpyArray"
            )

        if points.ndim != 1 and points.shape[1] != 1:
            raise ValueError(
                "PointArray must be a 1D array or 2D array with one column"
            )

        super(PointArray1D, self).__init__(points, dtype=dtype, **kwargs)

        assert (
            self.dim == 1
        ), "Constructing 'PointSet1D' requires one dimensional points"

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    def transform(self, dx: float = 0.0) -> "PointArray1D":
        """In-place transformation of the points cluster by rotation and translation

        Parameters
        ----------
        dx : float
            Translation along x axis

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

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def plot(self, axs, points_plt_opt: dict = None):
        """Plots the points"""

        assert (
            self.dim == 1
        ), "PointArray Plotting is supported only for 1D and 2D points"
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

    def __init__(self, points: np.ndarray, dtype=None, **kwargs):
        """Constructs a PointArray2D from a NumpyArray

        Parameters
        ----------
        points : NDArray
            Two dimensional Numpy array of point coordinates, with one point per row
        dtype : np.dtype
            Data type of the points, defaults to FloatType

        Raises
        ------
        TypeError
            If points is not a NumpyArray
        NotImplementedError
            If points is not two-dimensional

        """

        super(PointArray2D, self).__init__(points, dtype=dtype, **kwargs)

        assert (
            self.dim == 2
        ), "PonitArray2D construction produced a PointArray with dimension != 2"

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
    ) -> "PointArray2D":
        """In-place transformation of the points cluster by rotation and translation

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

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def sort(self) -> "PointArray2D":
        raise NotImplementedError("sort is not implemented")

    def plot(
        self,
        axs,
        b_box: bool = False,
        b_box_plt_opt: dict = None,
        points_plt_opt: dict = None,
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
    def __init__(self, points, **kwargs):
        super(PointArray3D, self).__init__(points, **kwargs)
        assert (
            self.dim == 3
        ), "Constructing 'Points3D' requires three dimensional points"

    @property
    def x(self) -> NDArray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> NDArray:
        return self.coordinates[:, 1]

    @property
    def z(self) -> NDArray:
        return self.coordinates[:, 2]

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """ """
        raise NotImplementedError("make_periodic_tiles is not implemented")


class BoundingBox:
    """A class for performing n-dimensional bounding box operations"""

    def __init__(self, lower_bound: list | tuple, upper_bound: list | tuple):
        """Bounding Box constructor

        It is expexcted that the number of elements in the lower and upper bound are same. Also, the lower bound must be less than the upper bound.

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
            raise ValueError("lower bound and upper bound must have same length")

        for i, j in zip(self.lb, self.ub):
            if i >= j:
                raise ValueError(
                    f"Expecting lower bounds to be less than upper bounds. "
                    f"But, {i} of lower bound is greater than {j} of upper bound"
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
                f"point 'p' dimension {len(p)} does not match bounding box dimension {self.dim}"
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
            lb1 <= ub2 and ub1 >= lb2 if incl_bounds else lb1 < ub2 and ub1 > lb2
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


class _TopologicalCurve:
    """Base class for all topological curves"""

    def __init__(self):
        self.points: PointArrayND = None

    def plot(
        self, axs, b_box=False, cycle=False, b_box_plt_opt=None, points_plt_opt=None
    ):
        self.points.cycle = cycle
        self.points.plot(axs, b_box, b_box_plt_opt, points_plt_opt)


class _TopologicalClosedShape:
    """Base class for all topological shapes in n-dimensions and closed"""

    def __init__(self):
        self.boundary: PointArrayND = None

    @property
    def bounding_box(self):
        return NotImplementedError("bounding_box is not implemented")


class TopologicalClosedShape2D(_TopologicalClosedShape):
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
        assert self.boundary is not None, "Boundary is not defined"
        assert (
            self.boundary.dim == 2
        ), f"Plot is supported for boundary in 2D only, but {self.boundary.dim}D points were provided"
        self.boundary.cycle = cycle
        self.boundary.plot(axs, b_box, b_box_plt_opt, points_plt_opt)
