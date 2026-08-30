import math
from collections.abc import Iterator, Sequence
from typing import Literal, Union, Any

import numpy as np
import numpy.typing as npt

from .transformation import (
    AngleUnits,
    transform_points,
    transformation_matrix_2d,
    transformation_matrix_3d,
)
from .utils import _assert_a_sequence, _assert_a_sequence_of_numbers


class PointND:
    """Point class for representing a point in N-dimensional space"""

    __slots__ = ("coordinates",)

    def __init__(self, *coords: float) -> None:
        """Constructs a point from the given coordinates

        Parameters
        ----------
        *coords : float
            The coordinates of the point in N-dimensional space
        """
        _assert_a_sequence_of_numbers(coords, name="coordinates")
        self.coordinates = tuple(float(c) for c in coords)

    # ============================
    #       CORE METHODS
    # ============================

    @classmethod
    def from_sequence(cls, s: Union[Sequence[float], "PointND"]) -> "PointND":
        """Constructs a point from a sequence of coordinates

        Parameters
        ----------
        s : Union[Sequence[float], PointND]
            A sequence of coordinates or another PointND instance
        """
        if isinstance(s, cls):
            return s

        _assert_a_sequence(s, name="coordinates")
        return cls(*s)

    # ============================
    #       MAGIC METHODS
    # ============================
    def __len__(self) -> int:
        """Returns the number of coordinates in the point"""
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> float:
        """Returns the coordinate at the given index"""
        return self.coordinates[idx]

    def __iter__(self) -> Iterator[float]:
        """Returns an iterator over the coordinates of the point"""
        return iter(self.coordinates)

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}({', '.join(map(str, self.coordinates))})"

    # =================================
    #       POINT PROPERTIES
    # =================================
    @property
    def dim(self) -> int:
        """Returns the dimension of the point"""
        return len(self.coordinates)

    def _check_same_dimension(
        self, other: Union["PointND", Sequence[float]]
    ) -> None:
        """Asserts that the current point and other point have the same dimension"""
        if isinstance(other, PointND):
            other_dim = other.dim
        else:
            _assert_a_sequence(other, name="other point")
            other_dim = len(other)

        if self.dim != other_dim:
            raise ValueError(
                f"Dimension mismatch: {self.dim}D vs {other_dim}D"
            )

    # =================================
    #       GEOMETRIC PROPERTIES
    # =================================
    def distance_to(self, q: Union["PointND", Sequence[float]]) -> float:
        """
        Returns the Euclidean distance between the current point
        and another point 'q'

        Parameters
        ----------
        q : Union["PointND", Sequence[float]]
            The other point to calculate the distance to.

        Returns
        -------
        float
            The Euclidean distance between the current point and point 'q'.
        """

        self._check_same_dimension(q)
        return math.hypot(*(a - b for a, b in zip(self.coordinates, q)))

    def in_bounds(
        self,
        lower_bound: Union["PointND", Sequence[float]],
        upper_bound: Union["PointND", Sequence[float]],
    ) -> bool:
        """Checks if the current point is within the given bounds

        Parameters
        ----------
        lower_bound : Union["PointND", Sequence[float]]
            The lower bound point or sequence of coordinates
        upper_bound : Union["PointND", Sequence[float]]
            The upper bound point or sequence of coordinates

        Returns
        -------
        bool
            True if the current point is within the bounds, False otherwise.
        """

        self._check_same_dimension(lower_bound)
        self._check_same_dimension(upper_bound)

        return all(
            (lb <= c <= ub)
            for c, lb, ub in zip(self.coordinates, lower_bound, upper_bound)
        )

    def is_close_to(
        self,
        q: Union["PointND", Sequence[float]],
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
        self._check_same_dimension(q)
        return all(
            math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
            for a, b in zip(self.coordinates, q)
        )


# ===========================================================================
#                               Point2D
# ===========================================================================


class Point2D(PointND):
    __slots__ = ()  # No additional attributes for 2D points

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}(x={self.x}, y={self.y})"

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

    def slope(
        self, q: Union["Point2D", Sequence[float]], eps: float = 1e-06
    ) -> float:
        """Returns the slope of the line joining the current point and other
        point 'q'.

        Parameters
        ----------
        q : Union["Point2D", Sequence[float]]
            The other point to calculate the slope with respect to.
        eps : float
            A small value to avoid division by zero. Defaults to 1e-06.

        Returns
        -------
        float
            The slope of the line joining the current point and other
            point 'q'.
        """
        q = self.__class__.from_sequence(q)
        dx = q.x - self.x

        if abs(dx) < eps:
            return float("inf")  # Vertical line, slope is infinite

        return float((q.y - self.y) / dx)

    def angle(
        self, q: Union["Point2D", Sequence[float]], degrees=False
    ) -> float:
        """Returns the angle between the current point and other point `q` in
        radians or degrees, measured counter-clockwise from the positive x-axis.

        Parameters
        ----------
        q : Union["PointND", Sequence[float]]
            Point or sequence of float
        degrees : bool
            If True, returns angle in degrees, otherwise in radians

        Returns
        -------
        float
            The measured angle

        Examples
        --------
        >>> Point2D(1.0, 2.0).angle([3.0, 4.0])
        0.7853981633974483

        """
        q = self.__class__.from_sequence(q)
        angle = math.atan2(q.y - self.y, q.x - self.x)

        if angle < 0:
            angle += 2 * math.pi

        return math.degrees(angle) if degrees else angle

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle: float = 0.0,
        *,
        angle_units: AngleUnits = "radians",
        pivot: Union["Point2D", Sequence[float]] = (0.0, 0.0),
        order: Literal[
            "rotate_then_translate", "translate_then_rotate"
        ] = "rotate_then_translate",
    ) -> "Point2D":
        """Returns a new point transformed by rotation and translation
        around the given pivot point.

        Parameters
        ----------
        dx : float
            Translation in the x-direction, defaults to 0.0
        dy : float
            Translation in the y-direction, defaults to 0.0
        angle : float
            Rotation angle in radians (or degrees if `degrees=True`), defaults to 0.
        angle_units : AngleUnits
            Units of the angle, either "radians" or "degrees", defaults to "radians"
        pivot : Union["Point2D", Sequence[float]]
            The pivot point for rotation, defaults to (0.0, 0.0)
        order : Literal["rotate_then_translate", "translate_then_rotate"]
            The order of transformations. If "rotate_then_translate",
            the point is first rotated around the pivot and then translated.
            If "translate_then_rotate", the point is first translated and then
            rotated around the pivot. Defaults to "rotate_then_translate".

        Returns
        -------
        Point2D
            The transformed point
        """
        transformation_martrix = transformation_matrix_2d(
            angle=angle,
            dx=dx,
            dy=dy,
            angle_units=angle_units,
            pivot=pivot,
            order=order,
        )
        points = np.array([[self.x, self.y]], dtype=np.float64)
        _p = transform_points(points=points, matrix=transformation_martrix)
        return self.__class__(_p[0, 0], _p[0, 1])


# ===========================================================================
#                               Point3D
# ===========================================================================


class Point3D(PointND):
    __slots__ = ()

    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)

    def __repr__(self) -> str:
        """Returns the string representation of the point"""
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

    @property
    def z(self) -> float:
        return self.coordinates[2]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class PointArrayND:
    """A collection of points in N-dimensional space. Coordinates are stored
    internally as a contiguous NumPy array with shape ``(n_points, n_dims)``
    and dtype ``float64``.
    """

    __slots__ = ("_coordinates",)

    def __init__(
        self, points: Sequence[Sequence[float]] | npt.NDArray[np.float64]
    ) -> None:
        """Constructs a PointArray from a NumpyArray of points

        Parameters
        ----------
        points : Sequence[Sequence[float]] | npt.NDArray[np.float64]
            A sequence of sequences of coordinates or a NumpyArray of shape
            (n_points, n_dims) and dtype float64. If a sequence of sequences
            is provided, the outer sequence should have length n_points and
            each inner sequence should have length n_dims. If a NumpyArray is
            provided, it should have shape (n_points, n_dims) of float64 dtype.
        """
        self._coordinates = self._validate_points(points)

    # ============================
    # Private methods
    # ============================
    @staticmethod
    def _validate_points(
        points: Any, dtype: np.dtype = np.dtype(np.float64)
    ) -> npt.NDArray[np.float64]:
        """Validates the points in the array and returns a NumpyArray of points
        with the expected dtype"""
        try:
            points = np.asarray(points, dtype=dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "PointArray construction requires a sequence "
                "of points or a NumpyArray"
            ) from e

        if points.ndim != 2:
            raise ValueError("Points must be a 2D array (n_points x n_dims)")

        if points.shape[0] == 0:
            raise ValueError("PointArray must have at least one point")

        return np.ascontiguousarray(points)

    @classmethod
    def from_dim_sequences(
        cls,
        sequences: Sequence[Sequence[float]],
        names: Sequence[str] | None = None,
    ) -> "PointArrayND":
        """
        Constructs a PointArray from a sequence of sequences of coordinates
        """
        if isinstance(sequences, np.ndarray):
            return cls(sequences)

        if names is not None and len(names) != len(sequences):
            raise ValueError(
                f"Expected {len(sequences)} names, got {len(names)}"
            )

        _assert_a_sequence(sequences, name="sequence of dimensions")

        for idx, sequence in enumerate(sequences):
            _assert_a_sequence_of_numbers(
                sequence,
                name=f"Dimension {idx} sequence" if not names else names[idx],
            )

        seq_lengths = {len(sequence) for sequence in sequences}
        if len(seq_lengths) != 1:
            raise ValueError(
                "All dimension sequences must have the same length, "
                f"but got lengths: {seq_lengths}"
            )

        return cls(np.column_stack(sequences))

    @classmethod
    def from_named_dims(cls, **data: Sequence[float]) -> "PointArrayND":
        """
        Constructs a PointArray from a dictionary of sequences of coordinates
        """
        if not data:
            raise ValueError("Input dictionary is empty")

        return cls.from_dim_sequences(
            list(data.values()), names=list(data.keys())
        )

    # ============================
    #       MAGIC METHODS
    # ============================

    def __len__(self) -> int:
        """Returns the number of points in the PointArray"""
        return self._coordinates.shape[0]

    def __getitem__(self, idx: int | slice | tuple) -> np.ndarray:
        """Returns the point(s) at the given index or slice"""
        return self._coordinates[idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._coordinates)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"points={self._coordinates!r}, "
            f"dim={self.dim}, "
            f"dtype={self.dtype})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} with {len(self)} points in {self.dim}D"
        )

    # ============================
    #       POINT PROPERTIES
    # ============================
    @property
    def dim(self) -> int:
        return self._coordinates.shape[1]

    @property
    def dtype(self) -> np.dtype:
        return self._coordinates.dtype

    @property
    def coordinates(self) -> npt.NDArray[np.float64]:
        return self._coordinates

    # ============================
    #       GEOMETRY OPERATIONS
    # ============================
    def bounding_box(self) -> tuple[list[float], list[float]]:
        """Returns the bounding box of the current PointArray"""
        return (
            np.min(self._coordinates, axis=0).tolist(),
            np.max(self._coordinates, axis=0).tolist(),
        )

    # ============================
    #       UTILITY METHODS
    # ============================

    def copy(self) -> "PointArrayND":
        """Returns a copy of the current PointArray"""
        return self.__class__(self._coordinates.copy())

    def to_list(self) -> list[list[float]]:
        """Returns a list of lists of coordinates from the current PointArray"""
        return self._coordinates.tolist()


# endregion PointArrayND

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

    def __init__(
        self, points: npt.NDArray[np.float64] | Sequence[Sequence[float]]
    ) -> None:
        super().__init__(points)

        if self.dim != 2:
            raise ValueError(f"PointArray2D must have 2 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self._coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._coordinates[:, 1]

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle: float = 0.0,
        *,
        angle_units: AngleUnits = "radians",
        pivot: Sequence[float] = (0.0, 0.0),
        in_place: bool = False,
        order: Literal[
            "rotate_then_translate", "translate_then_rotate"
        ] = "rotate_then_translate",
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
        angle_units : AngleUnits
            Units of the angle, either "radians" or "degrees", default: "radians"
        pivot : tuple[float, float]
            Pivot point for rotation, default: (0.0, 0.0)
        in_place : bool
            If True, the transformation is applied in-place and the method returns None.
            If False, a new PointArray2D is returned with the transformed points.
        order : str
            Order of operations, either "rotate_then_translate" or
            "translate_then_rotate", default: "rotate_then_translate"

        Returns
        -------
        PointArray2D

        """
        transformation_matrix = transformation_matrix_2d(
            angle=angle,
            dx=dx,
            dy=dy,
            angle_units=angle_units,
            pivot=pivot,
            order=order,
        )

        _p = transform_points(
            points=self._coordinates,
            transformation_matrix=transformation_matrix,
        )

        if in_place:
            self._coordinates = _p
            return None

        return self.__class__(_p)


# endregion PointArray2D
# region PointArray3D


class PointArray3D(PointArrayND):
    """PointArray3D, a subclass of PointArray, with three dimensions"""

    __slots__ = ()

    def __init__(
        self, points: npt.NDArray[np.float64] | Sequence[Sequence[float]]
    ) -> None:
        super().__init__(points)

        if self.dim != 3:
            raise ValueError(f"PointArray3D must have 3 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self._coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._coordinates[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self._coordinates[:, 2]

    def transform(
        self,
        *,
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        angle_units: AngleUnits = "radians",
        pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
        in_place: bool = False,
        order: str = "rotate_then_translate",
        rotation_order: str = "xyz",
    ) -> Union["PointArray3D", None]:
        transformation_matrix = transformation_matrix_3d(
            angles=angles,
            dx=dx,
            dy=dy,
            dz=dz,
            angle_units=angle_units,
            pivot=pivot,
            order=order,
            rotation_order=rotation_order,
        )
        _p = transform_points(
            points=self._coordinates,
            transformation_matrix=transformation_matrix,
        )

        if in_place:
            self._coordinates = _p
            return None

        return self.__class__(_p)
