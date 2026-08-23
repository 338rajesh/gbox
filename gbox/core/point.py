import math
from collections.abc import Iterator, Sequence
from typing import Literal, Union


from .utils import (
    _assert_a_sequence,
    _assert_a_sequence_of_numbers,
    rotate_point_2d,
)


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
        degrees: bool = False,
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
        degrees : bool
            If True, the angle is interpreted as degrees, otherwise as radians.
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
        if degrees:
            angle = math.radians(angle)

        if order == "rotate_then_translate":
            new_x, new_y = rotate_point_2d(
                self.x,
                self.y,
                angle,
                pivot=pivot,
                degrees=False,
            )
            new_x += dx
            new_y += dy
        elif order == "translate_then_rotate":
            new_x, new_y = rotate_point_2d(
                self.x + dx,
                self.y + dy,
                angle,
                pivot=pivot,
                degrees=False,
            )
        else:
            raise ValueError(f"Invalid order: {order}")
        return self.__class__(new_x, new_y)


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
