import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.integrate import quad

from ..core.utils import (
    _validate_positive_float,
    _validate_float,
    _validate_positive_int,
)
from ..core.points import Point2D, PointArray2D


@dataclass
class Shape2DPosition:
    """
    A triplet of (x, y, theta) where theta is in radians measured in the
    counter-clockwise direction from the positive x-axis.
    """

    x: float
    y: float
    theta: float

    def __repr__(self):
        return f"Shape2DPosition(x={self.x}, y={self.y}, theta={self.theta})"

    def rotate(self, rot_angle, in_place=False):
        if in_place:
            self.theta += rot_angle
            return self
        return Shape2DPosition(self.x, self.y, self.theta + rot_angle)

    def translate(self, dx, dy, in_place=False):
        if in_place:
            self.x += dx
            self.y += dy
            return self
        return Shape2DPosition(self.x + dx, self.y + dy, self.theta)


class Shape2D(ABC):
    @property
    @abstractmethod
    def position(self) -> Shape2DPosition:
        pass

    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

    @property
    @abstractmethod
    def equivalent_circle_radius(self):
        pass

    @abstractmethod
    def contains_point(self, point):
        pass

    @abstractmethod
    def union_of_circles(self):
        pass

    def bounding_box(self) -> tuple[list[float], list[float]]:
        """
        Returns the bounding box of the shape as a tuple (min_x, min_y, max_x, max_y).
        """
        raise NotImplementedError(
            "Bounding box method not implemented for this shape."
        )


class Shapes2DArray(ABC):
    pass


# -------------------------------------------------------------------------


class Ellipse(Shape2D):
    __slots__ = (
        "_semi_major_length",
        "_semi_minor_length",
        "_position",
        "_theta_start",
        "_theta_end",
    )

    def __init__(
        self,
        semi_major_length: float,
        semi_minor_length: float,
        centre: tuple[float, float] = (0.0, 0.0),
        major_axis_angle: float = 0.0,
        theta_start: float = 0.0,
        theta_end: float = np.pi * 2.0,
    ):
        super().__init__()
        _args = self._validate_args(
            semi_major_length,
            semi_minor_length,
            centre,
            major_axis_angle,
            theta_start,
            theta_end,
        )
        self._semi_major_length = _args["semi_major_length"]
        self._semi_minor_length = _args["semi_minor_length"]
        self._position = Shape2DPosition(
            *_args["centre"], _args["major_axis_angle"]
        )
        self._theta_start = _args["theta_start"]
        self._theta_end = _args["theta_end"]

    def _validate_args(
        self,
        semi_major_length,
        semi_minor_length,
        centre,
        major_axis_angle,
        theta_start,
        theta_end,
    ):
        semi_major_length = _validate_positive_float(
            semi_major_length, "semi_major_length"
        )
        semi_minor_length = _validate_float(
            semi_minor_length, "semi_major_length"
        )
        if semi_major_length < semi_minor_length:
            raise ValueError("Semi-major axis must be >= semi-minor axis")
        centre = _validate_float(centre[0]), _validate_float(centre[1])
        major_axis_angle = _validate_float(major_axis_angle)
        theta_start = _validate_float(theta_start, low=0.0, high=np.pi * 2.0)
        theta_end = _validate_float(theta_end, low=0.0, high=np.pi * 2.0)
        if theta_start >= theta_end:
            raise ValueError("theta_start must be < theta_end")
        return dict(
            semi_major_length=semi_major_length,
            semi_minor_length=semi_minor_length,
            centre=centre,
            major_axis_angle=major_axis_angle,
            theta_start=theta_start,
            theta_end=theta_end,
        )

    @property
    def semi_major_length(self):
        return self._semi_major_length

    @property
    def semi_minor_length(self):
        return self._semi_minor_length

    @property
    def aspect_ratio(self) -> float:
        return self._semi_major_length / self._semi_minor_length

    @property
    def eccentricity(self) -> float:
        return np.sqrt(
            1 - ((self._semi_minor_length / self._semi_major_length) ** 2)
        )

    @property
    def position(self) -> Shape2DPosition:
        return self._position

    @property
    def theta_start(self) -> float:
        return self._theta_start

    @property
    def theta_end(self) -> float:
        return self._theta_end

    @property
    def area(self) -> float:
        return (
            0.5
            * self._semi_major_length
            * self._semi_minor_length
            * (self._theta_end - self._theta_start)
        )

    def _arc_length_integrand(self, t: float) -> float:
        return np.hypot(
            self._semi_major_length * np.sin(t),
            self._semi_minor_length * np.cos(t),
        )

    @property
    def perimeter(self):
        """Perimeter of the ellipse along the arc connecting the two endpoints."""
        return quad(
            self._arc_length_integrand, self._theta_start, self._theta_end
        )[0]

    def sample_points(
        self,
        num_points: int | None = None,
        point_density: float = 10.0,
    ) -> np.ndarray:
        """Samples points along the elliptical arc."""
        num_points = _validate_positive_int(
            num_points or max(16, int(point_density * self.perimeter)),
            name="num_points",
        )

        return self.points_at_parametric_points(
            np.linspace(self._theta_start, self._theta_end, num_points)
        )

    def points_at_parametric_points(self, theta: Sequence) -> PointArray2D:
        points_arr = PointArray2D.from_named_dims(
            x=self._semi_major_length * np.cos(theta),
            y=self._semi_minor_length * np.sin(theta),
        )
        points_arr.transform(
            self._position.theta,
            self._position.x,
            self._position.y,
            in_place=True,
            pivot=(0.0, 0.0),
            order="rotation_then_translation",
        )
        if not isinstance(points_arr, PointArray2D) or len(points_arr) == 0:
            raise ValueError(
                "Invalid points array or no points sampled along the arc"
            )

        return points_arr

    def point_at_angle(self, theta: float) -> Point2D:
        """Returns the point on the ellipse at the given angle."""
        x = self.semi_major_length * np.cos(theta)
        y = self.semi_minor_length * np.sin(theta)
        return Point2D(x, y).transform(
            self._position.theta, self._position.x, self._position.y
        )


class Circle(Ellipse):
    pass


class Rectangle(Shape2D):
    pass


class CirclesArray(Shapes2DArray):
    pass
