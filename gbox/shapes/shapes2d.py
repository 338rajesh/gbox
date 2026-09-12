import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, Literal

from scipy.integrate import quad

from ..core.utils import (
    _validate_positive_float,
    _validate_float,
    _validate_positive_int,
    _validate_dict,
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

    def transform(self, rot_angle, dx, dy, in_place=False):
        if in_place:
            self.x += dx
            self.y += dy
            self.theta += rot_angle
            return self
        return Shape2DPosition(self.x + dx, self.y + dy, self.theta + rot_angle)


class Shape2D(ABC):
    @property
    @abstractmethod
    def position(self) -> Shape2DPosition:
        pass

    @property
    @abstractmethod
    def area(self) -> float:
        pass

    @property
    @abstractmethod
    def perimeter(self) -> float:
        pass

    @property
    def equivalent_circle_radius(self) -> float:
        return float(np.sqrt(self.area / np.pi))

    @abstractmethod
    def contains_point(self, point) -> bool:
        pass

    @abstractmethod
    def union_of_circles(self):
        pass

    def bounding_box(self) -> tuple[list[float], list[float]]:
        """
        Returns the bounding box of the shape as a tuple (min_x, min_y, max_x, max_y).
        """
        raise NotImplementedError("Bounding box method not implemented for this shape.")


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
        self._semi_major_length: float = _args["semi_major_length"]
        self._semi_minor_length: float = _args["semi_minor_length"]
        self._position: Shape2DPosition = Shape2DPosition(
            *_args["centre"], _args["major_axis_angle"]
        )
        self._theta_start: float = _args["theta_start"]
        self._theta_end: float = _args["theta_end"]

    def _validate_args(
        self,
        semi_major_length,
        semi_minor_length,
        centre,
        major_axis_angle,
        theta_start,
        theta_end,
    ) -> dict:
        semi_major_length = _validate_positive_float(
            semi_major_length, "semi_major_length"
        )
        semi_minor_length = _validate_positive_float(
            semi_minor_length, "semi_minor_length"
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
        return np.sqrt(1 - ((self._semi_minor_length / self._semi_major_length) ** 2))

    @property
    def position(self) -> Shape2DPosition:
        return self._position

    @property
    def centre(self) -> tuple[float, float]:
        return (self._position.x, self._position.y)

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
    def perimeter(self) -> float:
        """Perimeter of the ellipse along the arc connecting the two endpoints."""
        return quad(self._arc_length_integrand, self._theta_start, self._theta_end)[0]

    @property
    def bounding_box(self) -> list[float]:
        """Returns the axis-aligned bounding box of the ellipse as
        a list of [min_x, min_y, max_x, max_y].
        """
        a2 = self._semi_major_length**2
        b2 = self._semi_minor_length**2
        cos_2 = np.cos(self._position.theta) ** 2
        sin_2 = np.sin(self._position.theta) ** 2
        hx = np.sqrt(a2 * cos_2 + b2 * sin_2)
        hy = np.sqrt(a2 * sin_2 + b2 * cos_2)
        cx, cy = self._position.x, self._position.y
        return [cx - hx, cy - hy, cx + hx, cy + hy]

    def clone(self) -> Self:
        return self.__class__(
            self._semi_major_length,
            self._semi_minor_length,
            self.centre,
            self._position.theta,
            self._theta_start,
            self._theta_end,
        )

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
            raise ValueError("Invalid points array or no points sampled along the arc")

        return points_arr

    def point_at_angle(self, theta: float) -> Point2D:
        """Returns the point on the ellipse at the given angle."""
        x = self.semi_major_length * np.cos(theta)
        y = self.semi_minor_length * np.sin(theta)
        return Point2D(x, y).transform(
            self._position.theta, self._position.x, self._position.y
        )

    def _origin_to_position(self, pose: Shape2DPosition) -> Self:
        """It transforms the ellipse from the origin to the specified position."""
        self._position.transform(pose.theta, pose.x, pose.y, in_place=True)
        return self

    @classmethod
    def from_params(
        cls,
        position_params: dict[str, float],
        size_params: dict[str, float],
    ) -> Self:
        """
        Construct an Ellipse from parameter dictionaries.

        Parameters
        ----------
        position_params : dict
            - 'xc': x-coordinate of the centre.
            - 'yc': y-coordinate of the centre.
            - 'major_axis_angle': Rotation angle in radians.
        size_params : dict
            - 'semi_major_length': Half-length of the primary axis.
            - 'semi_minor_length': Half-length of the secondary axis.
            - 'theta_start': Starting angle in radians, defaults to 0.
            - 'theta_end': Ending angle in radians, defaults to 2 * pi.

        Returns
        -------
        Ellipse
        """
        _validate_dict(
            position_params,
            ["xc", "yc", "major_axis_angle"],
            [float, float, float],
            name="position_params",
        )
        _validate_dict(
            size_params,
            ["semi_major_length", "semi_minor_length"],
            [float, float],
            name="size_params",
        )
        return cls(
            size_params["semi_major_length"],
            size_params["semi_minor_length"],
            (position_params["xc"], position_params["yc"]),
            position_params["major_axis_angle"],
            size_params.get("theta_start", 0.0),
            size_params.get("theta_end", np.pi * 2.0),
        )

    def contains_point(
        self, p: Point2D | Sequence[float], atol: float = 1e-6
    ) -> Literal[-1, 0, 1]:
        """Checks if a point is inside, on, or outside the ellipse.

        Parameters
        ----------
        p : Point2D
            Point to check.
        atol : float, optional. Essential a bandwidth of 2*atol around the
            ellipse is checked. Default 1e-6

        Returns
        -------
        -1 : point is outside the ellipse
        0 : point is on the ellipse
        1 : point is inside the ellipse

        """
        p = Point2D(p[0], p[1]).transform(
            -self._position.theta,
            -self._position.x,
            -self._position.y,
            order="translate_then_rotate",
        )
        val = (p.x**2 / self._semi_major_length**2) + (
            p.y**2 / self._semi_minor_length**2
        )
        if val > 1.0 + atol:
            return -1
        if val < 1.0 - atol:
            return 1
        return 0

    def r_shortest(self, xi: float) -> float:
        """Evaluates the shortest distance to the ellipse locus
        from a point on the major axis located at a distance ``xi``
        from the centre of the ellipse. This distance is the radius
        of the circle that is tangent to the ellipse at the point,
        assuming that the ellipse is centered at the origin and its
        major axis is aligned with the x-axis.

        Parameters
        ----------
        xi : float
            The x-coordinate of the point on the major axis from which
            the shortest distance to the ellipse is calculated.

        Returns
        -------
        float
            The shortest distance (radius) from the point on the major axis
            to the ellipse locus.

        Notes
        -----
        The formula used to calculate the shortest distance is derived from
        the equation of the ellipse and the geometry of the situation.

        $$r_{min} = b \\sqrt{1 - \\frac{x_i^2}{a^2 - b^2}}$$

        """
        if self._semi_major_length == self._semi_minor_length:
            return self.semi_minor_length

        r_min = self.semi_minor_length * np.sqrt(
            1.0 - ((xi * xi) / (self.semi_major_length**2 - self.semi_minor_length**2))
        )
        return float(r_min)

    def union_of_circles(self, dh: float = 0.05) -> list:
        """
        Approximates the ellipse as a union of circles along its major axis.

        Parameters
        ----------
        dh : float, optional
            The buffer thickness for the outer ellipse used to determine the
            spacing of the circles. Must be greater than 0. Default is 0.05

        Returns
        -------
        list of Circle
            A list of Circle objects that approximate the ellipse.

        """
        if self.aspect_ratio == 1.0:
            return [Circle(self.semi_major_length, self.centre)]

        if dh <= 0.0:
            raise ValueError(
                "buffer thickness dh must be > 0 for union_of_circles to converge"
            )
        _validate_float(
            dh,
            low=0.0,
            high=self._semi_minor_length,
            name="buffer thickness dh",
            closed_bounds=False,
        )

        ell_outer = Ellipse(
            self._semi_major_length * (1.0 + dh),
            self._semi_minor_length * (1.0 + dh),
            (self._position.x, self._position.y),
            self._position.theta,
        )
        e_i: float = self.eccentricity
        e_o: float = ell_outer.eccentricity
        m: float = 2.0 * e_o * e_o / (e_i * e_i)

        def min_radius() -> float:  # r_min : b^2/a
            r_min = (
                self.semi_minor_length**2
            ) / self.semi_major_length  # r_min : b^2/a
            return float(r_min)

        x_max = self.semi_major_length * e_i * e_i  # x range: (-ae^2, ae^2)
        r_min = min_radius()
        x_i = -1.0 * x_max  # start at x = -ae^2

        # construct circles at origin
        circles: list[Circle] = []
        while True:
            if x_i > x_max:
                circles.append(Circle(r_min, (x_max, 0.0)))
                break
            r_i = self.r_shortest(x_i)
            circles.append(Circle(r_i, (x_i, 0.0)))

            r_o = ell_outer.r_shortest(x_i)
            gap = max(r_o * r_o - r_i * r_i, 0.0)  # gap between circles
            x_i = (x_i * (m - 1.0)) + (m * e_i * np.sqrt(gap))

        circles_array = [c._origin_to_position(self._position) for c in circles]
        return circles_array


class Circle(Ellipse):
    def __init__(
        self,
        radius: float,
        centre: tuple[float, float] = (0, 0),
        theta_start=0,
        theta_end=np.pi * 2,
    ):
        super().__init__(
            semi_major_length=radius,
            semi_minor_length=radius,
            centre=centre,
            theta_start=theta_start,
            theta_end=theta_end,
        )

    @property
    def radius(self) -> float:
        return self._semi_major_length

    def clone(self) -> Self:
        return self.__class__(
            self._semi_major_length, self.centre, self._theta_start, self._theta_end
        )


class CirclesArray(Shapes2DArray):
    """A vectorized collection of circles that can be moved and rotated
    together as a single rigid group.
    """

    __slots__ = ("_centres", "_radii")

    def __init__(
        self,
        centres: PointArray2D | Sequence[tuple[float, float]],
        radii: Sequence[float] | float,
    ):
        if not isinstance(centres, PointArray2D):
            centres = PointArray2D.from_named_dims(
                x=np.asarray([c[0] for c in centres], dtype=float),
                y=np.asarray([c[1] for c in centres], dtype=float),
            )
        radii = np.asarray(radii, dtype=float)
        if radii.ndim == 0:
            radii = np.full(len(centres), float(radii))
        if len(radii) != len(centres):
            raise ValueError("radii length must match number of centres")
        if np.any(radii <= 0):
            raise ValueError("all radii must be positive")

        self._centres = centres
        self._radii = radii

    def __len__(self) -> int:
        return len(self._radii)

    @property
    def centres(self) -> PointArray2D:
        return self._centres

    @property
    def radii(self) -> np.ndarray:
        return self._radii

    @classmethod
    def from_circles(cls, circles: Sequence[Circle]) -> Self:
        centres = [(c.position.x, c.position.y) for c in circles]
        radii = [c.radius for c in circles]
        return cls(centres, radii)

    def to_circles(self) -> list[Circle]:
        return [
            Circle(float(r), (float(p.x), float(p.y)))
            for p, r in zip(self._centres, self._radii)
        ]

    def clone(self) -> Self:
        return self.__class__(self._centres.clone(), self._radii.copy())

    def translate(self, dx: float, dy: float, in_place: bool = False) -> Self:
        """Moves every circle in the group by the same (dx, dy)."""
        target = self if in_place else self.clone()
        target._centres.transform(
            0.0,
            dx,
            dy,
            in_place=True,
            pivot=(0.0, 0.0),
            order="rotation_then_translation",
        )
        return target

    def rotate(
        self,
        rot_angle: float,
        pivot: tuple[float, float] = (0.0, 0.0),
        in_place: bool = False,
    ) -> Self:
        """Rotates every circle's centre about `pivot` by `rot_angle`.
        Circle radii are unaffected by rotation.
        """
        target = self if in_place else self.clone()
        target._centres.transform(
            rot_angle,
            0.0,
            0.0,
            in_place=True,
            pivot=pivot,
            order="rotation_then_translation",
        )
        return target

    def transform(
        self,
        rot_angle: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        pivot: tuple[float, float] = (0.0, 0.0),
        in_place: bool = False,
    ) -> Self:
        """Combined rotate-then-translate of the whole group, rotating about
        `pivot`.
        """
        target = self if in_place else self.clone()
        target._centres.transform(
            rot_angle,
            dx,
            dy,
            in_place=True,
            pivot=pivot,
            order="rotation_then_translation",
        )
        return target

    def total_area(self) -> float:
        return float(np.sum(np.pi * self._radii**2))

    def bounding_box(self) -> list[float]:
        xs, ys = self._centres.x, self._centres.y
        return [
            float(np.min(xs - self._radii)),
            float(np.min(ys - self._radii)),
            float(np.max(xs + self._radii)),
            float(np.max(ys + self._radii)),
        ]

