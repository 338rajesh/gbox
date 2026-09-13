from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from ..core.utils import (
    Angle,
    TransformationOrder,
    _validate_positive_float,
    _validate_float,
    _validate_positive_int,
    _validate_dict,
)
from ..core.points import Point2D, PointArray2D
from ..core.transformation import transform_point_2d

PI = float(np.pi)


@dataclass(frozen=True, slots=True)
class Shape2DPose:
    """
    A triplet of (x, y, orientation) where orientation is in radians
    measured in the counter-clockwise direction from the positive x-axis.

    NOTE: This class is immutable. Use the `rotate`, `translate`, and `transform`
    methods to create new instances with modified values.
    """

    x: float
    y: float
    orientation: Angle

    def __repr__(self):
        return f"Shape2DPose(x={self.x}, y={self.y}, orientation={self.orientation})"

    def rotate(self, rot_angle: Angle) -> Self:
        """Returns a new Shape2DPose with the same (x, y) but a new orientation."""
        return Shape2DPose(self.x, self.y, self.orientation + rot_angle)

    def translate(self, dx: float, dy: float) -> Self:
        """Returns a new Shape2DPose with the same orientation but a new (x, y)."""
        return Shape2DPose(self.x + dx, self.y + dy, self.orientation)

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        rot_angle: Angle = Angle.rad(0.0),
        pivot: tuple[float, float] = None,
        order: TransformationOrder = TransformationOrder.ROTATE_THEN_TRANSLATE,
    ) -> Self:
        """Returns a new Shape2DPose with a new (x, y) and orientation."""
        if pivot is None:
            pivot = self.x, self.y
        new_x, new_y = transform_point_2d(
            self.x,
            self.y,
            dx=dx,
            dy=dy,
            angle=rot_angle,
            pivot=pivot,
            order=order,
        )
        return Shape2DPose(new_x, new_y, self.orientation + rot_angle)


class Shape2D(ABC):
    @property
    @abstractmethod
    def position(self) -> Shape2DPose:
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
        return float(np.sqrt(self.area / PI))

    @abstractmethod
    def contains_point(self, point: Point2D | Sequence[float]) -> Literal[-1, 0, 1]:
        """
        Checks if a point is inside, on, or outside the shape.

        Returns
        -------
        `-1` : point is outside the shape
        `0` : point is on the shape
        `1` : point is inside the shape
        """
        pass

    @abstractmethod
    def union_of_circles(self):
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> list[float]:
        """
        Returns the bounding box of the shape as a list of
        four floats: [min_x, min_y, max_x, max_y].
        """
        raise NotImplementedError("Bounding box method not implemented for this shape.")


class Shapes2DArray(ABC):
    pass


# -------------------------------------------------------------------------


class Ellipse(Shape2D):
    __slots__ = ("_semi_major_length", "_semi_minor_length", "_position")

    def __init__(
        self,
        semi_major_length: float,
        semi_minor_length: float,
        centre: tuple[float, float] = (0.0, 0.0),
        major_axis_angle: Angle = Angle.rad(0.0),
    ):
        super().__init__()
        _args = self._validate_args(
            semi_major_length, semi_minor_length, centre, major_axis_angle
        )
        self._semi_major_length: float = _args["semi_major_length"]
        self._semi_minor_length: float = _args["semi_minor_length"]
        self._position: Shape2DPose = Shape2DPose(
            _args["centre"][0], _args["centre"][1], _args["major_axis_angle"]
        )

    def _validate_args(
        self, semi_major_length, semi_minor_length, centre, major_axis_angle
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
        return dict(
            semi_major_length=semi_major_length,
            semi_minor_length=semi_minor_length,
            centre=centre,
            major_axis_angle=major_axis_angle,
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
    def position(self) -> Shape2DPose:
        return self._position

    @property
    def centre(self) -> tuple[float, float]:
        return (self._position.x, self._position.y)

    @property
    def area(self) -> float:
        return PI * self._semi_major_length * self._semi_minor_length

    def _arc_length_integrand(self, t: float) -> float:
        return np.hypot(
            self._semi_major_length * np.sin(t),
            self._semi_minor_length * np.cos(t),
        )

    @property
    def perimeter(self) -> float:
        """Perimeter of the ellipse along the arc connecting the two endpoints."""
        return quad(self._arc_length_integrand, 0, 2 * PI)[0]

    @property
    def bounding_box(self) -> list[float]:
        """Returns the axis-aligned bounding box of the ellipse as
        a list of [min_x, min_y, max_x, max_y].
        """
        a2 = self._semi_major_length**2
        b2 = self._semi_minor_length**2
        cos_2 = self._position.orientation.cos**2
        sin_2 = self._position.orientation.sin**2
        hx = np.sqrt(a2 * cos_2 + b2 * sin_2)
        hy = np.sqrt(a2 * sin_2 + b2 * cos_2)
        cx, cy = self._position.x, self._position.y
        return [cx - hx, cy - hy, cx + hx, cy + hy]

    def clone(self) -> Self:
        return self.__class__(
            self._semi_major_length,
            self._semi_minor_length,
            self.centre,
            self._position.orientation,
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

        return self.points_at_parametric_points(np.linspace(0, 2 * PI, num_points))

    def points_at_parametric_points(self, theta: Sequence) -> PointArray2D:
        points_arr = PointArray2D.from_named_dims(
            x=self._semi_major_length * np.cos(theta),
            y=self._semi_minor_length * np.sin(theta),
        )
        points_arr.transform(
            self._position.x,
            self._position.y,
            self._position.orientation,
            in_place=True,
        )
        if not isinstance(points_arr, PointArray2D) or len(points_arr) == 0:
            raise ValueError("Invalid points array or no points sampled along the arc")

        return points_arr

    def point_at_angle(self, theta: float) -> Point2D:
        """Returns the point on the ellipse at the given angle."""
        x = self.semi_major_length * np.cos(theta)
        y = self.semi_minor_length * np.sin(theta)
        return Point2D(x, y).transform(
            self._position.x, self._position.y, self._position.orientation
        )

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        d_theta: Angle = Angle.rad(0.0),
        *,
        pivot: tuple[float, float] = None,
        order: TransformationOrder = TransformationOrder.ROTATE_THEN_TRANSLATE,
    ) -> Self:
        """Returns a new Ellipse whose position is transformed
        by the given translation and rotation.

        Parameters
        ----------
        dx : float
            Translation in the x-direction.
        dy : float
            Translation in the y-direction.
        d_theta : Angle
            Rotation angle in radians. See :class:`gbox.core.utils.Angle` for more details.
        pivot : tuple[float, float], optional
            The pivot point for rotation. Default is the origin (0, 0).
        order : TransformationOrder, optional
            The order of transformations. Default is ROTATE_THEN_TRANSLATE.
            see :class:`gbox.core.utils.TransformationOrder` for more details.
        """
        if pivot is None:
            pivot = self._position.x, self._position.y

        new_position = self._position.transform(
            dx, dy, d_theta, pivot=pivot, order=order
        )
        return self.__class__(
            self._semi_major_length,
            self._semi_minor_length,
            (new_position.x, new_position.y),
            new_position.orientation,
        )

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

        Returns
        -------
        Ellipse
        """
        _validate_dict(
            position_params,
            ["xc", "yc", "major_axis_angle"],
            [float, float, Angle],
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
        p = Point2D.from_sequence(p)

        dx = p.x - self._position.x
        dy = p.y - self._position.y

        c = self._position.orientation.cos
        s = self._position.orientation.sin

        x = c * dx + s * dy
        y = -s * dx + c * dy

        val = (
            (x / self._semi_major_length) ** 2
            + (y / self._semi_minor_length) ** 2
            - 1.0
        )
        if val > atol:
            return -1
        if val < -atol:
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
            self._position.orientation,
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

        circles_array = [
            c.transform(self._position.x, self._position.y, self._position.orientation)
            for c in circles
        ]
        return circles_array


class Circle(Ellipse):
    def __init__(
        self,
        radius: float,
        centre: tuple[float, float] = (0, 0),
    ):
        super().__init__(
            semi_major_length=radius, semi_minor_length=radius, centre=centre
        )

    def transform(
        self,
        dx=0,
        dy=0,
        d_theta=Angle.rad(0),
        *,
        pivot=None,
        order=TransformationOrder.ROTATE_THEN_TRANSLATE,
    ):
        if pivot is None:
            pivot = self._position.x, self._position.y

        new_position = self._position.transform(
            dx, dy, d_theta, pivot=pivot, order=order
        )
        return self.__class__(self._semi_major_length, (new_position.x, new_position.y))

    @property
    def radius(self) -> float:
        return self._semi_major_length

    def clone(self) -> Self:
        return self.__class__(self._semi_major_length, self.centre)


class CirclesArray(Shapes2DArray):
    """A vectorized collection of circles that can be moved and rotated
    together as a single rigid group.
    """

    __slots__ = ("_centres", "_radii")

    def __init__(
        self,
        centres: PointArray2D | Sequence[tuple[float, float]] | npt.NDArray[np.float64],
        radii: Sequence[float] | float | npt.NDArray[np.float64],
    ):
        if isinstance(centres, np.ndarray):
            if centres.ndim != 2 or centres.shape[1] != 2:
                raise ValueError(
                    "centres must be a 2D array with shape (N, 2) for N circles"
                )
            centres = PointArray2D.from_named_dims(
                x=centres[:, 0].astype(float), y=centres[:, 1].astype(float)
            )

        if not isinstance(centres, PointArray2D):
            centres = PointArray2D.from_named_dims(
                x=np.asarray([c[0] for c in centres], dtype=float),
                y=np.asarray([c[1] for c in centres], dtype=float),
            )

        if isinstance(radii, (float, int)):
            radii = [float(radii)] * len(centres)
        radii = np.asarray(radii, dtype=float)
        if radii.ndim > 1:
            radii = radii.squeeze()
        if radii.ndim != 1:
            raise ValueError(
                f"radii must be a 1D array or a single float, got {radii.ndim}D"
            )
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
            Circle(float(r), (float(p[0]), float(p[1])))
            for p, r in zip(self._centres, self._radii)
        ]

    def clone(self) -> Self:
        return self.__class__(self._centres.copy(), self._radii.copy())

    def translate(self, dx: float, dy: float, in_place: bool = False) -> Self:
        """Moves every circle in the group by the same (dx, dy)."""
        target = self if in_place else self.clone()
        target._centres.transform(
            dx,
            dy,
            Angle.rad(0.0),
            in_place=True,
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )
        return target

    def rotate(
        self,
        rot_angle: Angle = Angle.rad(0.0),
        pivot: tuple[float, float] = (0.0, 0.0),
        in_place: bool = False,
    ) -> Self:
        """Rotates every circle's centre about `pivot` by `rot_angle`.
        Circle radii are unaffected by rotation.
        """
        target = self if in_place else self.clone()
        target._centres.transform(
            0.0,
            0.0,
            rot_angle,
            in_place=True,
            pivot=pivot,
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )
        return target

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        rot_angle: Angle = Angle.rad(0.0),
        pivot: tuple[float, float] = None,
        in_place: bool = False,
        order: TransformationOrder = TransformationOrder.ROTATE_THEN_TRANSLATE,
    ) -> Self:
        """Combined rotate-then-translate of the whole group, rotating about
        `pivot`.
        """
        target = self if in_place else self.clone()
        if pivot is None:
            pivot = Point2D(0.0, 0.0)
        target._centres.transform(
            dx,
            dy,
            rot_angle,
            in_place=True,
            pivot=pivot,
            order=order,
        )
        return target

    def bounding_box(self) -> list[float]:
        xs, ys = self._centres.x, self._centres.y
        return [
            float(np.min(xs - self._radii)),
            float(np.min(ys - self._radii)),
            float(np.max(xs + self._radii)),
            float(np.max(ys + self._radii)),
        ]
