from functools import lru_cache
from typing import Literal, Optional, Tuple, Union, List
from numpy.typing import NDArray

import numpy as np
from scipy.integrate import quad
from .base import (
    Point2D,
    PointArray2D,
    DEFAULT_FLOAT,
    PI,
    FloatType,
    BoundingBox,
)

TWO_PI = 2.0 * PI
DefaultFloatType = type(DEFAULT_FLOAT())
EPSILON = np.finfo(DEFAULT_FLOAT).eps

# TODO add magic methods for comparison, hashing, etc.

# region EllipticalArc


class EllipticalArc:
    __slots__ = (
        "semi_major",
        "semi_minor",
        "centre",
        "major_axis_angle",
        "theta_start",
        "theta_end",
        "aspect_ratio",
        "eccentricity",
        "_end_point_1",
        "_end_point_2",
    )

    def __init__(
        self,
        semi_major: FloatType,
        semi_minor: FloatType,
        centre: Tuple[FloatType, FloatType] | Point2D = (0.0, 0.0),
        major_axis_angle: FloatType = 0.0,
        theta_start: FloatType = 0.0,
        theta_end: FloatType = TWO_PI,
    ):
        self.centre = Point2D(centre[0], centre[1])
        self.semi_major = semi_major
        self.semi_minor = semi_minor
        self.major_axis_angle = major_axis_angle
        self.theta_start = theta_start
        self.theta_end = theta_end

        self._post_init_()

    def _post_init_(self):
        if self.semi_minor <= 0.0:
            raise ValueError("Semi-minor axis must be positive")
        if self.semi_major < self.semi_minor:
            raise ValueError("Semi-major axis must be >= semi-minor axis")

        self.aspect_ratio = self.semi_major / self.semi_minor
        self.eccentricity = np.sqrt(
            1 - ((self.semi_minor / self.semi_major) ** 2)
        )

        self._end_point_1 = self.point_at_angle(self.theta_start)
        self._end_point_2 = self.point_at_angle(self.theta_end)

    @property
    def area(self) -> DEFAULT_FLOAT:
        area = (
            0.5
            * self.semi_major
            * self.semi_minor
            * (self.theta_end - self.theta_start)
        )
        return DEFAULT_FLOAT(area)

    def perimeter(
        self, closure: Literal["open", "e2e", "ece"] = "open"
    ) -> DEFAULT_FLOAT:
        """
        Computes the perimeter of the elliptical arc.

        Parameters
        ----------
        closure : str
            Type of closure for the arc. Options are:
            - "open": only the arc length is considered.
            - "e2e": the arc length and the chord length between
            the two endpoints are considered.
            - "ece": the arc length and the chord lengths from the
            endpoints to the centre are considered.
        """
        arc_length, _ = quad(
            self._arc_len_integrand, self.theta_start, self.theta_end
        )
        if closure not in ("open", "e2e", "ece"):
            raise ValueError(
                f"Invalid closure type: {closure}. "
                "Must be 'open', 'e2e', or 'ece'."
            )
        if closure == "open":
            perimeter = arc_length
        elif closure == "e2e":
            chord_len = self._end_point_1.distance_to(self._end_point_2)
            perimeter = arc_length + chord_len
        elif closure == "ece":
            chord_len_1 = self._end_point_1.distance_to(self.centre)
            chord_len_2 = self._end_point_2.distance_to(self.centre)
            perimeter = arc_length + chord_len_1 + chord_len_2
        return DEFAULT_FLOAT(perimeter)

    def _arc_len_integrand(self, theta: FloatType) -> FloatType:
        return np.hypot(
            self.semi_major * np.sin(theta), self.semi_minor * np.cos(theta)
        )

    def sample_points(
        self,
        num_points: Optional[int] = None,
        point_density: FloatType = 10.0,
    ) -> PointArray2D:
        """Samples points along the elliptical arc."""
        if num_points is None:
            num_points = max(16, int(point_density * self.perimeter()))

        theta = np.linspace(self.theta_start, self.theta_end, num_points)
        points = self.points_at_parametric_points(theta)
        if points is None:
            raise ValueError("Invalid points array")
        if len(points) == 0:
            raise ValueError("No points sampled along the arc")
        return points

    def point_at_angle(self, theta: FloatType) -> Point2D:
        """Returns the point on the ellipse at the given angle."""
        x = self.semi_major * np.cos(theta)
        y = self.semi_minor * np.sin(theta)
        return Point2D(x, y).transform(
            self.major_axis_angle, self.centre.x, self.centre.y
        )

    def points_at_parametric_points(self, theta: np.ndarray) -> PointArray2D:
        points = np.column_stack(
            (self.semi_major * np.cos(theta), self.semi_minor * np.sin(theta))
        )
        points_arr = PointArray2D(points)

        points_arr.transform(
            self.major_axis_angle,
            self.centre.x,
            self.centre.y,
            in_place=True,
            pivot=(0.0, 0.0),
            order="RT",  # Rotate >> Translate
        )
        if points_arr is None:
            raise ValueError("Invalid points array")
        return points_arr

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError(
            "Bounding box for EllipticalArc is not implemented yet."
        )


# endregion EllipticalArc
# region Ellipse


class Ellipse:
    __slots__ = (
        "arc",
        "_boundary_points",
        "_area",
        "_perimeter",
    )

    def __init__(
        self,
        semi_major: FloatType,
        semi_minor: FloatType,
        centre: Tuple[FloatType, FloatType] | Point2D = (0.0, 0.0),
        major_axis_angle: FloatType = 0.0,
    ):
        """
        Closed ellipse shape.

        Parameters
        ----------
        semi_major : FloatType
            Semi-major axis length.
        semi_minor : FloatType
            Semi-minor axis length.
        centre : Tuple[FloatType, FloatType]
            Centre of the ellipse in (x, y) coordinates.
        major_axis_angle : FloatType
            Angle of the major axis in radians.
        """
        self.arc = EllipticalArc(
            semi_major,
            semi_minor,
            centre,
            major_axis_angle,
            0.0,
            TWO_PI,
        )
        self._boundary_points = None

    def copy(self):
        return self.__class__(
            self.arc.semi_major,
            self.arc.semi_minor,
            self.arc.centre,
            self.arc.major_axis_angle,
        )

    @property
    def semi_major(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(self.arc.semi_major)

    @property
    def semi_minor(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(self.arc.semi_minor)

    @property
    def centre(self) -> Point2D:
        return self.arc.centre

    @property
    def major_axis_angle(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(self.arc.major_axis_angle)

    @property
    def aspect_ratio(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(self.arc.aspect_ratio)

    @property
    def eccentricity(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(self.arc.eccentricity)

    @property
    @lru_cache(maxsize=1)
    def area(self) -> DEFAULT_FLOAT:
        """Area of the ellipse."""
        if hasattr(self, "_area"):
            return self._area
        else:
            self._area = DEFAULT_FLOAT(PI * self.semi_major * self.semi_minor)
            return self._area

    @property
    @lru_cache(maxsize=1)
    def perimeter(self) -> DEFAULT_FLOAT:
        """Perimeter of the ellipse."""
        if hasattr(self, "_perimeter"):
            return self._perimeter
        else:
            self._perimeter = self.arc.perimeter()
            return self._perimeter

    def eval_boundary_points(
        self, num_points: Optional[int] = None, point_density: FloatType = 10.0
    ) -> None:
        """Evaluate boundary points of the ellipse."""
        if self._boundary_points is None:
            self._boundary_points = self.arc.sample_points(
                num_points, point_density
            )

    def get_boundary_points(
        self, num_points: int = 100, point_density: FloatType = 10.0
    ) -> PointArray2D:
        """Returns the boundary points of the ellipse."""
        self.eval_boundary_points(num_points, point_density)
        if self._boundary_points is None:
            raise ValueError("Boundary points is not available")
        return self._boundary_points

    def get_bounding_box(self) -> BoundingBox:
        a2, b2 = self.semi_major**2, self.semi_minor**2
        cos_2 = np.cos(self.major_axis_angle) ** 2
        sin_2 = np.sin(self.major_axis_angle) ** 2
        hx = np.sqrt(a2 * cos_2 + b2 * sin_2)
        hy = np.sqrt(a2 * sin_2 + b2 * cos_2)
        x_min = self.centre.x - hx
        x_max = self.centre.x + hx
        y_min = self.centre.y - hy
        y_max = self.centre.y + hy
        return BoundingBox([x_min, y_min], [x_max, y_max])

    def contains(
        self, p: Point2D | Tuple[FloatType, FloatType], atol: FloatType = 1e-6
    ) -> Literal[-1, 0, 1]:
        """Checks if a point is inside, on, or outside the ellipse.

        Returns
        -------
        -1 : point is outside the ellipse
        0 : point is on the ellipse
        1 : point is inside the ellipse

        """
        p = Point2D(p[0], p[1]).transform(
            -self.major_axis_angle,
            -self.centre.x,
            -self.centre.y,
            order="TR",  # Translate >> Rotate as we are moving backwards
        )
        val = (p.x**2 / self.semi_major**2) + (p.y**2 / self.semi_minor**2)
        if val > 1.0 + atol:
            return -1
        if val < 1.0 - atol:
            return 1
        return 0

    def r_shortest(self, xi: FloatType) -> DEFAULT_FLOAT:
        """Evaluates the shortest distance to the ellipse locus
        from a point on the major axis located at a distance xi
        from the centre of the ellipse.
        """
        if self.semi_major == self.semi_minor:
            return DEFAULT_FLOAT(self.semi_minor)
        else:
            r_min = self.semi_minor * np.sqrt(
                1.0 - ((xi * xi) / (self.semi_major**2 - self.semi_minor**2))
            )
            return DEFAULT_FLOAT(r_min)

    def union_of_circles(self, dh: FloatType = 0.0) -> "CirclesArray":
        # raise NotImplementedError("uns is not implemented")
        if self.aspect_ratio == 1.0:
            return CirclesArray([Circle(self.semi_major, self.centre)])

        assert dh >= 0, f"Expecting buffer dh >= 0, but got {dh}."

        ell_outer = Ellipse(
            self.semi_major + dh,
            self.semi_minor + dh,
            self.centre,
            self.major_axis_angle,
        )
        e_i: FloatType = self.eccentricity
        e_o: FloatType = ell_outer.eccentricity
        m: FloatType = 2.0 * e_o * e_o / (e_i * e_i)

        def min_radius() -> FloatType:  # r_min : b^2/a
            r_min = (self.semi_minor**2) / self.semi_major  # r_min : b^2/a
            return DEFAULT_FLOAT(r_min)

        x_max = self.semi_major * e_i * e_i  # x range: (-ae^2, ae^2)
        r_min = min_radius()
        x_i = -1.0 * x_max  # start at x = -ae^2
        circles: List[Circle] = []

        # return CirclesArray([Circle(r_min, (x_max, 0.0))])
        while True:
            if x_i > x_max:
                circles.append(Circle(r_min, (x_max, 0.0)))
                break
            r_i = self.r_shortest(x_i)
            circles.append(Circle(r_i, (x_i, 0.0)))

            r_o = ell_outer.r_shortest(x_i)
            x_i = (x_i * (m - 1.0)) + (
                m * e_i * np.sqrt(r_o * r_o - r_i * r_i)
            )
        circles_array = CirclesArray(circles)
        circles_array.transform(
            self.major_axis_angle,
            dx=self.centre.x,
            dy=self.centre.y,
            pivot=(0.0, 0.0),
        )
        return circles_array

    def plot(
        self,
        axs,
        *,
        b_box: bool = False,
        uoc: bool = False,
        **kwargs,
    ):
        points = self.get_boundary_points()
        plot_kwargs = kwargs.get("plot_kwargs", {})
        axs.plot(points.x, points.y, **plot_kwargs)
        if b_box:
            bbox_plot_kwargs = kwargs.get("bbox_plot_kwargs", {})
            self.get_bounding_box().plot(axs, **bbox_plot_kwargs)
        if uoc:
            uoc_plot_kwargs = kwargs.get("uoc_plot_kwargs", {})
            uoc_dh = kwargs.get("uoc_dh", 0.1)
            self.union_of_circles(uoc_dh).plot(axs, **uoc_plot_kwargs)
        return axs


# endregion Ellipse
# ============================================================================
# region CircularArc


class CircularArc(EllipticalArc):
    def __init__(
        self,
        radius: FloatType,
        centre: Tuple[FloatType, FloatType] | Point2D = (0.0, 0.0),
        theta_1: FloatType = 0.0,
        theta_2: FloatType = TWO_PI,
    ):
        super().__init__(radius, radius, centre, 0.0, theta_1, theta_2)


# endregion CircularArc
# ============================================================================
# region Circle


class Circle:
    __slots__ = ["radius", "centre", "arc"]

    def __init__(
        self,
        radius: FloatType,
        centre: Tuple[FloatType, FloatType] | Point2D = (0.0, 0.0),
    ):
        assert radius > 0, "Radius must be greater than zero"
        self.radius: FloatType = radius
        self.centre: Point2D = Point2D(centre[0], centre[1])
        self.arc: Ellipse = Ellipse(radius, radius, centre)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(radius={self.radius}, "
            f"centre=({self.centre.x}, {self.centre.y}))"
        )

    @property
    def area(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(PI * self.radius * self.radius)

    @property
    def perimeter(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(TWO_PI * self.radius)

    def contains(
        self, p: Point2D | Tuple[FloatType, FloatType], tol=1e-8
    ) -> Literal[-1, 0, 1]:
        p = Point2D(p[0], p[1])
        dist = self.centre.distance_to(p)
        if dist > self.radius + tol:
            return -1
        elif dist < self.radius - tol:
            return 1
        else:
            return 0

    def distance_to(self, c: "Circle") -> DEFAULT_FLOAT:
        assert isinstance(c, Circle), "'c' must be of Circle type"
        return self.centre.distance_to(c.centre)


# endregion Circle
# region CirclesArray: Init


class CirclesArray:
    _slots_ = ("_data", "_size", "_capacity")

    def __init__(
        self,
        circles: Union[List[Circle], NDArray, None] = None,
        initial_capacity: int = 100,
    ):
        self._capacity: int = max(initial_capacity, 10)
        self._size = 0
        self._data = np.empty((self._capacity, 3), dtype=DEFAULT_FLOAT)  # xyr

        if circles is not None:
            self.add_circles(circles)

    def add_circles(self, circles: Union[List[Circle], NDArray]) -> None:
        """Add circles from either a list of Circle objects or a numpy array"""
        if isinstance(circles, np.ndarray):
            self._add_from_array(circles)
        else:
            self._add_from_objects(circles)

    def _add_from_objects(self, circles: List[Circle]):
        arr = np.array(
            [(c.centre.x, c.centre.y, c.radius) for c in circles],
            dtype=DEFAULT_FLOAT,
        )
        self._add_from_array(arr)

    def _add_from_array(self, arr: NDArray):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                "Input array for adding circles must have (N, 3) shape"
            )

        new_size = self._size + arr.shape[0]
        if new_size > self._capacity:
            self._grow_to(new_size)

        self._data[self._size : self._size + arr.shape[0]] = arr
        self._size += arr.shape[0]

    def _grow_to(self, required_capacity: int) -> None:
        while self._capacity < required_capacity:
            self._capacity = int(self._capacity * 1.5)

        new_data = np.empty((self._capacity, 3), dtype=DEFAULT_FLOAT)
        new_data[: self._size] = self._data[: self._size]
        self._data = new_data

    # endregion CirclesArray: Init
    # region CirArr: Transform

    def transform(
        self,
        angle: FloatType = 0.0,
        dx: FloatType = 0.0,
        dy: FloatType = 0.0,
        pivot: Point2D | Tuple[FloatType, FloatType] = (0.0, 0.0),
        scale: Union[FloatType, NDArray] = 1.0,
    ) -> None:
        """Updates the current circle set by transformation"""

        # Rotate and translate centre points array
        centres = self.centres_point_array()
        centres.transform(angle, dx, dy, pivot=pivot, in_place=True)
        self._data[: self._size, :2] = centres.coordinates

        # Vectorised Scaling of radii
        if not isinstance(scale, (float, type(DEFAULT_FLOAT()), np.ndarray)):
            raise TypeError(
                "Scale must be of type: FloatType, list, tuple or NDArray"
                f" but got '{scale.__class__.__name__}'"
            )
        scale = np.asarray(scale, dtype=DEFAULT_FLOAT)
        if (scale.ndim == 1 and scale.size == self._size) or (
            scale.ndim == 0 and scale.size == 1
        ):
            self._data[: self._size, 2] = self._data[: self._size, 2] * scale
        else:
            raise ValueError(
                "Scale must be either a scalar or a 1D array-like "
                f"with same length as number of circles, but got {scale}"
            )

    def clip(
        self,
        x_lim: Tuple[FloatType, FloatType] = (-np.inf, np.inf),
        y_lim: Tuple[FloatType, FloatType] = (-np.inf, np.inf),
        r_lim: Tuple[FloatType, FloatType] = (0.0, np.inf),
        inplace: bool = True,
    ) -> Optional[NDArray]:
        """Returns a copy of circles array within the limits"""
        data = self._data[: self._size]
        if inplace:
            np.clip(data[:, 0], x_lim[0], x_lim[1], out=data[:, 0])
            np.clip(data[:, 1], y_lim[0], y_lim[1], out=data[:, 1])
            np.clip(data[:, 2], r_lim[0], r_lim[1], out=data[:, 2])
        else:
            _x = np.clip(data[:, 0], x_lim[0], x_lim[1])
            _y = np.clip(data[:, 1], y_lim[0], y_lim[1])
            _r = np.clip(data[:, 2], r_lim[0], r_lim[1])
            return np.column_stack((_x, _y, _r)).astype(DEFAULT_FLOAT)

    # endregion CirArr: Transform
    # ========================================================================
    # region CirArr: Properties

    @property
    def xc(self) -> NDArray:
        return self._data[: self._size, 0]

    @property
    def yc(self) -> NDArray:
        return self._data[: self._size, 1]

    @property
    def radii(self) -> NDArray:
        return self._data[: self._size, 2]

    @property
    def centres(self) -> NDArray:
        return self._data[: self._size, :2]

    @property
    def data(self) -> NDArray:
        return self._data[: self._size]

    def centres_point_array(self) -> PointArray2D:
        return PointArray2D(self.centres)

    def bounding_box(self) -> BoundingBox:
        xlb = np.min(self.xc - self.radii).item()
        ylb = np.min(self.yc - self.radii).item()
        xub = np.max(self.xc + self.radii).item()
        yub = np.max(self.yc + self.radii).item()
        return BoundingBox([xlb, ylb], [xub, yub])

    def perimeters(self):
        return 2.0 * PI * self.radii

    def areas(self):
        return PI * self.radii * self.radii

    def distances_to(
        self, p: Point2D | Tuple[FloatType, FloatType]
    ) -> NDArray:
        return np.linalg.norm(self.centres - np.array(p), axis=1)

    # endregion CirArr: Properties
    # ========================================================================
    # region CirArr: Dunder

    def contains(
        self,
        p: Union[Point2D, Tuple[FloatType, FloatType]],
        rtol: FloatType = 1e-6,
    ) -> Literal[-1, 0, 1]:
        assert isinstance(p, (Point2D, tuple))
        distances = self.distances_to(p)
        if np.any(distances < self.radii - rtol):
            return 1
        elif np.all(distances > self.radii + rtol):
            return -1
        return 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self._size} circles"

    def __iter__(self):
        return iter(self.data)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Circle, List[Circle]]:
        if isinstance(index, int):
            _x, _y, _r = self._data[index]
            return Circle(_r, (_x, _y))
        return [Circle(_r, (_x, _y)) for _x, _y, _r in self._data[index]]

    # endregion CirArr: Dunder
    # ========================================================================
    # region CirArr: Visual

    def evaluate_boundaries(
        self, pointer_per_circle: int = 36, cycle: bool = False
    ):
        theta = np.linspace(0.0, TWO_PI, pointer_per_circle, endpoint=cycle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x = self.radii[:, None] * cos_t + self.xc[:, None]
        y = self.radii[:, None] * sin_t + self.yc[:, None]
        return np.stack((x, y), axis=-1)  # (num_circles, num_points, 2)

    def plot(self, axs, *, plot_bbox: bool = False, **kwargs) -> None:
        boundaries = self.evaluate_boundaries()

        # TODO try with matplotlib.collections.CircleCollection instead
        for a_circle_boundary in boundaries:
            plt_kwargs = kwargs.get("plot_kwargs", {})
            axs.plot(
                a_circle_boundary[:, 0], a_circle_boundary[:, 1], **plt_kwargs
            )
        if plot_bbox:
            bbox_plt_kwargs = kwargs.get("bbox_plot_kwargs", {})
            self.bounding_box().plot(axs, **bbox_plt_kwargs)
