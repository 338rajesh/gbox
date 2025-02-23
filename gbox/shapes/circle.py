from .ellipse import EllipticalArc, Ellipse
from ..core import float_type
from ..constants import TWO_PI


class CircularArc(EllipticalArc):
    def __init__(
            self,
            radius: float_type,
            centre: tuple[float_type, float_type] = (0.0, 0.0),
            theta_1: float_type = 0.0,
            theta_2: float_type = TWO_PI
    ):
        super(CircularArc, self).__init__(
            radius, radius, centre, 0.0, theta_1, theta_2
        )


# class Circle(Ellipse):
#     def __init__(
#         self,
#         radius,
#         centre=(0.0, 0.0),
#         theta_1: float = 0.0,
#         theta_2: float = TWO_PI,
#     ):
#         assert radius > 0, "Radius must be greater than zero"
#         assert theta_1 < theta_2, "Theta 1 must be less than theta 2"
#         assert (
#             theta_1 >= 0.0 and theta_2 <= TWO_PI
#         ), "Theta 1 and theta 2 must be between 0 and 2.0 * pi"

#         super(Circle, self).__init__()

#         self.radius = radius
#         self.centre = Point2D(*centre)
#         self.boundary: PointSet2D = None
#         self.theta_1 = theta_1
#         self.theta_2 = theta_2

#         self.area = PI * radius * radius
#         self.perimeter = TWO_PI * radius

#     def eval_boundary(self, num_points=None, arc_length=0.1, min_points=100):
#         if num_points is None:
#             num_points = max(
#                 int(np.ceil(TWO_PI * self.radius / arc_length)), min_points
#             )

#         theta = np.linspace(self.theta_1, self.theta_2, num_points)

#         xy = np.empty((num_points, 2))
#         xy[:, 0] = self.radius * np.cos(theta)
#         xy[:, 1] = self.radius * np.sin(theta)

#         xy[:, 0] += self.centre.x
#         xy[:, 1] += self.centre.y

#         self.boundary = PointSet2D(xy)

#         return self

#     def contains_point(self, p: PointType, tol=1e-8) -> typing.Literal[-1, 0, 1]:
#         p = Point2D.from_seq(p)
#         assert p.dim == 2, "Expecting 2D points"
#         dist: float = self.centre.distance_to(p)
#         if dist > self.radius + tol:
#             return -1
#         elif dist < self.radius - tol:
#             return 1
#         else:
#             return 0

#     def distance_to(self, c: "Circle") -> float:
#         assert isinstance(c, Circle), "'c' must be of Circle type"
#         return self.centre.distance_to(c.centre)

#     def plot(
#         self, axs, b_box=False, b_box_plt_opt=None, points_plt_opt=None, cycle=True
#     ) -> None:
#         if self.boundary is None:
#             self.eval_boundary()
#         return super().plot(axs, b_box, b_box_plt_opt, points_plt_opt, cycle)


# class CircleSet:
#     def __init__(self, *circles: Circle, initial_capacity: int = 100):
#         if len(circles) == 0:
#             raise ValueError("Must have at least one circle")
#         for a_c in circles:
#             if not isinstance(a_c, Circle):
#                 raise TypeError("All elements must be of type Circle")

#         # Pre-allocating memory
#         self.capacity: int = initial_capacity
#         self._data: NDArray = np.empty((self.capacity, 3))
#         self.size: int = 0
#         self.boundaries: list[PointSet2D] = []

#         self.add_circles(*circles)

#     def add_circles(self, *c: Circle) -> None:

#         # Check, if the pre-allocated memory is sufficient
#         req_size = len(c) + self.size
#         if req_size > self.capacity:
#             self._grow_to(req_size)

#         new_data = np.array([c.centre.as_list() + [c.radius] for c in c])

#         # Adding the new circles
#         self._data[self.size: len(c) + self.size] = new_data

#         self.size += len(c)

#     def _grow_to(self, new_size: int) -> None:
#         while self.capacity < new_size:
#             self.capacity = int(self.capacity * 1.5)

#         new_data = np.empty((self.capacity, 3))
#         new_data[: self.size] = self._data[: self.size]
#         self._data = new_data

#     def transform(
#         self,
#         dx: float = 0.0,
#         dy: float = 0.0,
#         angle: float = 0.0,
#         scale: float | NDArray = 1.0,
#         pivot: tuple[float, float] = (0.0, 0.0),
#     ) -> "CircleSet":
#         """Updates the current circle set by transformation"""

#         # Scaling the Radius
#         if not isinstance(scale, (float, list, tuple, NDArray)):
#             raise TypeError(
#                 "Scale must be of type: float, list, tuple or NDArray")
#         scale = np.atleast_1d(scale)
#         assert scale.ndim == 1, "Scale must be a 1D array"
#         if scale.size not in (1, self.size):
#             raise ValueError(
#                 "Scale must be a float or have same length as the number of circles"
#             )
#         self._data[: self.size, 2] = self._data[: self.size, 2] * scale

#         # Applying Rotation and Translation, if required
#         if angle != 0.0:
#             x = self._data[: self.size, 0] - pivot[0]
#             y = self._data[: self.size, 1] - pivot[1]
#             x_ = x * np.cos(angle) - y * np.sin(angle) + pivot[0]
#             y_ = x * np.sin(angle) + y * np.cos(angle) + pivot[1]
#             self._data[: self.size, 0] = x_
#             self._data[: self.size, 1] = y_

#         if dx != 0.0:
#             self._data[: self.size, 0] += dx

#         if dy != 0.0:
#             self._data[: self.size, 1] += dy

#         return self

#     @property
#     def data(self) -> NDArray:
#         return self._data[: self.size]

#     @property
#     def centres(self) -> NDArray:
#         return self._data[: self.size, :2]

#     @property
#     def xc(self) -> NDArray:
#         return self._data[: self.size, 0]

#     @property
#     def yc(self) -> NDArray:
#         return self._data[: self.size, 1]

#     @property
#     def radii(self) -> NDArray:
#         return self._data[: self.size, 2]

#     def __len__(self) -> int:
#         return self.size

#     def __repr__(self) -> str:
#         return f"CircleSet:\n{self.size} circles"

#     def __iter__(self):
#         return iter(self.data)

#     def clip(self, r_min, r_max):
#         raise NotImplementedError("clip is not implemented")

#     def evaluate_boundaries(self, num_points=None, arc_length=0.1, min_points=100):
#         if num_points is None:
#             num_points = max(
#                 int(np.ceil(TWO_PI * np.max(self.radii) / arc_length)), min_points
#             )

#         t = np.linspace(0.0, TWO_PI, num_points)
#         xy = np.empty((self.size, num_points, 2))
#         xy[:, :, 0] = self.radii[:, None] * np.cos(t)
#         xy[:, :, 1] = self.radii[:, None] * np.sin(t)

#         xy[:, :, 0] = xy[:, :, 0] + self.xc[:, None]
#         xy[:, :, 1] = xy[:, :, 1] + self.yc[:, None]

#         self.boundaries = tuple(PointSet2D(a_xy) for a_xy in xy)
#         return self

#     def bounding_box(self) -> BoundingBox:
#         xlb = np.min(self.xc - self.radii)
#         xub = np.max(self.xc + self.radii)
#         ylb = np.min(self.yc - self.radii)
#         yub = np.max(self.yc + self.radii)
#         return BoundingBox([xlb, ylb], [xub, yub])

#     def perimeters(self):
#         return TWO_PI * self.radii

#     def areas(self):
#         return PI * self.radii * self.radii

#     def distances_to(self, p: PointType) -> NDArray:
#         p = Point2D.from_seq(p)
#         assert p.dim == 2, "other point must be of dimension 2"
#         return np.linalg.norm(self.centres - p.as_array(), axis=1)

#     def contains_point(self, p: PointType, tol=1e-8) -> typing.Literal[-1, 0, 1]:
#         p = Point2D.from_seq(p)
#         assert p.dim == 2, "other point must be of dimension 2"
#         distances = self.distances_to(p)
#         if np.any(distances < self.radii - tol):
#             return 1
#         elif np.all(distances > self.radii + tol):
#             return -1
#         return 0

#     def plot(
#         self, axs, b_box=False, b_box_plt_opt=None, points_plt_opt=None, cycle=True
#     ) -> None:
#         self.evaluate_boundaries()
#         if b_box:
#             self.bounding_box().plot(axs, **b_box_plt_opt)
#         for idx, a_boundary in enumerate(self.boundaries):
#             if "label" in points_plt_opt and idx > 0:
#                 del points_plt_opt["label"]
#             a_boundary.plot(axs, points_plt_opt=points_plt_opt)
