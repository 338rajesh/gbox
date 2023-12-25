# from scipy.spatial import Voronoi
from numpy import ndarray, array, concatenate, sin, cos, sqrt


def rotational_matrix(angle: float):
    return [[+cos(angle), sin(angle)], [-sin(angle), cos(angle)], ]


def rotate(x, y, angle, xc=0.0, yc=0.0, ):
    return tuple((array([[x - xc, y - yc]]) @ rotational_matrix(angle)).ravel())


class Point:
    def __init__(self, x, y):
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        self.x = x
        self.y = y

    def distance(self, x_2, y_2):
        return sqrt((self.x - x_2) ** 2 + (self.y - y_2) ** 2)

    def slope(self, x_2, y_2, eps=1e-16):
        return (y_2 - self.y) / (x_2 - self.x + (eps if x_2 == self.x else 0.0))

    def line_eqn(self, x_2, y_2):
        m = self.slope(x_2, y_2)
        return m, -1.0, y_2 - (m * x_2)


class Points(list):
    def __init__(self, points: ndarray = None):
        super(Points, self).__init__()
        if points is None:
            points = array([[0.0, 0.0]])
        assert isinstance(points, ndarray), (
            "Points must be supplied as numpy.ndarray, with each column indicating a dimension"
        )
        self.points = points
        self._x = None
        self._y = None

    @property
    def x(self):
        self._x = self.points[:, 0:1]
        return self._x

    @property
    def y(self):
        self._y = self.points[:, 1:2]
        return self._y

    @property
    def dim(self):
        return self.points.shape[-1]

    def __len__(self):
        return self.points.shape[0]

    def append(self, new_points: ndarray, end=True, ):
        assert isinstance(new_points, ndarray), f"Only points of numpy.ndarray kind can be appended."
        assert self.points.ndim == new_points.ndim, (
            f"Inconsistent number of dimensions, {self.points.ndim} != {new_points.ndim}"
        )
        assert self.points.shape[-1] == new_points.shape[-1], "Inconsistent number of coordinates at a point."
        self.points = concatenate((self.points, new_points) if end else (new_points, self.points), axis=0)
        return self

    def close_loop(self):
        self.points = concatenate((self.points, self.points[0:1, ...]), axis=0)

    def transform(self, angle=0.0, dx=0.0, dy=0.0):
        """ Transforms the points cluster by rotation and translation """
        self.points = (self.points @ rotational_matrix(angle)) + [dx, dy]
        return self

    def reverse(self):
        self.points = self.points[::-1, :]
        return self

    def make_periodic_tiles(self, bbox):
        assert bbox.dim == self.dim, "mismatch in points and bbox dimensions"
        periodic_points = []
        for i in range(3):  # shifting x
            for j in range(3):  # shifting y
                a_grid_points = concatenate((
                    (self.points[:, 0:1] - bbox.lx) + (i * bbox.lx),
                    (self.points[:, 1:2] - bbox.ly) + (j * bbox.ly),
                ), axis=1)
                if bbox.dim == 3:
                    for k in range(3):  # shifting z
                        a_grid_points = concatenate(
                            (a_grid_points, (self.points[:, 2:3] - bbox.lz) + (k * bbox.lz),),
                            axis=1
                        )
                periodic_points.append(a_grid_points)
        return concatenate(periodic_points, axis=0)

    def reflect(self, p1, p2):
        a, b, c = Point(*p1).line_eqn(*p2)
        f = 2.0 * (((a * self.x) + (b * self.y) + c) / (a ** 2 + b ** 2))
        return Points(concatenate((self.x - (a * f), self.y - (b * f)), axis=1))

    def copy(self):
        return

# TODO Voronoi tessellation
# TODO Voronoi Query
# TODO

#
# class PeriodicVoronoi:
#
#     def __init__(self, points: ndarray, bounding_box: tuple[float]):
#         self.points: ndarray = points
#         self.bbox: BoundingBox = BoundingBox(*bounding_box)
#         self.dim: int = points.shape[1]
#
#         assert self.dim == self.bbox.dim, "Mismatch in the dimension of the points and that of the bounding box"
