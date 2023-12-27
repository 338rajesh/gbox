# from scipy.spatial import Voronoi
from numpy import ndarray, array, concatenate
from numpy import sin, cos, sqrt


def rotational_matrix(angle: float):
    """
    Rotational matrix for rotating a plane through angle in counter clock wise direction.

    >>> from math import pi
    >>> rotational_matrix(0.25 * pi)
    [[0.7071067811865476, 0.7071067811865476], [-0.7071067811865476, 0.7071067811865476]]

    :rtype: list[list]
    """
    return [[+cos(angle), sin(angle)], [-sin(angle), cos(angle)], ]


def rotate(x: float, y: float, angle: float, xc: float = 0.0, yc: float = 0.0, ):
    """
    Rotate the points `x` and `y` by specified angle about the point (xc, yc).

    >>> import gbox
    >>> from math import pi
    >>> gbox.points.rotate(1.0, 1.0, 0.25 * pi, 0.0, 0.0)
    (0.0, 1.41421356237)

    :rtype: tuple[float, float]
    """
    return tuple((array([[x - xc, y - yc]]) @ rotational_matrix(angle)).ravel())


class Point:
    def __init__(self, x: float, y: float):
        """
        A point object
        """
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        self.x = x
        self.y = y

    def distance(self, x_2: float, y_2: float):
        """
        :param x_2: x-coordinate
        :param y_2: y-coordinate
        :return: distance between the current point and (x_2, y_2)

        >>> from gbox import Point
        >>> Point(0.0, 0.0).distance(3.0, 4.0)
        5.0

        """
        return sqrt((self.x - x_2) ** 2 + (self.y - y_2) ** 2)

    def slope(self, x_2: float, y_2: float, eps=1e-16):
        """
        :param x_2: x-coordinate
        :param y_2: y-coordinate
        :param eps: A small number to avoid zero-division errors
        :return: the slope of the line connecting this points to (x_2, y_2)

        >>> from gbox import Point
        >>> Point(0.0, 0.0).slope(3.0, 3.0)
        1.0

        """
        return (y_2 - self.y) / (x_2 - self.x + (eps if x_2 == self.x else 0.0))

    def line_eqn(self, x_2: float, y_2: float):
        """
        Finds the equation of line connecting this point to (x_2, y_2)
        :param x_2:
        :param y_2:
        :return: a, b, c of the line equation ax + by + c = 0
        :rtype: tuple[float, float, float]

        >>> from gbox import Point
        >>> Point(0.0, 2.0).line_eqn(-1.0, 6.0)
        (-4.0, -1.0, 2.0)

        """
        m = self.slope(x_2, y_2)
        return m, -1.0, y_2 - (m * x_2)


class Points(list):
    """ Collection of **ordered** points """

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
        self.append(self.points[0:1, ...])

    def transform(self, angle=0.0, dx=0.0, dy=0.0):
        """
            Transforms the points cluster by rotation and translation

        >>> from gbox import Points
        >>> from numpy import array, pi
        >>> p = Points(array([[0.0, 0.0], [2.0, 2.0], [3.0, 5.0]]))
        >>> p.points
        array([[0., 0.],
            [2., 2.],
            [3., 5.]])
        >>> q = p.transform(angle=0.2 * pi, dx=2.0, dy=2.0)
        >>> q.points
        array([[ 2.44246348,  4.79360449],
            [ 1.15838444,  7.31375151],
            [-1.38576811,  9.19185901]])

        """
        self.points = (self.points @ rotational_matrix(angle)) + [dx, dy]
        return self

    def reverse(self):
        """
            Reverses the order of points **in-place**

        >>> from gbox import Points
        >>> from numpy import array, pi
        >>> p = Points(array([[0.0, 0.0], [2.0, 2.0], [3.0, 5.0]]))
        >>> p.points
        array([[0., 0.],
            [2., 2.],
            [3., 5.]])
        >>> p.reverse()
        >>> p.points
        array([[3., 5.],
            [2., 2.],
            [0., 0.]])

        """
        self.points = self.points[::-1, :]
        return self

    def reflect(self, p1: tuple[float, float], p2: tuple[float, float]):
        """
        Reflects the current points about a line connecting p1 and p2

        >>> from gbox import Points
        >>> from numpy import array, pi
        >>> p = Points(array([[0.0, 0.0], [2.0, 2.0], [3.0, 5.0]]))
        >>> p.points
        array([[0., 0.],
               [2., 2.],
               [3., 5.]])
        >>> q = p.reflect((0.0, 0.0), (1.0, 2.0))
        >>> q.points
        array([[2.2, 5.4],
               [0.4, 2.8],
               [0. , 0. ]])

        """
        a, b, c = Point(*p1).line_eqn(*p2)
        f = 2.0 * (((a * self.x) + (b * self.y) + c) / (a ** 2 + b ** 2))
        return Points(concatenate((self.x - (a * f), self.y - (b * f)), axis=1))

    def make_periodic_tiles(self, bbox):  # TODO
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
