# This is the base level module with only dependency on the utils.py


from matplotlib.pyplot import subplots, show, savefig, axis, grid
from numpy import ndarray, pi, sqrt, array, concatenate

from .utils import rotational_matrix, is_ordered


class Point:
    def __init__(self, x: float, y: float):
        """
        A point object
        """
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        self.x = x
        self.y = y
        self.xy = (x, y)

    def distance(self, x_2: float, y_2: float):
        """
        :param x_2: x-coordinate
        :param y_2: y-coordinate
        :return: distance between the current point and (x_2, y_2)

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

        >>> Point(0.0, 2.0).line_eqn(-1.0, 6.0)
        (-4.0, -1.0, 2.0)

        """
        m = self.slope(x_2, y_2)
        return m, -1.0, y_2 - (m * x_2)

    def in_box(self, xlb, ylb, xub, yub, include_bounds=True) -> bool:
        if include_bounds:
            return (xlb <= self.x <= xub) and (ylb <= self.y <= yub)
        else:
            return (xlb < self.x < xub) and (ylb < self.y < yub)

    def rotate(self, angle, xc=0.0, yc=0.0, ) -> tuple[float, float]:
        return tuple((array([[self.x - xc, self.y - yc]]) @ rotational_matrix(angle)).ravel())

    def rotate_(self, angle, xc=0.0, yc=0.0, ):
        self.x, self.y = tuple((array([[self.x - xc, self.y - yc]]) @ rotational_matrix(angle)).ravel())
        return self


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
        self._b_box = None
        #
        self.vor = None

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

    @property
    def bbox(self):
        self._b_box = (
                [self.points[:, i].min() for i in range(self.dim)]
                + [self.points[:, i].max() for i in range(self.dim)]
        )
        return self._b_box

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

        >>> from numpy import array, pi
        >>> p = Points(array([[0.0, 0.0], [2.0, 2.0], [3.0, 5.0]]))
        >>> p.points
        array([[0., 0.],
               [2., 2.],
               [3., 5.]])
        >>> q = p.reflect((0.0, 0.0), (1.0, 2.0))
        >>> q.points
        ... array([[2.2, 5.4], [0.4, 2.8], [0., 0.0],])
        """
        a, b, c = Point(*p1).line_eqn(*p2)
        f = 2.0 * (((a * self.x) + (b * self.y) + c) / (a ** 2 + b ** 2))
        return Points(concatenate((self.x - (a * f), self.y - (b * f)), axis=1))

    def copy(self):
        return

    def make_periodic_tiles(self, bounds: list = None, order: int = 1):
        """
        It tiles the points about the current position, in the space by the specified number of times on each side.

        :param bounds: The bounds of the points cluster
        :param order: int, The number of times points are tiled in each direction.
        :return: Points object containing the tiled points.
        :rtype: Points
        """
        bbox = self.bbox if bounds is None else bounds
        bbox_lengths = [bbox[i + self.dim] - bbox[i] for i in range(self.dim)]
        indices = list(range(-order, order + 1, 1))
        periodic_points = []

        for i in indices:  # shifting x
            for j in indices:  # shifting y
                a_grid_points = concatenate((self.x + (i * bbox_lengths[0]), self.y + (j * bbox_lengths[1]),), axis=1)
                if self.dim == 3:
                    for k in indices:  # shifting z
                        a_grid_points = concatenate(
                            (a_grid_points, self.points[:, 2:3] + (k * bbox_lengths[2]),),
                            axis=1
                        )
                periodic_points.append(a_grid_points)
        return Points(concatenate(periodic_points, axis=0))

    def plot_bounding_box(self, axs, bounds=None):
        BoundingBox2D(*(self.bbox if bounds is None else bounds)).plot(axs)

    def plot(self, axs=None, file_path=None, show_fig=False, hide_axis=True, b_box=False, **sct_opt):
        """ Plots the points

        :param axs: The axs on which points are plotted. If not specified, it will create a new axs with default
         options.
        :param file_path: The path of the image (with appropriate extension) to save the figure.
        :param hide_axis: Enable/disable the plot
        :param show_fig: Should the figure be displayed using ``matplotlib.pyplot.show()``
        :param b_box: Whether to plot the bounding box around the points
        :param sct_opt: All the key-word arguments that can be taken by ``matplotlib.pyplot.scatter()`` function.
        :return: None,
        :rtype: None
        """
        if self.dim != 2:
            raise NotImplementedError("!!At present only one two dimensional points plotting is supported!!")
        if axs is None:
            fig, axs = subplots()
        axs.scatter(self.points[:, 0:1], self.points[:, 1:2], **sct_opt)
        if b_box:
            self.plot_bounding_box(axs)

        if hide_axis:
            axis('off')
        if file_path is not None:
            savefig(file_path)
        if show_fig:
            show()


class BoundingBox2D:
    def __init__(self, xlb=-1.0, ylb=-1.0, xub=1.0, yub=1.0):
        is_ordered(xlb, xub, "x lower bound", "x upper bound")
        is_ordered(ylb, yub, "y lower bound", "y upper bound")
        lx, ly = xub - xlb, yub - ylb
        self.xlb = xlb
        self.xub = xub
        self.ylb = ylb
        self.yub = yub
        self.xc = 0.5 * (xlb + xub)
        self.yc = 0.5 * (ylb + yub)
        self.lx = lx
        self.ly = ly
        self._locus = None

    def has_point(self, p: list | tuple | Point):
        if isinstance(p, Point):
            return (self.xlb <= p.x <= self.xub) and (self.ylb <= p.y <= self.yub)
        elif isinstance(p, (list, tuple)):
            return (self.xlb <= p[0] <= self.xub) and (self.ylb <= p[1] <= self.yub)

    @property
    def locus(self):
        self._locus = Points(array([
            [self.xlb, self.ylb],
            [self.xub, self.ylb],
            [self.xub, self.yub],
            [self.xlb, self.yub],
            [self.xlb, self.ylb],
        ]))
        return self._locus.points

    def plot(
            self,
            axs=None,
            show_fig=False,
            file_path=None,
            hide_axis=True,
            show_grid=False,
            face_color='None',
            edge_color='k',
            **plot_opt
    ):
        if axs is None:
            fig, axs = subplots()
        axs.fill(self.locus[:, 0], self.locus[:, 1], facecolor=face_color, edgecolor=edge_color, **plot_opt)
        axis('off') if hide_axis else None
        grid() if show_grid else None
        if file_path is not None:
            savefig(file_path)
        if show_fig:
            show()
        return axs
