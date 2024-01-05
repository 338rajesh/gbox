# This is gbox: Geometry Box module

"""
Assumptions:

* All the angular units are in the radians

"""
from math import pi

from matplotlib.pyplot import subplots, show, savefig, close, axis, tight_layout
from numpy import ndarray, pi, sqrt, stack, array, concatenate, tan, linspace, zeros_like, cos, sin, sum, roll, arcsin
from scipy.spatial import Voronoi, ConvexHull

from .utils import PLOT_OPTIONS, assert_positivity, rotational_matrix, get_pairs, is_ordered, assert_range


def rotate(x: float, y: float, angle: float, xc: float = 0.0, yc: float = 0.0, ):
    """
    Rotate the points `x` and `y` by specified angle about the point (xc, yc).

    >>> from math import pi
    >>> rotate(1.0, 1.0, 0.25 * pi, 0.0, 0.0)
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

    def in_box(self, xlb, ylb, xub, yub, include_bounds=True) -> bool:
        if include_bounds:
            return (xlb <= self.x <= xub) and (ylb <= self.y <= yub)
        else:
            return (xlb < self.x < xub) and (ylb < self.y < yub)


class VoronoiCells(Voronoi):
    def __init__(self, points: ndarray, tile_order=0, clip=False, bounds=None, **kwargs):
        # tiling points if required, otherwise points are passed as they are
        if tile_order > 0:
            points = Points(points).make_periodic_tiles(tile_order).points
        super(VoronoiCells, self).__init__(points, **kwargs)
        self.tile_order = tile_order
        self.clip = clip
        self.bounds = bounds
        #
        # Get ridge_dict with keys: ids of points on either side of the ridge & values: ids of vertices on the ridges
        self.ridge_dict_inv: dict = {tuple(v): list(k) for (k, v) in self.ridge_dict.items()}
        self._cells_vertices = None
        self._cells_neighbour_points = None
        self.valid_cells = self.get_valid_cells()
        self._cells_volume = None
        self._neighbour_distances = None

    def get_valid_cells(self):
        # Get all the valid cells information: list[tuple[coordinates of cell point, ids of the cell vertices]]
        _valid_cells = [
            (self.points[count], self.regions[i]) for (count, i) in enumerate(self.point_region)
            if (self.regions[i] != [] and -1 not in self.regions[i])
        ]
        if self.clip:
            assert self.bounds is not None, "If clip is set to True, bounds must be provided."
            # Filter out all the cells outside the bounds of actual points (not including the tiled points)
            _valid_cells = [
                ((xc, yc), j) for ((xc, yc), j) in _valid_cells if BoundingBox2D(*self.bounds).has_point([xc, yc])
            ]
        return _valid_cells

    @property
    def cells_vertices(self):
        self._cells_vertices = {
            cell_centre: self.vertices[cell_vertices_ids] for (cell_centre, cell_vertices_ids) in self.valid_cells
        }
        return self._cells_vertices

    @property
    def cells_neighbour_points(self):
        cells_neighbours = {}
        for (cell_centre, cell_vertices_ids) in self.valid_cells:
            cell_ridges = get_pairs(cell_vertices_ids, loop=True)  # Making a list of current cell ridges
            #
            neighbour_points = []
            for (r1, r2) in cell_ridges:
                if (r1, r2) in self.ridge_dict_inv.keys():
                    b = self.points[self.ridge_dict_inv[(r1, r2)]]
                elif (r2, r1) in self.ridge_dict_inv.keys():
                    b = self.points[self.ridge_dict_inv[(r2, r1)]]
                else:
                    raise ValueError()
                neighbour_points.append(b)
            neighbour_points = array(neighbour_points)

            if neighbour_points.size != 0:
                cells_neighbours[cell_centre] = neighbour_points
        self._cells_neighbour_points = cells_neighbours
        return self._cells_neighbour_points

    @property
    def cells_volume(self):
        if self.ndim == 2:  # it is much computationally efficient than ConvexHull
            self._cells_volume = [Polygon(v).area for v in self.cells_vertices.values()]
        else:
            self._cells_volume = [ConvexHull(v).volume for v in self.cells_vertices.values()]
        return self._cells_volume

    @property
    def neighbour_distances(self):
        self._neighbour_distances = {
            cell_centre: [Point(*p1).distance(*p2) for (p1, p2) in neighbour_point_coordinates]
            for (cell_centre, neighbour_point_coordinates) in self.cells_neighbour_points.items()
        }
        return self._neighbour_distances

    def plot(self,
             axs=None,
             show_fig=False,
             file_path=None,
             pnt_plt_opt: dict = None,
             ridge_plt_opt: dict = None,
             bounds_plt_opt: dict = None,
             ):
        assert self.ndim == 2, NotImplementedError("At present only two dimension vor plotting is supported!")
        if ridge_plt_opt is None:
            ridge_plt_opt = {'face_color': 'None', 'edge_color': 'b', 'linewidth': 1.0}
        if pnt_plt_opt is None:
            pnt_plt_opt = {'c': 'b', 'marker': '*', 's': 5.0}
        if bounds_plt_opt is None:
            bounds_plt_opt = {'face_color': 'None', 'edge_color': 'k', 'linestyle': 'solid', 'linewidth': 0.75}

        if axs is None:
            fig, axs = subplots()
        #
        # Axis decorations
        axis('off')
        axis('equal')
        tight_layout()
        for ((xc, yc), cell_vertices_coordinates) in self.cells_vertices.items():
            Polygon(cell_vertices_coordinates).plot(axs, **ridge_plt_opt)
            axs.scatter(xc, yc, **pnt_plt_opt)

        if bounds_plt_opt is not None:
            BoundingBox2D(*self.bounds).plot(axs, **bounds_plt_opt)

        # Output
        if file_path is not None:
            savefig(file_path)
        if show_fig:
            show()


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

    def make_periodic_tiles(self, order: int = 1):
        """
        It tiles the points about the current position, in the space by the specified number of times on each side.
        :param order: int, The number of times points are tiled in each direction.
        :return: Points object containing the tiled points.
        :rtype: Points
        """
        bbox = self.bbox
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


# ======================================================================================================================
#                                             CURVES
# ======================================================================================================================


class Shape:
    """ Base class for all shapes """
    pass


class Shape2D(Shape):
    """ Base class for the two-dimensional shapes """

    def __init__(self):
        self._locus: Points = Points()
        self._num_locus_points: int = 100
        self._b_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    @property
    def num_locus_points(self) -> int:
        """
            Number of points along the locus of the shape

        :rtype: Points

        """
        return self._num_locus_points

    @num_locus_points.setter
    def num_locus_points(self, val):
        self._num_locus_points = val

    @property
    def locus(self):
        """
            The locus of 2D shapes

         :rtype: Points

        """
        return self._locus

    @locus.setter
    def locus(self, value):
        if isinstance(value, ndarray):
            self._locus = Points(value)
        elif isinstance(value, Points):
            self._locus = value
        else:
            raise TypeError(f"locus must be either 'numpy.ndarray' or 'Points' type but not {type(value)}")

    @property
    def bounding_box(self):
        return self._b_box

    @bounding_box.setter
    def bounding_box(self, val):
        assert len(val) == 4, "Bounds must be supplied as a tuple / list of four real numbers"
        assert (val[2] > val[0] and val[3] > val[1]), "bounds must be in the order of x_min, y_min, x_max, y_max"
        self._b_box = val


class StraightLine(Shape2D):
    """
        Line segment, defined by its length, starting point and orientation with respect to the positive x-axs.


        >>> line = StraightLine(5.0, (1.0, 1.0,), 0.25 * pi)
        >>> line.length
        ... 5.0
        >>> line.slope
        ... 0.9999999999999999
        >>> line.equation()
        ... (0.9999999999999999, -1.0, 1.1102230246251565e-16)
        >>> line.locus.points
        ... array([[1., 1.],
        ...      [1.03571246, 1.03571246],
        ...      [1.07142493, 1.07142493],
        ...      .
        ...      .
        ...      [4.49982144, 4.49982144],
        ...      [4.53553391, 4.53553391]])

    """

    def __init__(
            self,
            length: float = 2.0,
            start_point: tuple[float, float] = (0.0, 0.0),
            angle: float = 0.0,
    ):
        super(StraightLine, self).__init__()
        self.length = length
        self.x0, self.y0 = start_point
        self.angle = angle

        self._slope = 0.0

    @property
    def slope(self):
        self._slope = tan(self.angle)
        return self._slope

    def equation(self):
        """ Returns a, b, c of the line equation in the form of ax + by + c = 0 """
        return self.slope, -1.0, (self.y0 - (self.slope * self.x0))

    @property
    def locus(self):
        xi = linspace(0.0, self.length, self.num_locus_points)
        self._locus = Points(stack((xi, zeros_like(xi)), axis=1))
        self._locus.transform(self.angle, self.x0, self.y0)
        return self._locus


class EllipticalArc(Shape2D):
    """
    >>> ellipse_arc = EllipticalArc(2.0, 1.0, 0.0, pi * 0.5, (2.0, 5.0), 0.4 * pi )
    >>> ellipse_arc.locus.points
    >>> ellipse_arc.locus.points  # returns locus with 100 points by default
    ... array([[2.61803399, 6.90211303],
    ...        [2.60286677, 6.90677646],
    ...        [2.58754778, 6.91095987],
    ...        .
    ...        .
    ...        .
    ...        [1.0588689 , 5.33915695],
    ...        [1.04894348, 5.30901699]])
    >>> ellipse_arc.num_locus_points = 6
    >>> ellipse_arc.locus.points
    ... array([[2.61803399, 6.90211303],
    ...        [2.29389263, 6.9045085 ],
    ...        [1.94098301, 6.7204774 ],
    ...        [1.59385038, 6.36803399],
    ...        [1.28647451, 5.88167788],
    ...        [1.04894348, 5.30901699]])
    """

    def __init__(
            self,
            smj: float = 2.0,
            smn: float = 1.0,
            theta_1: float = 0.0,
            theta_2: float = pi / 2,
            centre=(0.0, 0.0),
            smj_angle: float = 0.0,
    ):
        super(EllipticalArc, self).__init__()
        self.smj = smj
        self.smn = smn
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.xc, self.yc = self.centre = centre
        self.smj_angle = smj_angle
        # self.locus: Points = Points()

    @property
    def locus(self):
        theta = linspace(self.theta_1, self.theta_2, self.num_locus_points)
        self._locus = Points(stack((self.smj * cos(theta), self.smn * sin(theta)), axis=1))
        self._locus.transform(self.smj_angle, self.xc, self.yc)
        return self._locus


class CircularArc(EllipticalArc):
    def __init__(self, r=1.0, theta_1=0.0, theta_2=2.0 * pi, centre=(0.0, 0.0)):
        super(CircularArc, self).__init__(r, r, theta_1, theta_2, centre, 0.0)


# ======================================================================================================================
#                                             CLOSED SHAPES
# ======================================================================================================================


class ShapePlotter:
    """
        Plotter for various shapes
    """

    def __init__(
            self,
            locus: Points,
            axs=None,
            f_path=None,
            closure=True,
            linewidth=None,
            show_grid=None,
            hide_axes=None,
            face_color=None,
            edge_color=None,
    ):
        """

        :param axs: Shape is plotted on this axs and returns the same, If not provided, a figure will be created
         with default options which will be saved at `f_path` location if the `f_path` is specified.
         Otherwise, it will be displayed using matplotlib.pyplot.show() method.
        :param f_path: str, file path to save the figure
        :param closure: Whether to make loop by connecting the last point with the first point.
        :param face_color: str, Color to fill the shape
        :param edge_color: str, Color of the edge
        :param linewidth: float,
        :param show_grid: bool, enable/disable the grid on figure
        :param hide_axes: bool, enable/disable the axs on figure
        :return: None

        """
        if show_grid is None:
            show_grid = PLOT_OPTIONS.show_grid
        if hide_axes is None:
            hide_axes = PLOT_OPTIONS.hide_axes
        if linewidth is None:
            linewidth = PLOT_OPTIONS.linewidth
        if face_color is None:
            face_color = PLOT_OPTIONS.face_color
        if edge_color is None:
            edge_color = PLOT_OPTIONS.edge_color
        #
        assert locus is not None, "Plotting a shape requires locus but it is set to `None` at present."
        if closure:
            locus.close_loop()
        self.locus: Points = locus
        self.axs = axs
        self.f_path = f_path
        self.closure = closure
        self.linewidth = linewidth
        self.show_grid = show_grid
        self.hide_axes = hide_axes
        self.face_color = face_color
        self.edge_color = edge_color

    def _plot_on_axis(self, _axs, fill=True, title=None, **plt_opt):
        if fill:
            _axs.fill(
                self.locus.points[:, 0],
                self.locus.points[:, 1],
                facecolor=self.face_color,
                edgecolor=self.edge_color,
                linewidth=self.linewidth,
                **plt_opt
            )
        else:
            _axs.plot(
                self.locus.points[:, 0],
                self.locus.points[:, 1],
                color=self.edge_color,
                linewidth=self.linewidth,
                **plt_opt
            )

        axis('equal')

        if title is not None:
            _axs.set_title(title)
        if self.show_grid:
            _axs.grid()
        if self.hide_axes:
            axis('off')
        return _axs

    def _plot(self, fill_plot=True, title=None, **plt_opt):

        def _plt():
            if fill_plot:
                self._plot_on_axis(self.axs, fill=True, title=title, **plt_opt)
            else:
                self._plot_on_axis(self.axs, fill=False, title=title, **plt_opt)

        if self.axs is None:
            _, self.axs = subplots(1, 1)
            _plt()
            if self.f_path is None:
                try:
                    show()
                except ValueError as e:
                    print(f"Tried to display the figure but not working due to {e}")
            else:
                savefig(self.f_path)
                close('all')
        else:
            return _plt()

    def line_plot(self, title=None, **plt_opt):
        """
            Line plot of the shapes
        """
        self._plot(fill_plot=False, title=title, **plt_opt)

    def fill_plot(self, title=None, **plt_opt):
        self._plot(fill_plot=True, title=title, **plt_opt)


class Curve2D(Shape2D):
    """ Curve in tw-dimensional space """

    def plot(
            self,
            axs=None,
            f_path=None,
            closure=True,
            linewidth=None,
            show_grid=None,
            hide_axes=None,
            edge_color='b',
            title=None,
            **plt_opt
    ):
        ShapePlotter(
            self.locus, axs, f_path, closure, linewidth, show_grid, hide_axes,
            edge_color=edge_color,
        ).line_plot(title=title, **plt_opt)


class ClosedShape2D(Shape2D):
    """
        Closed Shape in the two-dimensional space or a plane is defined by
        the locus of points, pivot point (lying on or inside or outside) the locus and angle made by a pivot axs.
        The pivot point and axs are used for convenience and are set to `(0.0, 0.0)` and 0.0 degrees by default.
    """

    def __init__(
            self,
            pivot_point=(0.0, 0.0),
            pivot_angle=0.0,
    ):
        super(ClosedShape2D, self).__init__()
        self.pxc, self.pyc = self.pivot_point = pivot_point
        self.pivot_angle = pivot_angle
        #
        self._area = 0.0
        self._perimeter = 0.0
        self._sf = 1.0

    @property
    def area(self):
        """

        :rtype: float

        """
        return self._area

    @property
    def perimeter(self):
        """

        :rtype: float

        """
        return self._perimeter

    @property
    def shape_factor(self):
        """

        :rtype: float

        """
        assert_positivity(self.area, 'Area')
        assert_positivity(self.perimeter, 'Perimeter')
        self._sf = self.perimeter / sqrt(4.0 * pi * self.area)
        return self._sf

    def plot(
            self,
            axs=None,
            f_path=None,
            closure=True,
            linewidth=None,
            show_grid=None,
            hide_axes=None,
            face_color=None,
            edge_color=None,
            title=None,
            **plt_opt
    ):
        """

        :rtype: None

        """
        ShapePlotter(
            self.locus, axs, f_path, closure, linewidth, show_grid, hide_axes,
            face_color=face_color, edge_color=edge_color,
        ).fill_plot(title=title, **plt_opt)


class ShapesList(list):
    """
        List of multiple shapes
    """

    def __init__(self):
        super(ShapesList, self).__init__()
        self._loci = Points()
        self._perimeters = ()
        self._areas = ()
        self._shape_factors = ()

    def plot(self, **kwargs):
        """
        A convenient method for plotting multiple shapes, and it takes same arguments and key-word arguments as
        the ClosedShapes2D.plot()

        :rtype: None

        """
        for i in range(self.__len__()):
            self.__getitem__(i).plot(**kwargs)

    @property
    def loci(self):
        """
        Evaluates locus of all the shapes in the list. The first dimension of the loci refers to shapes

        :rtype: Points

        """
        self._loci = Points(stack([self.__getitem__(i).locus.points for i in range(self.__len__())], axis=0))
        return self._loci

    @property
    def perimeters(self):
        """
            Evaluates perimeters of all the shapes in the list.

        :rtype: list[float]

        """
        self._perimeters = [self.__getitem__(i).perimeter for i in range(self.__len__())]
        return self._perimeters

    @property
    def areas(self):
        """
            Evaluates area of all the shapes in the list.

        :rtype: list[float]

        """
        self._areas = [self.__getitem__(i).area for i in range(self.__len__())]
        return self._areas

    @property
    def shape_factors(self):
        """
            Evaluates shape factors of all the shapes in the list.

        :rtype: list[float]

        """
        self._shape_factors = [self.__getitem__(i).shape_factor for i in range(self.__len__())]
        return self._shape_factors


class ClosedShapesList(ShapesList):
    """ List of multiple closed shapes """

    def __init__(self):
        super(ClosedShapesList, self).__init__()

    @staticmethod
    def validate_incl_data(a, n):
        assert isinstance(a, ndarray), "given inclusion data must be an numpy.ndarray"
        assert a.shape[1] == n, f"Incorrect number of columns, found {a.shape[1]} instead of {n}"
        return


class Ellipse(ClosedShape2D):
    """
    Ellipse defined its centre, orientation of semi-major axs with the positive x-axs, starting and ending points
    (defined by the parametric values theta_1 and theta_2), semi-major and semi-minor axs lengths. It has perimeter,
    area, shape factor, locus, bounding box and union of circles representation properties.

    >>> ellipse = Ellipse()
    >>> ellipse.smj  # prints semi-major axs length, a
    >>> ellipse.smn  # prints semi-minor axs length, b
    >>> ellipse.pivot_point  # prints centre of the ellipse
    >>> ellipse.pivot_angle  # prints orientation of the semi-major axs of the ellipse
    >>> ellipse.shape_factor  # prints shape factor of the ellipse

    """

    def __init__(self,
                 smj: float = 2.0,
                 smn: float = 1.0,
                 theta_1=0.0,
                 theta_2=2.0 * pi,
                 centre=(0.0, 0.0),
                 smj_angle=0.0,
                 ):
        is_ordered(smn, smj, 'Semi minor axs', 'Semi major axs')
        self.smj = smj
        self.smn = smn
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        super(Ellipse, self).__init__(centre, smj_angle)
        self._ecc = 1.0

    @property
    def perimeter(self):
        """
        Perimeter is approximated using the following Ramanujan formula

        .. math::
            p = \\pi[3(a+b) - \\sqrt{(3a + b)(a + 3b)}]

        """
        self._perimeter = pi * (
                (3.0 * (self.smj + self.smn))
                - sqrt(((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn)))
        )
        return self._perimeter

    @property
    def area(self):
        """
        Area is evaluated using the following formula,

        .. math::
            A = \\pi a b

        """
        self._area = pi * self.smj * self.smn
        return self._area

    @staticmethod
    def _eval_eccentricity(_a: float, _b: float) -> float:
        return sqrt(1 - (_b * _b) / (_a * _a))

    @property
    def eccentricity(self):
        """
        Eccentricity of the ellipse, evaluated using

        .. math::
            e = \\sqrt{1 - \\frac{b^2}{a^2}}

        """
        self._ecc = self._eval_eccentricity(self.smj, self.smn)
        return self._ecc

    @property
    def locus(self):
        """
        Determines the points along the locus of the ellipse.

        .. math::
            x = a \\cos{ \\theta },  y = b \\sin{ \\theta }; \\;\\; \\theta \\in [\\theta_1, \\theta_2]
        """
        #
        self._locus = EllipticalArc(
            self.smj,
            self.smn,
            self.theta_1,
            self.theta_2,
            self.pivot_point,
            self.pivot_angle,
        ).locus
        return self._locus

    @property
    def bounding_box(self):
        """
        Returns the coordinate-axs aligned bounds of the ellipse using the following formulae

        .. math::
            x = x_c \\pm \\sqrt{a^2 \\cos^2 \\theta + b^2 \\sin^2 \\theta}

            y = y_c \\pm \\sqrt{a^2 \\sin^2 \\theta + b^2 \\cos^2 \\theta}

        """
        k1 = sqrt((self.smj ** 2) * (cos(self.pivot_angle) ** 2) + (self.smj ** 2) * (sin(self.pivot_angle) ** 2))
        k2 = sqrt((self.smj ** 2) * (sin(self.pivot_angle) ** 2) + (self.smj ** 2) * (cos(self.pivot_angle) ** 2))
        self._b_box = self.pxc - k1, self.pyc - k2, self.pxc + k1, self.pyc + k2
        return self._b_box

    def union_of_circles(self, buffer: float = 0.01) -> ClosedShapesList:
        """
        Returns union of circles representation for the Ellipse

        :param buffer: A small thickness around the shape to indicate the buffer region.

        :rtype: ClosedShapesList
        """
        assert buffer > 0.0, f"buffer must be a positive real number, but not {buffer}"
        e, e_outer = self.eccentricity, self._eval_eccentricity(self.smj + buffer, self.smn + buffer)
        zeta = e_outer / e
        k = self.smj * e * e
        xi = -k  # starting at -ae^2

        def r_shortest(_xi, _a, _b):
            return _b * sqrt(1.0 - ((_xi ** 2) / (_a ** 2 - _b ** 2)))

        circles = ClosedShapesList()
        while True:
            if xi > k:
                circles.append(Circle(self.smn * self.smn / self.smj, cent=(k, 0.0)))
                break
            ri = r_shortest(xi, self.smj, self.smn)
            circles.append(Circle(ri, cent=(xi, 0.0)))
            r_ip1 = r_shortest(xi, self.smj + buffer, self.smn + buffer)
            xi = (xi * ((2.0 * zeta * zeta) - 1.0)) + (2.0 * e_outer * zeta * sqrt(
                (r_ip1 * r_ip1) - (ri * ri)
            ))

        return circles


class Circle(Ellipse):
    """
    Inherits all the methods and properties from the `Ellipse()` using same semi-major and semi-minor axs lengths.
    """

    def __init__(self, radius=2.0, cent=(0.0, 0.0)):
        super().__init__(radius, radius, centre=cent)


class Polygon(ClosedShape2D):
    def __init__(self, vert: ndarray = None):
        super(Polygon, self).__init__()
        self.vertices = vert
        self._side_lengths = ()

    @property
    def area(self):
        """
        Evaluates the area of a polygon using the following formula

        """
        a = sum(self.vertices * roll(roll(self.vertices, 1, 0), 1, 1), axis=0)
        self._area = 0.5 * abs(a[0] - a[1])
        return self._area

    @property
    def side_lengths(self):
        self._side_lengths = sqrt(sum((self.vertices - roll(self.vertices, 1, axis=0)) ** 2, axis=1))
        return self._side_lengths

    @property
    def perimeter(self):
        self._perimeter = sum(self.side_lengths)
        return self._perimeter

    @property
    def locus(self):
        self._locus = Points(self.vertices)
        return self._locus

    @property
    def bounding_box(self):
        self._b_box = self.vertices.min(axis=0).tolist() + self.vertices.max(axis=0).tolist()
        return self._b_box


class RegularPolygon(ClosedShape2D):
    """
    Regular Polygon with `n`-sides

    """

    def __init__(self,
                 num_sides: int = 3,
                 corner_radius: float = 0.15,
                 side_len: float = 1.0,
                 centre: tuple[float, float] = (0.0, 0.0),
                 pivot_angle: float = 0.0,
                 ):
        """

        :param num_sides:  int, number of sides which must be greater than 2
        :param corner_radius: float, corner radius to add fillets
        :param side_len: float, side length
        :param centre: tuple[float, float], centre
        :param pivot_angle: float, A reference angle in radians, measured from the positive x-axs with the normal
            to the first side of the polygon.

        """
        assert_positivity(corner_radius, 'Corner radius', absolute=False)
        assert_range(num_sides, 3)
        #
        self.num_sides = int(num_sides)
        self.side_len = side_len
        self.alpha = pi / self.num_sides
        self.corner_radius = corner_radius
        #
        super(RegularPolygon, self).__init__(centre, pivot_angle)
        # crr: corner radius ratio should lie between [0, 1]
        self.crr = (2.0 * self.corner_radius * tan(self.alpha)) / self.side_len
        self.cot_alpha = cos(self.alpha) / sin(self.alpha)
        return

    @property
    def perimeter(self):
        """

        :rtype: float
        """
        self._perimeter = self.num_sides * self.side_len * (1.0 - self.crr + (self.crr * self.alpha * self.cot_alpha))
        return self._perimeter

    @property
    def area(self):
        self._area = 0.25 * self.num_sides * self.side_len * self.side_len * self.cot_alpha * (
                1.0 - ((self.crr * self.crr) * (1.0 - (self.alpha * self.cot_alpha)))
        )
        return self._area

    @property
    def locus(self):
        # TODO find the optimal number of points for each line segment and circular arc
        h = self.side_len - (2.0 * self.corner_radius * tan(self.alpha))
        r_ins = 0.5 * self.side_len * self.cot_alpha
        r_cir = 0.5 * self.side_len / sin(self.alpha)
        k = r_cir - (self.corner_radius / cos(self.alpha))
        # For each side: a straight line + a circular arc
        loci = []
        for j in range(self.num_sides):
            theta_j = 2.0 * j * self.alpha
            edge_i = StraightLine(h, rotate(r_ins, -0.5 * h, theta_j, 0.0, 0.0), (0.5 * pi) + theta_j).locus
            arc_i = CircularArc(self.corner_radius, -self.alpha, self.alpha, (0.0, 0.0), ).locus.transform(
                theta_j + self.alpha, k * cos(theta_j + self.alpha), k * sin(theta_j + self.alpha)
            )
            loci.extend([edge_i, arc_i])
        self._locus = Points(concatenate([a_loci.points[:-1, :] for a_loci in loci], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class Rectangle(ClosedShape2D):
    def __init__(
            self,
            smj=2.0,
            smn=1.0,
            rc: float = 0.0,
            centre=(0.0, 0.0),
            smj_angle=0.0
    ):
        is_ordered(smn, smj, 'Semi minor axs', 'Semi major axs')
        self.smj = smj
        self.smn = smn
        self.rc = rc
        super(Rectangle, self).__init__(centre, smj_angle)
        return

    @property
    def perimeter(self):
        self._perimeter = 4 * (self.smj + self.smn) - (2.0 * (4.0 - pi) * self.rc)
        return self._perimeter

    @property
    def area(self):
        self._area = (4.0 * self.smj * self.smn) - ((4.0 - pi) * self.rc * self.rc)
        return self._area

    @property
    def locus(self):
        a, b, r = self.smj, self.smn, self.rc
        aa, bb = 2.0 * (a - r), 2.0 * (b - r)
        curves = [
            StraightLine(bb, (a, -b + r), 0.5 * pi),
            CircularArc(r, 0.0 * pi, 0.5 * pi, (a - r, b - r)),
            StraightLine(aa, (a - r, b), 1.0 * pi),
            CircularArc(r, 0.5 * pi, 1.0 * pi, (r - a, b - r)),
            StraightLine(bb, (-a, b - r), 1.5 * pi),
            CircularArc(r, 1.0 * pi, 1.5 * pi, (r - a, r - b)),
            StraightLine(aa, (-a + r, -b), 2.0 * pi),
            CircularArc(r, 1.5 * pi, 2.0 * pi, (a - r, r - b)),
        ]
        self._locus = Points(concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class Capsule(Rectangle):
    def __init__(
            self,
            smj: float = 2.0,
            smn: float = 1.0,
            centre=(0.0, 0.0),
            smj_angle=0.0,
    ):
        super(Capsule, self).__init__(smj, smn, smn, centre, smj_angle)


class CShape(ClosedShape2D):
    def __init__(
            self,
            r_out=2.0,
            r_in=1.0,
            theta_c: float = 0.5 * pi,
            centre=(0.0, 0.0),
            pivot_angle: float = 0.0,
    ):
        is_ordered(r_in, r_out, 'Inner radius', 'Outer radius')
        self.r_in = r_in
        self.r_out = r_out
        self.r_tip = (r_out - r_in) * 0.5
        self.r_mean = (r_out + r_in) * 0.5
        self.theta_c = theta_c
        self.pivot_point = centre
        self.pivot_angle = pivot_angle
        super(CShape, self).__init__()
        return

    @property
    def perimeter(self):
        self._perimeter = (2.0 * pi * self.r_tip) + (2.0 * self.theta_c * self.r_mean)
        return self._perimeter

    @property
    def area(self):
        self._area = (pi * self.r_tip * self.r_tip) + (2.0 * self.theta_c * self.r_tip * self.r_mean)
        return self._area

    @property
    def locus(self):
        c_1 = rotate(self.r_mean, 0.0, self.theta_c, 0.0, 0.0)
        curves = [
            CircularArc(self.r_tip, pi, 2.0 * pi, (self.r_mean, 0.0), ),
            CircularArc(self.r_out, 0.0, self.theta_c, (0.0, 0.0), ),
            CircularArc(self.r_tip, self.theta_c, self.theta_c + pi, c_1, ),
            CircularArc(self.r_in, self.theta_c, 0.0, (0.0, 0.0)),
        ]
        self._locus = Points(concatenate([a_curve.locus.points[:-1, :] for a_curve in curves], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class NLobeShape(ClosedShape2D):

    def __init__(self,
                 num_lobes: int = 2,
                 r_lobe: float = 1.0,
                 ld_factor: float = 0.5,
                 centre=(0.0, 0.0),
                 pivot_angle: float = 0.0,
                 ):
        assert_range(num_lobes, 2, tag='Number of lobes')
        assert_range(ld_factor, 0.0, 1.0, False, 'lobe distance factor')
        self.pivot_point = centre
        self.pivot_angle = pivot_angle
        super(NLobeShape, self).__init__()
        #
        #
        self.num_lobes = int(num_lobes)
        self.r_lobe = r_lobe
        self.ld_factor = ld_factor
        self.alpha = pi / num_lobes
        #
        self.theta = arcsin(sin(self.alpha) * ((self.r_outer - r_lobe) / (2.0 * r_lobe)))
        #
        self._r_outer = None

    @property
    def r_outer(self):
        self._r_outer = self.r_lobe * (1.0 + ((1.0 + self.ld_factor) / sin(self.alpha)))
        return self._r_outer

    @property
    def perimeter(self):
        self._perimeter = 2.0 * self.num_lobes * self.r_lobe * (self.alpha + (2.0 * self.theta))
        return self._perimeter

    @property
    def area(self):
        self._area = self.num_lobes * self.r_lobe * self.r_lobe * (
                self.alpha + (2.0 * (1.0 + self.ld_factor) * sin(self.alpha + self.theta) / sin(self.alpha))
        )
        return self._area

    @property
    def locus(self):
        r_l, r_o = self.r_lobe, self.r_outer
        beta = self.theta + self.alpha
        c_1 = (r_o - r_l, 0.0)
        c_2 = (r_o - r_l + (2.0 * r_l * cos(beta)), 2.0 * r_l * sin(beta))
        curves = []
        for j in range(self.num_lobes):
            # Making in a lobe along the positive x-axs
            curve_1 = CircularArc(r_l, -beta, beta, c_1).locus
            curve_2 = CircularArc(r_l, -self.theta, self.theta).locus.transform(pi + self.alpha, *c_2)
            curve_2.reverse()
            # Rotating to the respective lobe direction
            beta_j = 2.0 * j * self.alpha
            curves.extend([curve_1.transform(beta_j), curve_2.transform(beta_j)])
        #
        self._locus = Points(concatenate([a_curve.points[:-1, :] for a_curve in curves], axis=0))
        self._locus.transform(self.pivot_angle, self.pxc, self.pyc)
        return self._locus


class BoundingBox2D(ClosedShape2D):
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
        super(BoundingBox2D, self).__init__(pivot_point=(self.xc, self.yc))

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
        return self._locus


class Circles(ClosedShapesList):
    def __init__(self, xyr: ndarray):
        self.validate_incl_data(xyr, 3)
        super(Circles, self).__init__()
        self.xc, self.yc, self.r = xyr.T
        self.extend([Circle(r, (x, y)) for (x, y, r) in xyr])


class Capsules(ClosedShapesList):
    def __init__(self, xyt_ab):
        self.validate_incl_data(xyt_ab, 5)
        super(Capsules, self).__init__()
        self.extend([Capsule(a, b, (x, y), tht) for (x, y, tht, a, b) in xyt_ab])


class RegularPolygons(ClosedShapesList):
    def __init__(self, xyt_arn):
        self.validate_incl_data(xyt_arn, 6)
        super(RegularPolygons, self).__init__()
        self.extend([RegularPolygon(n, rc, a, (x, y), tht) for (x, y, tht, a, rc, n) in xyt_arn])


class Ellipses(ClosedShapesList):
    def __init__(self, xyt_ab):
        self.validate_incl_data(xyt_ab, 5)
        super(Ellipses, self).__init__()
        self.extend([Ellipse(a, b, centre=(x, y), smj_angle=tht) for (x, y, tht, a, b) in xyt_ab])


class Rectangles(ClosedShapesList):
    def __init__(self, xyt_abr):
        self.validate_incl_data(xyt_abr, 6)
        super(Rectangles, self).__init__()
        self.extend([Rectangle(a, b, r, (x, y), tht) for (x, y, tht, a, b, r) in xyt_abr])


class CShapes(ClosedShapesList):
    def __init__(self, xyt_ro_ri_ang):
        self.validate_incl_data(xyt_ro_ri_ang, 6)
        super(CShapes, self).__init__()
        self.extend([CShape(ro, ri, ang, (x, y), tht) for (x, y, tht, ro, ri, ang) in xyt_ro_ri_ang])


class NLobeShapes(ClosedShapesList):
    def __init__(self, xyt_abr):
        self.validate_incl_data(xyt_abr, 6)
        super(NLobeShapes, self).__init__()
        self.extend([NLobeShape(a, b, r, (x, y), tht) for (x, y, tht, a, b, r) in xyt_abr])
