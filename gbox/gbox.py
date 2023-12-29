# This is gbox module file

"""
Assumptions:

* All the angular units are in the radians

"""
from matplotlib.pyplot import subplots, show, savefig, close
from numpy import ndarray, pi, sqrt, stack

from .points import Points
from .utils import PLOT_OPTIONS, assert_positivity


class ShapePlotter:
    """
        Plotter for various shapes
    """

    def __init__(
            self,
            locus: Points,
            axis=None,
            f_path=None,
            closure=True,
            linewidth=None,
            show_grid=None,
            hide_axes=None,
            face_color=None,
            edge_color=None,
    ):
        """

        :param axis: Shape is plotted on this axis and returns the same, If not provided, a figure will be created
         with default options which will be saved at `f_path` location if the `f_path` is specified.
         Otherwise, it will be displayed using matplotlib.pyplot.show() method.
        :param f_path: str, file path to save the figure
        :param closure: Whether to make loop by connecting the last point with the first point.
        :param face_color: str, Color to fill the shape
        :param edge_color: str, Color of the edge
        :param linewidth: float,
        :param show_grid: bool, enable/disable the grid on figure
        :param hide_axes: bool, enable/disable the axis on figure
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
        self.axis = axis
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
        _axs.axis('equal')
        if title is not None:
            _axs.set_title(title)
        if self.show_grid:
            _axs.grid()
        if self.hide_axes:
            _axs.axis('off')
        return _axs

    def _plot(self, fill_plot=True, title=None, **plt_opt):

        def _plt():
            if fill_plot:
                self._plot_on_axis(self.axis, fill=True, title=title, **plt_opt)
            else:
                self._plot_on_axis(self.axis, fill=False, title=title, **plt_opt)

        if self.axis is None:
            _, self.axis = subplots(1, 1)
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
    def num_locus_points(self):
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


class Curve2D(Shape2D):
    """ Curve in tw-dimensional space """

    def plot(
            self,
            axis=None,
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
            self.locus, axis, f_path, closure, linewidth, show_grid, hide_axes,
            edge_color=edge_color,
        ).line_plot(title=title, **plt_opt)


class ClosedShape2D(Shape2D):
    """
        Closed Shape in the two-dimensional space or a plane is defined by
        the locus of points, pivot point (lying on or inside or outside) the locus and angle made by a pivot axis.
        The pivot point and axis are used for convenience and are set to `(0.0, 0.0)` and 0.0 degrees by default.
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
            axis=None,
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
            self.locus, axis, f_path, closure, linewidth, show_grid, hide_axes,
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
