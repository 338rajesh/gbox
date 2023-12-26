# This is gbox module file

"""
Assumptions:

* All the angular units are in the radians

"""
from matplotlib.pyplot import subplots, show, savefig, close
from numpy import ndarray, pi, sqrt, stack

from .points import Points
from .utils import PLOT_OPTIONS, assert_positivity


class Shape:
    pass


class Shape2D(Shape):
    def __init__(self):
        self._locus: Points = Points()
        self._num_locus_points: int = 100

    @property
    def num_locus_points(self):
        return self._num_locus_points

    @num_locus_points.setter
    def num_locus_points(self, val):
        assert assert_positivity(val, val_type=int)
        self._num_locus_points = val

    @property
    def locus(self):
        return self._locus

    @locus.setter
    def locus(self, value):
        if isinstance(value, ndarray):
            self._locus = Points(value)
        elif isinstance(value, Points):
            self._locus = value
        else:
            raise TypeError(f"locus must be either 'numpy.ndarray' or 'Points' type but not {type(value)}")


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
        return self._area

    @property
    def perimeter(self):
        return self._perimeter

    @property
    def shape_factor(self):
        assert_positivity(self.area, 'Area')
        assert_positivity(self.perimeter, 'Perimeter')
        self._sf = self.perimeter / sqrt(4.0 * pi * self.area)
        return self._sf

    def plot(
            self,
            axis=None,
            f_path=None,
            closure=True,
            face_color=None,
            edge_color=None,
            linewidth=None,
            show_grid=None,
            hide_axes=None,
            **plt_opt
    ):
        """

        :param axis: Shape is plotted on this axis and returns the same, If not provided, a figure will be created
         with default options which will be saved at `f_path` location if the `f_path` is specified.
         Otherwise, it will be displayed using matplotlib.pyplot.show() method.
        :param f_path:
        :param closure: Whether to make loop by connecting the last point with the first point.
        :param face_color: Color to fill the shape
        :param edge_color: Color
        :param linewidth:
        :param show_grid:
        :param hide_axes:
        :param plt_opt: The plotting key-word arguments, that are taken by `matplotlib.patches.Polygon()`.
        :return:
        """
        if face_color is None:
            face_color = PLOT_OPTIONS.face_color
        if edge_color is None:
            edge_color = PLOT_OPTIONS.edge_color
        if show_grid is None:
            show_grid = PLOT_OPTIONS.show_grid
        if hide_axes is None:
            hide_axes = PLOT_OPTIONS.hide_axes
        if linewidth is None:
            linewidth = PLOT_OPTIONS.linewidth

        assert self.locus is not None, "Plotting a shape requires locus but it is set to `None` at present."
        if closure:
            self.locus.close_loop()

        def _plot(_axs):
            _axs.fill(
                self.locus.points[:, 0], self.locus.points[:, 1],
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                **plt_opt
            )
            _axs.axis('equal')
            if show_grid:
                _axs.grid()
            if hide_axes:
                _axs.axis('off')
            return _axs

        if axis is None:
            _, axis = subplots(1, 1)
            _plot(axis)
            if f_path is None:
                show()
            else:
                savefig(f_path)
                close('all')
        else:
            return _plot(axis)


class ShapesList(list):

    def __init__(self):
        super(ShapesList, self).__init__()
        self._loci = Points()
        self._perimeters = ()
        self._areas = ()
        self._shape_factors = ()

    def plot(self, *args, **kwargs):
        for i in range(self.__len__()):
            self.__getitem__(i).plot(*args, **kwargs)

    @property
    def loci(self):
        self._loci = Points(stack([self.__getitem__(i).locus.points for i in range(self.__len__())], axis=0))
        return self._loci

    @property
    def perimeters(self):
        self._perimeters = tuple(self.__getitem__(i).perimeter for i in range(self.__len__()))
        return self._perimeters

    @property
    def areas(self):
        self._areas = tuple(self.__getitem__(i).area for i in range(self.__len__()))
        return self._areas

    @property
    def shape_factors(self):
        self._shape_factors = tuple(self.__getitem__(i).shape_factor for i in range(self.__len__()))
        return self._shape_factors


class ClosedShapesList(ShapesList):
    def __init__(self):
        super(ClosedShapesList, self).__init__()

    @staticmethod
    def validate_incl_data(a, n):
        assert isinstance(a, ndarray), "given inclusion data must be an numpy.ndarray"
        assert a.shape[1] == n, f"Incorrect number of columns, found {a.shape[1]} instead of {n}"
        return
