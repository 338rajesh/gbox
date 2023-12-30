from io import BytesIO
from math import inf
from multiprocessing import cpu_count

from matplotlib.pyplot import savefig
from numpy import reshape, frombuffer, uint8, sin, cos


def rotational_matrix(angle: float):
    """
    Rotational matrix for rotating a plane through angle in counter clock wise direction.

    >>> from math import pi
    >>> rotational_matrix(0.25 * pi)
    [[0.7071067811865476, 0.7071067811865476], [-0.7071067811865476, 0.7071067811865476]]

    :rtype: list[list]
    """
    return [[+cos(angle), sin(angle)], [-sin(angle), cos(angle)], ]


def get_pairs(a: list, loop=False) -> list:
    if loop:
        a.append(a[0])
    return [i for i in zip(a[:-1], a[1:])]


# ==================================
#           Multi-Processing Utils
# ==================================


def validated_num_cores(n):
    assert isinstance(n, int), "Number of cores must be an integer"
    if n > cpu_count():
        print("Given number of cores greater than available, setting to maximum.")
        n = cpu_count()
    return n


def assert_positivity(k, tag: str = None, val_type=float, absolute=True):
    assert isinstance(k, val_type), f"Invalid type of {tag}"
    assert k is not None and (k > 0.0 if absolute else k >= 0.0), (
        f"{'It' if tag is None else str(tag)} must be a positive real number but not {k}"
    )


def assert_range(k, mn=-inf, mx=inf, closed=True, tag=None):
    assert (mn <= k <= mx if closed else mn < k < mx), (
        f"The {tag if tag is not None else 'Value'} {k} is out of the bounds [{mn}, {mx}]."
    )


def is_ordered(a, b, am: str, bm: str):
    assert a <= b, f"{am}: {a} > {bm}: {b} "


# ===================================
#       PLOTTING UTILITIES
# ===================================
def get_fig_array(_fig):
    io_buffer = BytesIO()
    savefig(io_buffer, format="raw")
    io_buffer.seek(0)
    _image_array = reshape(
        frombuffer(io_buffer.getvalue(), dtype=uint8),
        newshape=(int(_fig.bbox.bounds[3]), int(_fig.bbox.bounds[2]), -1)
    )
    io_buffer.close()
    return _image_array


class PlotOptions:
    def __init__(self):
        self._face_color = 'g'
        self._edge_color = 'k'
        self._hide_axis = True
        self._show_grid = False
        self._linewidth = 2.0

    @property
    def face_color(self):
        return self._face_color

    @face_color.setter
    def face_color(self, val):
        self._face_color = val

    @property
    def edge_color(self):
        return self._edge_color

    @edge_color.setter
    def edge_color(self, val):
        self._edge_color = val

    @property
    def hide_axes(self):
        return self._hide_axis

    @hide_axes.setter
    def hide_axes(self, val):
        self._hide_axis = val

    @property
    def show_grid(self):
        return self._show_grid

    @show_grid.setter
    def show_grid(self, val):
        self._show_grid = val

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, val):
        self._linewidth = val


PLOT_OPTIONS = PlotOptions()
