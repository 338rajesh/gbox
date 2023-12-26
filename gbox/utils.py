from math import inf


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
