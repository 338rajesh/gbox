# This is geometry_box module file

"""
Assumptions:

* All the angular units are in the radians

"""


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
