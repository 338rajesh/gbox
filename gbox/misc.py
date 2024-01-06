from matplotlib.pyplot import subplots, show, savefig, axis, tight_layout
from numpy import ndarray, array
from scipy.spatial import Voronoi, ConvexHull

from .elements import Points, BoundingBox2D, Point
from .shapes import Polygon
from .utils import get_pairs


class VoronoiCells(Voronoi):
    def __init__(self, points: ndarray, tile_order=0, clip=False, bounds=None, **kwargs):
        # tiling points if required, otherwise points are passed as they are
        if tile_order > 0:
            points = Points(points).make_periodic_tiles(bounds, tile_order).points
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
