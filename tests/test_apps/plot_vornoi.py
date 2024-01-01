from os import path
import numpy as np

import gbox as gb

points = np.random.rand(500, 2)

vor_cells = gb.gbox.VoronoiCells(
    points,
    tile_order=1,
    clip=True,
    bounds=(0.0, 0.0, 1.0, 1.0),
)
vor_cells.plot(
    file_path=path.join(path.dirname(__file__), "vor.png"),
    ridge_plt_opt={'face_color': 'None', 'edge_color': 'r', 'linewidth': 1.0},
    pnt_plt_opt={'c': 'b', 's': 5.0, 'marker': '*'},
    bounds_plt_opt={'face_color': 'None', 'edge_color': 'g', 'linewidth': 1.0},
)
print(vor_cells.cells_volume[:5])
