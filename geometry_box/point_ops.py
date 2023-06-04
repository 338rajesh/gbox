import numpy as np
from .primitives import Point
from .shapes2D import BoundingBox


class PointsCluster:
    def __init__(self, points: list[Point] | np.ndarray):
        self.points = points
        self.dim = points.shape[1]
        return

    def make_periodic_tiles(self, bbox: BoundingBox, buffer_factor: float = 1.0):
        assert bbox.dim == self.dim, "mismatch in points and bbox dimensions"
        periodic_points = []
        for i in range(3):  # shifting x
            for j in range(3):  # shifting y
                a_grid_points = np.concatenate((
                    (self.points[:, 0:1] - bbox.lx) + (i * bbox.lx),
                    (self.points[:, 1:2] - bbox.ly) + (j * bbox.ly),
                ), axis=1)
                if bbox.dim == 3:
                    for k in range(3):  # shifting z
                        a_grid_points = np.concatenate(
                            (a_grid_points, (self.points[:, 2:3] - bbox.lz) + (k * bbox.lz),),
                            axis=1
                        )
                periodic_points.append(a_grid_points)

        return np.concatenate(periodic_points, axis=0)
