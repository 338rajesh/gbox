"""

ACTIONS_TODO

+ Get Voronoi tessellation
+ Voronoi query
+ Query
+

"""

import numpy as np
from scipy.spatial import Voronoi
from .shapes2D import BoundingBox


class PeriodicVoronoi:

    def __init__(self, points: np.ndarray, bounding_box: tuple[float]):
        self.points: np.ndarray = points
        self.bbox: BoundingBox = BoundingBox(*bounding_box)
        self.dim: int = points.shape[1]

        assert self.dim == self.bbox.dim, "Mismatch in the dimension of the points and that of the bounding box"


