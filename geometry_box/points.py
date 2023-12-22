# from scipy.spatial import Voronoi
from numpy import (
    ndarray, array, concatenate,
    sin, cos,
)


def rotational_matrix(angle: float):
    return [[+cos(angle), sin(angle)], [-sin(angle), cos(angle)], ]


def rotate(x, y, angle, xc=0.0, yc=0.0, ):
    return tuple((array([[x - xc, y - yc]]) @ rotational_matrix(angle)).ravel())


class Points(list):
    def __init__(self, points: ndarray = None):
        super(Points, self).__init__()
        if points is None:
            points = array([[0.0, 0.0]])
        assert isinstance(points, ndarray), (
            "Points must be supplied as numpy.ndarray, with each column indicating a dimension"
        )
        self.points = points

    @property
    def dim(self):
        return self.points.shape[-1]

    def __len__(self):
        return self.points.shape[0]

    def append(self, new_points: ndarray):
        assert isinstance(new_points, ndarray), f"Only points of numpy.ndarray kind can be appended."
        assert self.points.ndim == new_points.ndim, (
            f"Inconsistent number of dimensions, {self.points.ndim} != {new_points.ndim}"
        )
        assert self.points.shape[-1] == new_points.shape[-1], "Inconsistent number of coordinates at a point."
        self.points = concatenate((self.points, new_points), axis=0)

    def close_loop(self):
        self.points = concatenate((self.points, self.points[0:1, ...]), axis=0)

    def transform(self, angle=0.0, dx=0.0, dy=0.0):
        """ Transforms the points cluster by rotation and translation """
        self.points = (self.points @ rotational_matrix(angle)) + [dx, dy]

    def make_periodic_tiles(self, bbox):
        assert bbox.dim == self.dim, "mismatch in points and bbox dimensions"
        periodic_points = []
        for i in range(3):  # shifting x
            for j in range(3):  # shifting y
                a_grid_points = concatenate((
                    (self.points[:, 0:1] - bbox.lx) + (i * bbox.lx),
                    (self.points[:, 1:2] - bbox.ly) + (j * bbox.ly),
                ), axis=1)
                if bbox.dim == 3:
                    for k in range(3):  # shifting z
                        a_grid_points = concatenate(
                            (a_grid_points, (self.points[:, 2:3] - bbox.lz) + (k * bbox.lz),),
                            axis=1
                        )
                periodic_points.append(a_grid_points)
        return concatenate(periodic_points, axis=0)

# TODO Voronoi tessellation
# TODO Voronoi Query
# TODO

#
# class PeriodicVoronoi:
#
#     def __init__(self, points: ndarray, bounding_box: tuple[float]):
#         self.points: ndarray = points
#         self.bbox: BoundingBox = BoundingBox(*bounding_box)
#         self.dim: int = points.shape[1]
#
#         assert self.dim == self.bbox.dim, "Mismatch in the dimension of the points and that of the bounding box"
