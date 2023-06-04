import numpy as np


class Primitive:
    pass


class Point(Primitive):
    def __init__(self, *args):
        self.point = args
        self.dim = len(args)
        return


class Origin(Point):
    def __init__(self, n_dim=2):
        super(Origin, self).__init__(tuple(0 for _ in range(n_dim)))
        return


class Vector(Primitive):
    def __init__(self, p1: Point = Origin(), p2: Point = Origin()):
        self.p1 = p1
        self.p2 = p2
        return

    def len(self):
        return np.sqrt(sum([i * i + j * j for (i, j) in zip(self.p1.point, self.p2.point)]))

    def transform(self, rotation_angle=None, translation_vector=None, order='R_T'):
        return
