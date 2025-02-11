
class Polygon(TopologicalClosedShape2D):
    def __init__(self, vertices: NDArray = None):
        super(Polygon, self).__init__()
        self.vertices = vertices


class RegularPolygon(Polygon):
    pass


class Rectangle(TopologicalClosedShape2D):
    pass