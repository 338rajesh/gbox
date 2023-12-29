from io import BytesIO

from PIL.Image import fromarray
from matplotlib.pyplot import figure, Axes, savefig, axis, xlim, ylim, clf
from numpy import ndarray, reshape, frombuffer, uint8

from ..closed_shapes import Circles, Ellipses, Rectangles, RegularPolygons, NLobeShapes, CShapes, BoundingBox2D
from ..gbox import ClosedShapesList

"""
# TODO

+ data:
    bounds of each unit cell
    inclusion shape and the respective shape's data

+ data input streams
    Single Unit Cell
    1. *.npz file with
        'bounds' key containing four values in the order of x_min, y_min, x_max, y_max of the unit cell
        and the inclusion shapes as key and the data as value. For example, for circular inclusions 'Circle' key 
        should contain (n, 3) shaped array with n being the number of circles and the three columns
        indicating x, y and radius of the respective circles. 
    1. *.json file 
        
    Batch of Unit Cells
    1. *.h5 file

"""


class Inclusions:

    @staticmethod
    def _shapes(incl_shape: str, incl_data: ndarray):
        incl_shape = incl_shape.upper()
        if incl_shape == "CIRCLE":
            a_shape = Circles
        elif incl_shape == "ELLIPSE":
            a_shape = Ellipses
        elif incl_shape == "RECTANGLE":
            a_shape = Rectangles
        elif incl_shape == "REGULARPOLYGON":
            a_shape = RegularPolygons
        elif incl_shape == "NLOBESHAPE":
            a_shape = NLobeShapes
        elif incl_shape == "CSHAPE":
            a_shape = CShapes
        else:
            raise KeyError(f"Invalid Shape Key: {incl_shape}.")
        return a_shape(incl_data)

    def __init__(self, inclusions_data: dict):
        shapes = ClosedShapesList()
        for (k, v) in inclusions_data.items():
            shapes.extend(self._shapes(k, v))
        self.shapes = shapes


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


class UnitCell:
    def __init__(self):
        pass


class UnitCell2D(UnitCell):
    def __init__(
            self,
            bounds: list,
            inclusions: dict,
    ):
        """
        Unit Cell in two-dimensional space defined by bounds of the continuous phase (matrix) as a list and
        the reinforcements

        :param bounds: A list of `x_min`, `y_min`, `x_max`, `y_max` in the same order
        :param inclusions: A python dictionary with keys representing inclusion shapes and values representing
        spatial and shape information.

        >>> unit_cell = UnitCell2D(\
            [-1.0, -1.0, 1.0, 1.0],\
             {"CIRCLE": [[0.0, 0.0, 1.0], [5.0, 5.0, 2.0]], "ELLIPSE": [[0.0, 5.0, 0.0, 2.0, 1.0]]} \
        )

        """
        super(UnitCell2D, self).__init__()
        self.matrix = BoundingBox2D(*bounds)
        self.inclusions = Inclusions(inclusions).shapes

    def plot(
            self,
            image_extension='png',
            matrix_color='black',
            matrix_edge_color='None',
            inclusion_color='white',
            inclusion_edge_color='None',
            fibre_edge_thickness=1.0,
            pixels=(256, 256),
            image_mode='L',
            dither: bool = 0,
            file_path=None,
    ):
        fig = figure(num=0, figsize=(pixels[0] * 0.01, pixels[1] * 0.01), frameon=False)
        ax = Axes(fig, rect=[0., 0., 1., 1.])
        fig.add_axes(ax)
        self.matrix.plot(
            axis=ax,
            face_color=matrix_color,
            edge_color=matrix_edge_color,
        )
        self.inclusions.plot(
            axis=ax,
            face_color=inclusion_color,
            edge_color=inclusion_edge_color,
            linewidth=fibre_edge_thickness,
        )
        axis('off')
        xlim([self.matrix.xlb, self.matrix.xub])
        ylim([self.matrix.ylb, self.matrix.yub])
        image_array = get_fig_array(fig)
        #
        if file_path is not None:
            if not file_path.split(".")[-1] == image_extension:
                file_path = ".".join(file_path.split(".")[:-1] + [image_extension, ])
            if image_mode in ('L', '1'):
                fromarray(image_array).convert(mode=image_mode, dither=dither).save(file_path)
            else:
                savefig(file_path)
        clf()
        return fig, ax, image_array
