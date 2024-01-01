from os import path

from PIL.Image import fromarray
from h5py import File
from matplotlib.pyplot import figure, Axes, savefig, axis, xlim, ylim, clf
from matplotlib.pyplot import show
from numpy import asarray, transpose, ndarray, savez_compressed, concatenate
from numpy import load as np_load
from numpy import save as np_save
from tqdm import tqdm

from ..gbox import BoundingBox2D, Circles, RegularPolygons, Ellipses, Rectangles, CShapes, NLobeShapes, Points
from ..gbox import ClosedShapesList
from ..utils import get_fig_array

"""


## Plotting 

The unit cell plotting considers the presence of inclusions of different shapes. If all the inclusions in unit cell 
are of same shape then, simply provide that single shape information.

+ data needed for plotting unit cell:
    bounds of each unit cell
    inclusion shape and the respective shape's data

+ data input streams
    Single Unit Cell
    1. *.npz file with
        'bounds' key containing four values in the order of x_min, y_min, x_max, y_max of the unit cell
        and the inclusion shapes as key and the data as value. For example, for circular inclusions 'Circle' key 
        should contain (n, 3) shaped array with n being the number of circles and the three columns
        indicating x, y and radius of the respective circles. Other shapes can be added in the same way. 
    1. *.json file 
    1. *.dat or *.txt file
        
    Batch of Unit Cells: It is recommended to use the HDF5 format that has file directory kind of structure, 
    but one can always use Single Unit Cell plotting function in a loop.
    1. *.h5 file with the following structure.
        \root\
            |
            |___UnitCell-1 group with attributes 'xlb', 'xub', 'ylb', 'yub' and contains data sets of various 
            |   |   inclusion shapes information present in this unit cell
            |   |
            |   |___Shape-1 data-set with spatial and size information
            |   |___Shape-2 data-set with spatial and size information
            |   |___Shape-3 data-set with spatial and size information
            |   
            |
            |___UnitCell-2 group with attributes 'xlb', 'xub', 'ylb', 'yub' and contains data sets of various 
            |   |   inclusion shapes information present in this unit cell  
            |   |
            |   |___Shape-1 data-set with spatial and size information
            |   |___Shape-2 data-set with spatial and size information
            |   
            |               

"""


class UnitCell:
    def __init__(self):
        pass


class UnitCell2D(UnitCell):
    _default_periodicity = True

    def __init__(
            self,
            bounds: list = None,
            inclusions: dict = None,
            npz_fp: str = None
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
        if bounds is None and inclusions is None:
            if npz_fp is not None:
                pass
            else:
                raise FileNotFoundError("The bounds and inclusions source is missing")
        self.bounds = bounds
        self.matrix = BoundingBox2D(*bounds)
        self.inclusions = Inclusions(inclusions)
        #
        self.voronoi = None

    def _plot(
            self,
            matrix_color='black',
            matrix_edge_color='None',
            inclusion_color: str | dict = 'white',
            inclusion_edge_color: str | dict = 'None',
            inclusion_edge_thickness=1.0,
            pixels=(256, 256),

    ):
        fig = figure(num=0, figsize=(pixels[0] * 0.01, pixels[1] * 0.01), frameon=False)
        ax = Axes(fig, rect=[0., 0., 1., 1.])
        fig.add_axes(ax)
        self.matrix.plot(
            axis=ax,
            face_color=matrix_color,
            edge_color=matrix_edge_color,
        )
        self.inclusions.shapes.plot(
            axis=ax,
            face_color=inclusion_color,
            edge_color=inclusion_edge_color,
            linewidth=inclusion_edge_thickness,
        )
        axis('off')
        xlim([self.matrix.xlb, self.matrix.xub])
        ylim([self.matrix.ylb, self.matrix.yub])
        return fig, ax

    def plot(
            self,
            file_path=None,
            image_extension='png',
            matrix_color='black',
            matrix_edge_color='None',
            inclusion_color: str | dict = 'white',
            inclusion_edge_color: str | dict = 'None',
            inclusion_edge_thickness=1.0,
            pixels=(256, 256),
    ):
        """
            Plots a single two-dimensional unit cell.

        :param file_path: str, if image extension is provided, then it will override the file_path extension.
        :param image_extension: str, Image extensions like 'png', 'pdf', 'jpg'. Defaults to 'png'.
        :param matrix_color: str, Color of the continuous phase
        :param matrix_edge_color: str, Unit cell edge color
        :param inclusion_color: str, Inclusion(s) color.
        :param inclusion_edge_color: str,
        :param inclusion_edge_thickness: float, thickness of the inclusion edge
        :param pixels: tuple[float, float] Image resolution

        :rtype: None
        """
        self._plot(
            matrix_color, matrix_edge_color, inclusion_color, inclusion_edge_color, inclusion_edge_thickness, pixels
        )
        if file_path is None:  # If file_path is not provided, then show the image.
            show()
        else:
            # Overriding the file_path extension with image_extension option, if they aren't same.
            if not file_path.split(".")[-1] == image_extension:
                file_path = ".".join(file_path.split(".")[:-1] + [image_extension, ])
            savefig(file_path)
        clf()

    def get_image_array(
            self,
            matrix_color='black',
            matrix_edge_color='None',
            inclusion_color: str | dict = 'white',
            inclusion_edge_color: str | dict = 'None',
            inclusion_edge_thickness=1.0,
            pixels=(256, 256),
            image_mode='L',
            dither=0,
    ):
        """

        :param matrix_color:
        :param matrix_edge_color:
        :param inclusion_color:
        :param inclusion_edge_color:
        :param inclusion_edge_thickness:
        :param pixels:
        :param image_mode: Image modes as used by PIL.Image. Defaults to grayscale with code 'L'
            for a list of image modes see: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
        :param dither:
        :return: A numpy array with pixel values in uint8 format.
        :rtype: ndarray[uint8]
        """
        fig, axs = self._plot(
            matrix_color, matrix_edge_color, inclusion_color, inclusion_edge_color, inclusion_edge_thickness, pixels
        )
        return asarray(fromarray(get_fig_array(fig)).convert(mode=image_mode, dither=dither))

    def eval_voronoi(self, periodic=_default_periodicity):
        self.voronoi = Points(self.inclusions.pivot_points()).eval_voronoi(
            tile_order=(1 if periodic else 0), clip=True, bounds=self.bounds
        )
        return self

    def plot_voronoi(self, axs=None, show_fig=False, file_path=None, bounds=None, periodic=_default_periodicity):
        if self.voronoi is None:
            self.eval_voronoi(periodic)
        self.voronoi.plot_voronoi(axs, show_fig, file_path, self.bounds)

    def eval_metrics(self, metrics=(), periodic=_default_periodicity):
        if self.voronoi is None:
            self.eval_voronoi(periodic)
        vc_vertices = self.voronoi.cells_vertices
        vc_neighbours = self.voronoi.cells_neighbours
        for a_metric in metrics:
            if a_metric == 'voronoi_cells_area':
                pass
            elif a_metric in ('first_nnd', 'second_nnd', 'third_nnd'):
                pass
            elif a_metric == 'first_nna':
                pass
        return


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
        self.data = inclusions_data

    def pivot_points(self):
        return concatenate([v[:, 0:2] for v in self.data.values()], axis=0)


def parse_unit_cell_file_data(data_fp: str | list[str]) -> UnitCell2D | dict[str, UnitCell2D]:
    """ This function extracts the unit-cell information from various types of inputs. The extracted information is
    returned as a single UnitCell2D instance (if input contains single unit-cell) or a python dictionary of multiple
    UnitCell2D instances (if input contains multiple unit-cells).

    :param data_fp: str | list[str], file path of the unit-cells data. This can be either a single .h5 file or a
     list of file paths which contain single unit cell information.
    :return: A single UnitCell2D instance or a dictionary containing multiple UnitCell2D instances with its
     identifier as key.
    :rtype: dict[str, UnitCell2D] | UnitCell2D

    """

    def _npz_parser(_fp):
        npz_data = dict(np_load(_fp))
        assert 'bounds' in npz_data.keys(), (
            f"NPZ file must contain 'bounds' key with unit cell bounds as (x_min, y_min, x_max, y_max)."
        )
        return UnitCell2D(list(npz_data.pop('bounds')), npz_data)

    inclusions: dict[str, UnitCell2D] = {}
    if isinstance(data_fp, str):  # a single file path that contains data of all unit-cells
        f_extn = data_fp.split(".")[-1]
        if f_extn == "h5":
            h5_fp = File(data_fp, mode='r')
            for (k, v) in h5_fp.items():
                inclusions[k] = UnitCell2D(
                    [v.attrs[i] for i in ('xlb', 'ylb', 'xub', 'yub')],
                    {ak: transpose(av) for (ak, av) in v.items()}
                )
            h5_fp.close()
        elif f_extn == "npz":
            return _npz_parser(data_fp)
        else:
            assert data_fp.split(".")[-1] == "h5", f"Expecting a *.h5 file but {f_extn} is found."
    elif all(isinstance(i, str) for i in data_fp):
        for a_data_fp in data_fp:
            file_name, f_extn = path.basename(a_data_fp).split(".")
            if f_extn == "npz":
                inclusions[file_name] = _npz_parser(a_data_fp)
            elif f_extn == "json":
                pass  # TODO
            elif f_extn in ("dat", "txt"):
                pass  # TODO
            else:
                print(f"Unsupported file type {f_extn}.")
            #
    else:
        raise TypeError(f"data must be either a single file path or list of file paths")
    return inclusions


class UnitCells2D(dict):
    """
        A subclass of python dictionary to hold unit cells, whose with a typical key-value pair looks as follows

            'Unit-cell-i': UnitCell2D()
            # 'Unit-cell-i': {'bounds': bounds, 'shape-1': shape_1_data, 'shape-2': shape_2_data,...,}


    """

    def __init__(self, data_fp: str | list[str]):
        super(UnitCells2D, self).__init__()
        self.update(parse_unit_cell_file_data(data_fp))  # Updating the dictionary with parsed data

    def plot(
            self,
            log_dir: str,
            num_cores=None,
            image_extension="png",
            p_bar=False,
            **kwargs
    ):
        """

        :param log_dir: Directory to save the unit cell plots
        :param num_cores: Number of cores to speed up the plotting. With this options make sure that program is run
         under ``if __name__ == '__main__'``
        :param image_extension: The image extension like 'png', 'pdf'...
        :param p_bar: Enable/Disable the progress bar.
        :param kwargs: All the key-word arguments needed by ``UnitCell2D.plot()``
        :return: None
        """
        if num_cores is None:
            loop = tqdm(self.items(), desc="Plotting unit cells") if p_bar else self.items()
            for (k, v) in loop:
                v.plot(file_path=path.join(log_dir, f"{k}.{image_extension}"), **kwargs)
        else:
            raise NotImplementedError("At present multiprocessing is not supported.")
            # num_cores = validated_num_cores(num_cores)
            # pool = Pool()
            # pool.imap()
            # for (k, v) in self.items():
            #     v.plot(file_path=path.join(log_dir, f"{k}.{image_extension}"), **kwargs)
            # pool.close()
            # pool.join()

    def get_images_array(
            self,
            file_path=None,
            num_cores=None,
            p_bar=False,
            **kwargs
    ):
        """

        :param file_path:
        :param num_cores:
        :param p_bar:
        :param kwargs:
        :return:
        """
        if num_cores is None:
            images_arrays = []
            loop = tqdm(self.values(), desc="Getting image arrays") if p_bar else self.values()
            for v in loop:
                images_arrays.append(v.get_image_array(**kwargs))
            if file_path is None:
                return asarray(images_arrays)
            else:
                if file_path.split(".")[-1] == "npy":
                    np_save(file_path, asarray(images_arrays))
                elif file_path.split(".")[-1] == "npz":
                    savez_compressed(file_path, asarray(images_arrays))
                else:
                    raise ValueError("Invalid file extension. It can be either *.npy or *.npz")
        else:
            # uc_instances = [i.get_image_array for i in self.values()]
            raise NotImplementedError("At present multiprocessing is not supported.")
