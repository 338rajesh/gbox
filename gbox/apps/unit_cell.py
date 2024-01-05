from os import path

from PIL.Image import fromarray
from h5py import File
from matplotlib.pyplot import figure, Axes, savefig, axis, xlim, ylim, clf
from matplotlib.pyplot import show
from numpy import asarray, transpose, ndarray, savez_compressed, concatenate
from numpy import load as np_load
from numpy import save as np_save
from tqdm import tqdm

from ..gbox import BoundingBox2D, Circles, RegularPolygons, Ellipses, Rectangles, CShapes, NLobeShapes
from ..gbox import ClosedShapesList
from ..gbox import VoronoiCells
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


def parse_2d_unit_cell_file_data(
        data_fp: str | list[str], periodic=False
) -> dict[str, dict[str, tuple[list, dict[str, ndarray], bool]]]:
    """ This function extracts the unit-cell information from various types of input files. The extracted information
    is returned as a python dictionary with each key being the file name and the value being the extracted  information
    of the unit cell(s). Here, the structure of the output is as follows,

    - output is a dictionary with {file_name: uc_data} key, value pairs; wherein
    - uc_data is a dictionary with {a_unit_cell_id: a_unit_cell_data} key, value pairs; wherein
    - a_unit_cell_data is a 3-tuple with (bounds, a_uc_incl_data, periodicity); wherein
    - bounds is a list of bounds
    - a_uc_incl_data is a dictionary with {inclusion_shape: inclusion_data} key, value pairs

    :param data_fp: str | list[str], file path of the unit-cells data. This can be either a single .h5 file or a
     list of file paths which contain single unit cell information.
    :param periodic: Specify the default periodicity option, if raw data contains `periodic` key in `npz` or
     `periodic` attribute in `.h5` file, it will preside over this setting.
    :return: A dictionary with the key, value pairs as explained above.

    """

    def _npz_parser(_fp) -> dict[str, tuple[list, dict[str, ndarray], bool]]:
        """
            A *.npz expected to contain SINGLE UNIT CELL information
        """
        npz_data = dict(np_load(_fp))
        _periodic = npz_data.pop('periodic') if 'periodic' in npz_data.keys() else periodic
        assert 'bounds' in npz_data.keys(), (
            f"NPZ file must contain 'bounds' key with unit cell bounds as (x_min, y_min, x_max, y_max)."
        )
        _bounds = list(npz_data.pop('bounds'))
        return {'unit-cell': (_bounds, npz_data, _periodic)}

    def _h5_parser(_fp) -> dict[str, tuple[list, dict[str, ndarray], bool]]:
        """
            It returns multiple unit cells information as a dictionary
        """
        _uc_info = {}
        h5_fp = File(_fp, mode='r')
        for (k, v) in h5_fp.items():
            _uc_info[k] = (
                [v.attrs[i] for i in ('xlb', 'ylb', 'xub', 'yub')],
                {ak: transpose(av) for (ak, av) in v.items()},
                v.attrs['periodic'] if 'periodic' in v.attrs.keys() else periodic,
            )
        h5_fp.close()
        return _uc_info

    def _json_parser(_fp):
        raise NotImplementedError("!! At present JSON file data is not supported !!")

    def _dat_parser(_fp):
        raise NotImplementedError("!! At present DAT file data is not supported !!")

    def _a_single_file_parser(a_fp) -> dict[str, tuple[list, dict[str, ndarray], list]]:
        f_extn = a_fp.split(".")[-1]
        if f_extn == "h5":
            return _h5_parser(a_fp)
        elif f_extn == "npz":
            return _npz_parser(a_fp)
        elif f_extn == "json":
            return _json_parser(a_fp)
        elif f_extn in ("dat", "txt"):
            return _dat_parser(a_fp)
        else:
            raise ValueError(f"Invalid file extension: {f_extn}")

    uc_info = {}
    if isinstance(data_fp, str):  # a single file path that contains data of all unit-cells
        f_name, _ = path.basename(data_fp).split(".")
        # Possible types: *.h5, *.json, *.npz, *.dat, *.txt
        uc_info[f_name] = _a_single_file_parser(data_fp)
    elif all(isinstance(i, str) for i in data_fp):
        for a_data_fp in data_fp:
            f_name, _ = path.basename(a_data_fp).split(".")
            # Possible types: *.h5, *.json, *.npz, *.dat, *.txt
            uc_info[f_name] = _a_single_file_parser(a_data_fp)
    else:
        raise TypeError(f"data must be either a single file path or list of file paths")
    return uc_info


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


class UnitCell:
    def __init__(self, bounds, inclusions, periodic):
        self.bounds: list = bounds
        self.inclusions: dict = inclusions
        self._periodicity: bool = periodic

    @property
    def periodicity(self):
        return self._periodicity


class UnitCell2D(UnitCell):
    """ Unit Cell in two-dimensional space """

    def __init__(
            self,
            bounds: list,
            inclusions: dict,
            periodic: bool = True,
    ):
        """

        :param bounds: A list of `x_min`, `y_min`, `x_max`, `y_max` in order
        :param inclusions: A python dictionary with keys representing inclusion shapes and values representing spatial
         and shape information.
        :param periodic: Indicating the periodicity of the unit-cell

        """
        super(UnitCell2D, self).__init__(bounds, inclusions, periodic)
        #
        self.matrix = BoundingBox2D(*self.bounds)
        self.inclusions = Inclusions(self.inclusions)
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
            axs=ax,
            face_color=matrix_color,
            edge_color=matrix_edge_color,
        )
        self.inclusions.shapes.plot(
            axs=ax,
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

    def eval_voronoi(self, tile_order=0, clip=True):
        """
        Evaluates the Voronoi diagram of the current unit cell
        :return:
        """
        self.voronoi = VoronoiCells(
            self.inclusions.pivot_points(),
            tile_order=1 if self.periodicity else tile_order,
            clip=clip,
            bounds=self.bounds,
        )
        return self

    def plot_voronoi_diagram(
            self,
            axs=None,
            show_fig=False,
            file_path=None,
            **kwargs,
    ):
        """
            Plots the voronoi diagram of the current unit cell
        :param axs:
        :param show_fig:
        :param file_path:
        :param kwargs:
        :return:
        """
        if self.voronoi is None:
            self.eval_voronoi()
        self.voronoi.plot(axs, show_fig, file_path, **kwargs)

    # def eval_metrics(self, metrics=()):
    #     if self.voronoi is None:
    #         self.voronoi = self.eval_voronoi()
    #     vc_vertices = self.voronoi.cells_vertices
    #     vc_neighbours = self.voronoi.cells_neighbours
    #     for a_metric in metrics:
    #         if a_metric == 'voronoi_cells_area':
    #             pass
    #         elif a_metric in ('first_nnd', 'second_nnd', 'third_nnd'):
    #             pass
    #         elif a_metric == 'first_nna':
    #             pass
    #     return


class UnitCells2D(dict):
    """
        A subclass of python dictionary to hold unit cells, whose with a typical key-value pair looks as follows

            'Unit-cell-i': UnitCell2D()
            # 'Unit-cell-i': {'bounds': bounds, 'shape-1': shape_1_data, 'shape-2': shape_2_data,...,}


    """

    def __init__(self, **kwargs):
        super(UnitCells2D, self).__init__(**kwargs)
        # self.update(parse_2d_unit_cell_file_data(data_fp, periodic))  # Updating the dictionary with parsed data

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
        Finds the images array

        :param file_path: str, if specified with .npy or .npz extension, it will save as numpy array on the disk.
        :param num_cores: int,
        :param p_bar: bool, enable/disable the progress bar
        :param kwargs: Key-word arguments that can be taken by ``UnitCell2D.get_image_array()``
        :return:
        """
        if num_cores is None:
            images_arrays = []
            loop = tqdm(self.values(), desc="Getting image arrays") if p_bar else self.values()
            for v in loop:
                images_arrays.append(v.get_image_array(**kwargs))
            images_array = asarray(images_arrays)
            if file_path is not None:
                if file_path.split(".")[-1] == "npy":
                    np_save(file_path, images_array)
                elif file_path.split(".")[-1] == "npz":
                    savez_compressed(file_path, images_array)
                else:
                    raise ValueError("Invalid file extension. It can be either *.npy or *.npz")
            return images_array
        else:
            # uc_instances = [i.get_image_array for i in self.values()]
            raise NotImplementedError("At present multiprocessing is not supported.")

    def plot_voronoi_diagrams(self, plots_dir, f_name=lambda a: f"{a}_vor.png", **kwargs):
        for (k, v) in self.items():
            v.plot_voronoi_diagram(file_path=path.join(plots_dir, f_name(k)), **kwargs)


class UnitCellsDataFile:
    """
        A class for handling the unit cell data files.


    """

    valid_file_extensions = ("npz", "h5", "json", "dat", "txt")

    def __init__(self, fp: str, periodic=False):
        self.fp = fp
        file_name_with_extn = path.basename(self.fp)
        self.f_name = ".".join(file_name_with_extn.split(".")[:-1])
        self.f_extension = file_name_with_extn.split(".")[-1]
        assert self.f_extension in self.valid_file_extensions , f"Invalid file extension {self.f_extension}."
        self.periodic = periodic

    def parse_npz(self):
        """
        Parses a single unit cell information from a NPZ kind of file

        :return: A UnitCell2D instance of the unit cell
        """
        unit_cell_data = dict(np_load(self.fp))
        _periodic = unit_cell_data.pop('periodic') if 'periodic' in unit_cell_data.keys() else self.periodic
        if 'bounds' in unit_cell_data.keys():
            _bounds = unit_cell_data.pop('bounds')
        else:
            raise KeyError("'bounds' key is expected in the npz file")
        # It is assumed that NPZ file contains information about a single unit cell
        return UnitCell2D(_bounds, unit_cell_data, _periodic)

    def parse_h5(self) -> UnitCells2D:
        """
        Parses multiple unit cells information from a h5 kind file

        :return: A UnitCells2D instance of the unit cell

        """
        h5_fp = File(self.fp, mode='r')
        unit_cells: UnitCells2D = UnitCells2D()
        for (k, v) in h5_fp.items():
            unit_cells[k] = UnitCell2D(
                [v.attrs[i] for i in ('xlb', 'ylb', 'xub', 'yub')],
                {ak: transpose(av) for (ak, av) in v.items()},
                v.attrs['periodic'] if 'periodic' in v.attrs.keys() else self.periodic
            )
        h5_fp.close()
        return unit_cells

    def parse_json(self):
        return

    def parse_dat(self):
        return

    def parse(self):
        if self.f_extension == "h5":
            return self.parse_h5()
        elif self.f_extension == "npz":
            return self.parse_npz()
        elif self.f_extension == "json":
            return self.parse_json()
        elif self.f_extension in ("dat", "txt"):
            return self.parse_dat()
