from pathlib import Path
from typing import Optional
from PIL import Image

from matplotlib.patches import Patch, Rectangle
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import io
import matplotlib.pyplot as plt


class PlotMixin:
    def get_patch(self, **kwargs) -> Patch:
        """
        Subclasses must implement this method to return the appropriate
        matplotlib patch. Any kwargs can be used to control styling.
        """
        raise NotImplementedError("Subclasses must implement get_patch()")

    @staticmethod
    def _get_image_array(_fig):
        io_buffer = io.BytesIO()
        plt.savefig(io_buffer, format="raw")
        io_buffer.seek(0)
        _image_array = np.reshape(
            np.frombuffer(io_buffer.getvalue(), dtype=np.uint8),
            shape=(int(_fig.bbox.bounds[3]), int(_fig.bbox.bounds[2]), -1),
        )
        io_buffer.close()
        return _image_array

    def plot(
        self, axs: Axes | None = None, **kwargs
    ) -> plt.Axes | tuple[plt.Figure, plt.Axes] | npt.NDArray[np.uint8] | None:
        """
        Adds the shape's patch to the provided axes.

        Parameters
        ----------
        axs : matplotlib.axes.Axes
            The axes to add the patch to.
        **kwargs :
            Additional keyword arguments passed to the respective patch.

        Returns
        -------
        Axes
            The modified axes.

        kwargs
        ------
        The following kwargs are supported:
        | kwarg | dtype | description  | default |
        |-------|-------|--------------|---------|
        | `shape_options` | dict | keyword arguments passed to the respective shape's `matplotlib.patches` patch constructor | `{"facecolor": "white", "edgecolor": "None"}` |
        | `bg_options` | dict | keyword arguments passed to the background rectangle | `{"facecolor": "black", "edgecolor": "None, "bounds": None}` |
        | `bounds` | tuple[float, float, float, float] | A 4-tuple specifying the bounds of the background rectangle. If not provided, it will be set to the bounds of the axes. | `None` |
        | `as_array` | bool | A numpy array is returned if True, otherwise a tuple of  | `False` |
        | `image_options` | dict | keyword arguments passed to the figure | `{"dpi": 100, "size": (256, 256), "mode": "L", "origin": "lower", "interpolation": None, "dtype": "uint8",}` |
        | `fig_options` | dict | keyword arguments passed to the figure | `{"axis": "off"}`|
        """
        # TODO: add support for bounding box of the patch
        shape_options = kwargs.get("shape_options", {})
        self_patch = self.get_patch(**shape_options)

        if isinstance(axs, Axes):
            axs.add_patch(self_patch)
            return axs


class ShapesPlotter:
    def __init__(
        self,
        shape_options=None,
        bg_options=None,
        fig_options=None,
        image_options=None,
    ):
        """

        Parameters
        ----------
        bg_options : dict
            Options for plotting the background. It may include the
            following keys:
            - `facecolor`: The facecolor of the background rectangle.
            - `edgecolor`: The edgecolor of the background rectangle.
            - `bounds`: A 4-tuple specifying the bounds of the background
            rectangle.
        object_options : dict
            Options for plotting the object.
        fig_options : dict
            Options for plotting the figure.
        image_options : dict
            Options for saving the image. It may include the following keys:
            - `dpi`: The resolution of the image in dots per inch.
            - `size`: The size of the image in pixels as 2-tuple
            (width, height).
            - `mode`: The mode of the image. Default is "L" (grayscale).
            - `as_array`: A boolean indicating whether to return the image
            as a numpy array.
            - `dtype`: The data type of the image. Default is "uint8".
        """
        image_options = image_options or {}
        fig_options = fig_options or {}
        bg_options = bg_options or {}

        self._shape_facecolor = shape_options.get("facecolor", "white")
        self._shape_edgecolor = shape_options.get("edgecolor", "None")

        dpi = image_options.get("dpi", 100)
        w_px, h_px = image_options.get("size", (256, 256))

        fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), frameon=False)
        axs = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(axs)
        plt.axis(fig_options.get("axis", "off"))

        self.fig = fig
        self.axs = axs

        self._add_background(bg_options)

    def _add_background(self, options):
        bg_facecolor = options.get("facecolor", "black")
        bg_edgecolor = options.get("edgecolor", "None")
        bg_bounds = options.get("bounds")
        if bg_bounds is not None:
            xlb, ylb, xub, yub = bg_bounds
            bb_patch = Rectangle(
                (xlb, ylb),
                xub - xlb,
                yub - ylb,
                edgecolor=bg_edgecolor,
                facecolor=bg_facecolor,
            )
            self.axs.add_patch(bb_patch)
            plt.xlim(xlb, xub)
            plt.ylim(ylb, yub)

    def add_shape(self, shape):
        patch = shape.get_patch(
            facecolor=self._shape_facecolor, edgecolor=self._shape_edgecolor
        )
        self.axs.add_patch(patch)

    def add_patch(self, patch):
        self.axs.add_patch(patch)

    def saveas(self, f_path: Path | str):
        if f_path:
            self.fig.savefig(f_path)

    def get_array(self):
        image_array = PlotMixin._get_image_array(self.fig)
        image_mode = self.image_options.get("mode", "L")
        image_dtype = np.dtype(self.image_options.get("dtype", "uint8"))
        if image_mode in ("L", "1"):
            img = Image.fromarray(image_array).convert(
                mode=image_mode, dither=Image.Dither.FLOYDSTEINBERG
            )
            image_array = np.array(img, dtype=image_dtype)
        return image_array

    def close(self):
        plt.close(self.fig)


def configure_axes(fig, ax, **kwargs):
    """
    Configure the figure with a predefined style.
    """
    # set fig size
    if "figsize" in kwargs:
        fig.set_size_inches(kwargs["figsize"])
    # set dpi
    if "dpi" in kwargs:
        fig.set_dpi(kwargs["dpi"])
    # off axes, always
    fig.patch.set_visible(False)
    # set face color, always
    if "facecolor" in kwargs:
        fig.patch.set_facecolor(kwargs["facecolor"])
    else:
        fig.patch.set_facecolor("white")
    # set aspect ratio, always to 'equal'
    if "aspect" in kwargs:
        ax.set_aspect(kwargs["aspect"])
    else:
        ax.set_aspect("equal")
    # set title, if provided
    if "title" in kwargs:
        fig.suptitle(
            kwargs["title"],
            fontsize=kwargs.get("title_fontsize", 16),
            fontweight="bold",
        )

    return fig


def _validate_dict(
    d: dict,
    keys: list,
    val_types: Optional[list] = None,
    ret_val: bool = False,
):
    """
    Validate that a dictionary contains specific keys.

    Parameters
    ----------
    d : dict
        The dictionary to validate.
    keys : list
        The list of keys that must be present in the dictionary.
    val_types : list, optional
        A list of types corresponding to each key in `keys`. If provided,
        the function will also check that the values associated with each key
        are of the specified type.
    ret_val : bool, optional
        If True, the function will return the values associated with the keys
        in the same order as the keys. Default is False.
    Raises
    ------
    ValueError
        If any of the specified keys are missing from the dictionary.
    """
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
    if not isinstance(keys, list):
        raise TypeError("Expected keys must be provided as a list.")

    missing_keys = [key for key in keys if key not in d]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    if val_types is None:
        val_types = [None] * len(keys)
    if len(keys) != len(val_types):
        raise ValueError("Length of keys and val_types must match.")
    for key, val_type in zip(keys, val_types):
        if val_type is None:
            continue
        if not isinstance(d[key], val_type):
            raise TypeError(
                f"Value for key '{key}' must be of type {val_type.__name__}, "
                f"but got {type(d[key]).__name__}."
            )
    if ret_val:
        return tuple(d[key] for key in keys)
