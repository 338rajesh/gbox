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

        bg_options = kwargs.get("bg_options", {})
        image_options = kwargs.get("image_options", {})
        fig_options = kwargs.get("fig_options", {})

        dpi = image_options.get("dpi", 100)
        w_px, h_px = image_options.get("size", (256, 256))
        fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), frameon=False)
        axs = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(axs)
        plt.axis(fig_options.get("axis", "off"))

        bg_bounds = kwargs.get("bounds", None)
        bg_facecolor = bg_options.get("facecolor", "black")
        bg_edgecolor = bg_options.get("edgecolor", "None")
        if bg_bounds is not None:
            xlb, ylb, xub, yub = bg_bounds
            bb_patch = Rectangle(
                (xlb, ylb),
                xub - xlb,
                yub - ylb,
                edgecolor=bg_edgecolor,
                facecolor=bg_facecolor,
            )
            axs.add_patch(bb_patch)
            plt.xlim(xlb, xub)
            plt.ylim(ylb, yub)

        axs.add_patch(self_patch)

        as_array = kwargs.get("as_array", False)

        if not as_array:
            return fig, axs
        else:
            image_array = self._get_image_array(fig)
            image_mode = image_options.get("mode", "L")
            image_dtype = np.dtype(image_options.get("dtype", "uint8"))
            if image_mode in ("L", "1"):
                img = Image.fromarray(image_array).convert(
                    mode=image_mode, dither=Image.Dither.FLOYDSTEINBERG
                )
                image_array = np.array(img, dtype=image_dtype)

            plt.close(fig)
            return image_array


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
