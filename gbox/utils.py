from typing import Optional

from matplotlib.patches import Patch
from matplotlib.axes import Axes


class PlotMixin:
    def get_patch(self, **kwargs) -> Patch:
        """
        Subclasses must implement this method to return the appropriate
        matplotlib patch. Any kwargs can be used to control styling.
        """
        raise NotImplementedError("Subclasses must implement get_patch()")

    def plot(self, axs: Axes, **kwargs) -> Axes:
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
        """
        # TODO: add support for bounding box of the patch
        patch = self.get_patch(**kwargs)
        axs.add_patch(patch)
        return axs


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
