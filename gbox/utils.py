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
