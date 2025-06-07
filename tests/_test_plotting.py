from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from gbox.ellipse import Ellipse
from utils import get_output_dir

OUTPUT_DIR = get_output_dir(
    Path(__file__).resolve().parent.joinpath("test_output")
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def plot_ellipses(sample_ellipse: Ellipse, **kwrags):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()

    sample_ellipse.plot(axs[0], b_box=False, uoc=False, **kwrags)
    axs[0].set_title("Ellipse-no_bbox-no_uoc")

    sample_ellipse.plot(axs[1], b_box=True, uoc=False, **kwrags)
    axs[1].set_title("Ellipse-with_bbox-no_uoc")

    sample_ellipse.plot(axs[2], b_box=True, uoc=True, **kwrags)
    axs[2].set_title("Ellipse-with_bbox-with_uoc")

    sample_ellipse.plot(axs[3], b_box=False, uoc=True, **kwrags)
    axs[3].set_title("Ellipse-no_bbox-with_uoc")

    for i in range(4):
        axs[i].set_aspect("equal")
        axs[i].grid(alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR.joinpath("test_plot_ellipse.png"))
    assert OUTPUT_DIR.joinpath("test_plot_ellipse.png").is_file()
    plt.close(fig)


if __name__ == "__main__":
    sample_ellipse = Ellipse(8.0, 2.0, (-3.0, 2.5), np.pi / 4)
    plot_ellipses(
        sample_ellipse,
        uoc_dh=0.01,
        plot_kwargs={"color": "r", "linewidth": 2},
        bbox_plot_kwargs={"color": "b", "linewidth": 1},
        uoc_plot_kwargs={"color": "g", "linewidth": 0.5},
    )
