from contextlib import contextmanager
import pathlib
import shutil

import matplotlib.pyplot as plt

__all__ = ["get_output_dir", "gb_plotter", "LineStyles"]


def get_output_dir(dir_path: str | pathlib.Path) -> pathlib.Path:
    """Returns a pathlib.Path object for the output directory"""
    dir_path = pathlib.Path(dir_path)
    if dir_path.exists():
        print(f"Cleaning existing output directory... {dir_path}")
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)
    return dir_path


@contextmanager
def gb_plotter(file_path=None, **kwargs):
    """Context manager for plotting"""
    fig, axs = plt.subplots(1, 1)
    exception_raised = False
    try:
        yield fig, axs
    except Exception as e:
        exception_raised = True
        raise e
    finally:
        # Add legend if labels are present
        handles, labels = axs.get_legend_handles_labels()
        if labels:  # Check if any labels exist
            axs.legend(handles, labels)

        if not exception_raised:
            fig.tight_layout()
            fig.savefig(file_path, **kwargs)
            assert file_path.is_file(), f"File {file_path} is not saved!"
        plt.close(fig)


class LineStyles:
    """Helper class for line styles"""

    THEME_1 = {
        "color": "blue",
        "linewidth": 1.2,
        "linestyle": "solid",
        "marker": "None",
    }

    THEME_2 = {
        "color": "red",
        "linewidth": 2.1,
        "linestyle": "dashed",
        "marker": "None",
    }

    THEME_3 = {
        "color": "green",
        "linewidth": 1.2,
        "linestyle": "dotted",
        "marker": "None",
    }

    THEME_4 = {
        "color": "black",
        "linewidth": 1.6,
        "linestyle": "dotted",
        "marker": "None",
    }
