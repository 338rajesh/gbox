from contextlib import contextmanager
import pathlib
import shutil
import operator as op

from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

SEQUENCE = (list, tuple)
REAL_NUMBER = (int, float)

operators = {
    "add": op.add,
    "sub": op.sub,
    "mul": op.mul,
    "div": op.truediv,
    "floor_div": op.floordiv,
    "eq": op.eq,
    "lt": op.lt,
    "le": op.le,
    "gt": op.gt,
    "ge": op.ge,
    "ne": op.ne,
    "and": op.and_,
    "or": op.or_,
    "not": op.not_,
}


def get_output_dir(dir_path) -> pathlib.Path:
    dir_path = pathlib.Path(dir_path)
    if dir_path.exists():
        print(f"Cleaning existing output directory... {dir_path}")
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)
    return dir_path


@contextmanager
def gb_plotter(file_path=None, **kwargs):

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


def rotation_matrix_2d(angle: float, unit="rad") -> np.ndarray:
    """Returns the rotational matrix for a given angle."""
    if unit == "deg":
        angle *= np.pi / 180
    else:
        assert unit == "rad", "Unit must be 'deg' or 'rad'"

    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def validated_num_cores(n) -> int:
    assert isinstance(n, int), "Number of cores must be an integer"
    assert n > 0, "Number of cores must be greater than 0"
    max_cpus = cpu_count()
    return n if n <= max_cpus else max_cpus


class LineStyles:

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
