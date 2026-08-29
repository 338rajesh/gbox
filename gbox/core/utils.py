import logging
from collections.abc import Sequence
from numbers import Number

import numpy as np



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_logger(
    name: str = __name__, level: int = logging.INFO
) -> logging.Logger:
    """Returns a logger with the given name and level"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def _is_a_number(x) -> bool:
    """Checks if the input is a number as defined in the numbers module"""
    return isinstance(x, Number)


def _assert_a_sequence(seq, name: str = "input") -> bool:
    """Checks if the input is a sequence"""
    if not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be a sequence, but got {type(seq)}")
    return True


def _assert_a_sequence_of_numbers(
    seq, name: str = "input", length=None
) -> bool:
    """Checks if the input is a sequence of numbers
     
    """
    if not all(_is_a_number(x) for x in seq):
        raise TypeError(
            f"All elements of {name} must be numbers (int or float), "
            f"but got {[type(x) for x in seq]}",
        )
    if length is not None:
        if not isinstance(length, int):
            raise ValueError(
                f"While asserting {name} to be a sequence of numbers, "
                f"the length argument must be an int, but got {type(length)}"
            )
        if len(seq) != length:
            raise ValueError(
                f"{name} must have length {length}, but got {len(seq)}"
            )
    return True


# ------------


def rotate_point_2d(
    x: float,
    y: float,
    angle: float,
    pivot: Sequence[float] = (0.0, 0.0),
    *,
    degrees: bool = False,
) -> tuple[float, float]:
    """Rotates a point (x, y) by the given angle around the pivot point"""

    if degrees:
        angle = np.deg2rad(angle)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x -= pivot[0]
    y -= pivot[1]
    x_new = x * cos_a - y * sin_a + pivot[0]
    y_new = x * sin_a + y * cos_a + pivot[1]
    return x_new, y_new
