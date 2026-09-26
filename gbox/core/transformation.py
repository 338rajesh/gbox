from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .utils import Angle, TransformationOrder, get_logger

logger = get_logger(__name__)


# ============================================================
# 2D transformations
# ============================================================


def rotation_matrix_2d(angle: Angle) -> npt.NDArray[np.float64]:
    """Return a 2D counter-clockwise rotation matrix."""
    return np.array(
        [
            [angle.cos, -angle.sin],
            [angle.sin, angle.cos],
        ],
        dtype=np.float64,
    )


def translation_matrix_2d(
    dx: float = 0.0,
    dy: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return a 2D homogeneous translation matrix."""
    return np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transformation_matrix_2d(
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    angle: Angle = None,
    pivot: Sequence[float] = (0.0, 0.0),
    order: TransformationOrder = TransformationOrder.ROTATE_THEN_TRANSLATE,
) -> npt.NDArray[np.float64]:
    """Return a 2D homogeneous transformation matrix.

    The transformation consists of rotation around ``pivot`` and
    translation by ``(dx, dy)``.

    ``ROTATE_THEN_TRANSLATE`` applies:

        p' = T @ R @ p

    ``TRANSLATE_THEN_ROTATE`` applies:

        p' = R @ T @ p
    """
    angle = angle or Angle(0.0, units="radians")
    if len(pivot) != 2:
        raise ValueError("2D pivot must contain exactly 2 coordinates.")

    if not isinstance(order, TransformationOrder):
        raise TypeError(
            f"order must be a TransformationOrder, got {type(order).__name__}."
        )

    rotation_2x2 = rotation_matrix_2d(angle)

    # p_new = pivot + R @ (p - pivot)  # due to rotation around pivot
    #       = R @ p + (pivot - R @ pivot)

    pivot = np.asarray(pivot, dtype=np.float64)
    pivot_offset = (pivot - rotation_2x2 @ pivot).reshape(
        2, 1
    )  # column vector
    rotation = np.block(
        [
            [rotation_2x2, pivot_offset],
            [np.zeros((1, 2)), np.ones((1, 1))],
        ]
    )

    translation = translation_matrix_2d(dx, dy)

    if order is TransformationOrder.ROTATE_THEN_TRANSLATE:
        return translation @ rotation

    # order is TransformationOrder.TRANSLATE_THEN_ROTATE
    return rotation @ translation


def transform_point_2d(
    x: float,
    y: float,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    angle: Angle = None,
    pivot: Sequence[float] = (0.0, 0.0),
    order: TransformationOrder = TransformationOrder.ROTATE_THEN_TRANSLATE,
) -> tuple[float, float]:
    """Transform a 2D point by rotation and translation.

    Parameters
    ----------
    x, y:
        Coordinates of the point.

    dx, dy:
        Translation to apply.

    angle:
        Counter-clockwise rotation angle.

    pivot:
        Point around which the rotation is performed.

    order:
        Whether rotation or translation is applied first.

    Returns
    -------
    tuple[float, float]
        The transformed ``(x, y)`` coordinates.
    """
    angle = angle or Angle.rad(0.0)
    matrix = transformation_matrix_2d(
        dx=dx, dy=dy, angle=angle, pivot=pivot, order=order
    )  # 3x3 matrix

    point = np.array([x, y, 1.0], dtype=np.float64)  # homogeneous coordinates
    transformed_point = matrix @ point  # matrix multiplication
    return float(transformed_point[0]), float(transformed_point[1])
