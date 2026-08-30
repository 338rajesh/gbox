from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt


AngleUnits: TypeAlias = Literal["radians", "degrees"]


# ============================================================
# Utilities
# ============================================================


def _angle_to_radians(angle: float, angle_units: AngleUnits) -> float:
    if angle_units == "radians":
        return angle

    if angle_units == "degrees":
        return float(np.deg2rad(angle))

    raise ValueError(
        f"Invalid angle_units: {angle_units!r}. "
        "Expected 'radians' or 'degrees'."
    )


# ============================================================
# 2D transformations
# ============================================================


def rotation_matrix_2d(
    angle: float,
    *,
    angle_units: AngleUnits = "radians",
) -> npt.NDArray[np.float64]:
    """Return a 2D counter-clockwise rotation matrix."""
    angle = _angle_to_radians(angle, angle_units)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    return np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
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
    angle: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    angle_units: AngleUnits = "radians",
    pivot: Sequence[float] = (0.0, 0.0),
    order: Literal[
        "rotate_then_translate",
        "translate_then_rotate",
    ] = "rotate_then_translate",
) -> npt.NDArray[np.float64]:
    """Return a 2D homogeneous transformation matrix.

    The transformation consists of rotation around ``pivot`` and
    translation by ``(dx, dy)``.
    """
    if len(pivot) != 2:
        raise ValueError("2D pivot must contain exactly 2 coordinates.")

    if order not in (
        "rotate_then_translate",
        "translate_then_rotate",
    ):
        raise ValueError(
            f"Invalid order: {order!r}. "
            "Expected 'rotate_then_translate' or "
            "'translate_then_rotate'."
        )

    angle = _angle_to_radians(angle, angle_units)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    px, py = pivot

    # Rotation around pivot:
    #
    # T(pivot) @ R @ T(-pivot)
    #
    rotation = np.array(
        [
            [cos_a, -sin_a, px - px * cos_a + py * sin_a],
            [sin_a, cos_a, py - px * sin_a - py * cos_a],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    translation = translation_matrix_2d(dx, dy)

    if order == "rotate_then_translate":
        return translation @ rotation
    else:  # translate_then_rotate
        return rotation @ translation


def rotate_point_2d(
    x: float,
    y: float,
    angle: float,
    *,
    angle_units: AngleUnits = "radians",
    pivot: Sequence[float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Rotate one 2D point around a pivot."""
    if len(pivot) != 2:
        raise ValueError("2D pivot must contain exactly 2 coordinates.")

    angle = _angle_to_radians(angle, angle_units)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    px, py = pivot

    temp_x = x - px
    temp_y = y - py

    new_x = temp_x * cos_a - temp_y * sin_a + px
    new_y = temp_x * sin_a + temp_y * cos_a + py

    return float(new_x), float(new_y)


# ============================================================
# 3D rotations
# ============================================================


def rotation_matrix_x(
    angle: float,
    *,
    angle_units: AngleUnits = "radians",
) -> npt.NDArray[np.float64]:
    """Return a 3D rotation matrix around the x-axis."""
    angle = _angle_to_radians(angle, angle_units)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(
    angle: float,
    *,
    angle_units: AngleUnits = "radians",
) -> npt.NDArray[np.float64]:
    """Return a 3D rotation matrix around the y-axis."""
    angle = _angle_to_radians(angle, angle_units)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(
    angle: float,
    *,
    angle_units: AngleUnits = "radians",
) -> npt.NDArray[np.float64]:
    """Return a 3D rotation matrix around the z-axis."""
    angle = _angle_to_radians(angle, angle_units)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# ============================================================
# 3D translations
# ============================================================


def translation_matrix_3d(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return a 3D homogeneous translation matrix."""
    return np.array(
        [
            [1.0, 0.0, 0.0, dx],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# ============================================================
# 3D combined transformation
# ============================================================


def transformation_matrix_3d(
    *,
    angles: Sequence[float] = (0.0, 0.0, 0.0),
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    angle_units: AngleUnits = "radians",
    pivot: Sequence[float] = (0.0, 0.0, 0.0),
    order: Literal[
        "rotate_then_translate",
        "translate_then_rotate",
    ] = "rotate_then_translate",
    rotation_order: Literal[
        "xyz",
        "xzy",
        "yxz",
        "yzx",
        "zxy",
        "zyx",
    ] = "xyz",
) -> npt.NDArray[np.float64]:
    """Return a 3D homogeneous transformation matrix.

    ``angles`` contains rotations about x, y and z respectively.
    """
    if len(angles) != 3:
        raise ValueError("angles must contain exactly 3 values.")

    if len(pivot) != 3:
        raise ValueError("3D pivot must contain exactly 3 coordinates.")

    if order not in (
        "rotate_then_translate",
        "translate_then_rotate",
    ):
        raise ValueError(
            f"Invalid order: {order!r}. "
            "Expected 'rotate_then_translate' or "
            "'translate_then_rotate'."
        )

    matrices = {
        "x": rotation_matrix_x(
            angles[0],
            angle_units=angle_units,
        ),
        "y": rotation_matrix_y(
            angles[1],
            angle_units=angle_units,
        ),
        "z": rotation_matrix_z(
            angles[2],
            angle_units=angle_units,
        ),
    }

    rotation_3d = np.eye(4, dtype=np.float64)

    for axis in rotation_order:
        rotation_3d = rotation_3d @ np.block(
            [
                [matrices[axis], np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )

    px, py, pz = pivot

    to_pivot = translation_matrix_3d(-px, -py, -pz)
    from_pivot = translation_matrix_3d(px, py, pz)

    rotation = from_pivot @ rotation_3d @ to_pivot
    translation = translation_matrix_3d(dx, dy, dz)

    if order == "rotate_then_translate":
        return translation @ rotation

    return rotation @ translation


# ============================================================
# Generic transformation
# ============================================================


def transform_points(
    points: npt.NDArray[np.float64],
    matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply a homogeneous transformation matrix to N-dimensional points."""
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array.")

    n_dims = points.shape[1]

    if matrix.shape != (n_dims + 1, n_dims + 1):
        raise ValueError(
            f"Expected a {(n_dims + 1)}x{(n_dims + 1)} transformation "
            f"matrix for {n_dims}D points."
        )

    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=points.dtype))
    )

    return np.ascontiguousarray((homogeneous @ matrix.T)[:, :n_dims])
