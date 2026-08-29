from collections.abc import Sequence
from typing import Any, Iterator, Union

import numpy as np
import numpy.typing as npt

from .utils import (
    _assert_a_sequence,
    _assert_a_sequence_of_numbers,
)


class PointArrayND:
    """A collection of points in N-dimensional space. Coordinates are stored
    internally as a contiguous NumPy array with shape ``(n_points, n_dims)``
    and dtype ``float64``.
    """

    __slots__ = ("_coordinates",)

    def __init__(
        self, points: Sequence[Sequence[float]] | npt.NDArray[np.float64]
    ) -> None:
        """Constructs a PointArray from a NumpyArray of points

        Parameters
        ----------
        points : Sequence[Sequence[float]] | npt.NDArray[np.float64]
            A sequence of sequences of coordinates or a NumpyArray of shape
            (n_points, n_dims) and dtype float64. If a sequence of sequences
            is provided, the outer sequence should have length n_points and
            each inner sequence should have length n_dims. If a NumpyArray is
            provided, it should have shape (n_points, n_dims) of float64 dtype.
        """
        self._coordinates = self._validate_points(points)

    # ============================
    # Private methods
    # ============================
    @staticmethod
    def _validate_points(
        points: Any, dtype: np.dtype = np.dtype(np.float64)
    ) -> npt.NDArray[np.float64]:
        """Validates the points in the array and returns a NumpyArray of points
        with the expected dtype"""
        try:
            points = np.asarray(points, dtype=dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "PointArray construction requires a sequence "
                "of points or a NumpyArray"
            ) from e

        if points.ndim != 2:
            raise ValueError("Points must be a 2D array (n_points x n_dims)")

        if points.shape[0] == 0:
            raise ValueError("PointArray must have at least one point")

        return np.ascontiguousarray(points)

    @classmethod
    def from_dim_sequences(
        cls,
        sequences: Sequence[Sequence[float]],
        names: Sequence[str] | None = None,
    ) -> "PointArrayND":
        """
        Constructs a PointArray from a sequence of sequences of coordinates
        """
        if isinstance(sequences, np.ndarray):
            return cls(sequences)

        if names is not None and len(names) != len(sequences):
            raise ValueError(
                f"Expected {len(sequences)} names, got {len(names)}"
            )

        _assert_a_sequence(sequences, name="sequence of dimensions")

        for idx, sequence in enumerate(sequences):
            _assert_a_sequence_of_numbers(
                sequence,
                name=f"Dimension {idx} sequence" if not names else names[idx],
            )

        seq_lengths = {len(sequence) for sequence in sequences}
        if len(seq_lengths) != 1:
            raise ValueError(
                "All dimension sequences must have the same length, "
                f"but got lengths: {seq_lengths}"
            )

        return cls(np.column_stack(sequences))

    @classmethod
    def from_named_dims(cls, **data: Sequence[float]) -> "PointArrayND":
        """
        Constructs a PointArray from a dictionary of sequences of coordinates
        """
        if not data:
            raise ValueError("Input dictionary is empty")

        return cls.from_dim_sequences(
            list(data.values()), names=list(data.keys())
        )

    # ============================
    #       MAGIC METHODS
    # ============================

    def __len__(self) -> int:
        """Returns the number of points in the PointArray"""
        return self._coordinates.shape[0]

    def __getitem__(self, idx: int | slice | tuple) -> np.ndarray:
        """Returns the point(s) at the given index or slice"""
        return self._coordinates[idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._coordinates)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"points={self._coordinates!r}, "
            f"dim={self.dim}, "
            f"dtype={self.dtype})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} with {len(self)} points in {self.dim}D"
        )

    # ============================
    #       POINT PROPERTIES
    # ============================
    @property
    def dim(self) -> int:
        return self._coordinates.shape[1]

    @property
    def dtype(self) -> np.dtype:
        return self._coordinates.dtype

    @property
    def coordinates(self) -> npt.NDArray[np.float64]:
        return self._coordinates

    # ============================
    #       GEOMETRY OPERATIONS
    # ============================
    def bounding_box(self) -> tuple[list[float], list[float]]:
        """Returns the bounding box of the current PointArray"""
        return (
            np.min(self._coordinates, axis=0).tolist(),
            np.max(self._coordinates, axis=0).tolist(),
        )

    def transform(
        self, matrix: np.ndarray, in_place: bool = False
    ) -> Union["PointArrayND", None]:
        """Transform using a transformation matrix

        Here, the transformation matrix should be of shape (dim+1, dim+1) for homogeneous coordinates.
        The last row of the matrix should be [0, 0, ..., 1]
        """
        if matrix.shape != (self.dim + 1, self.dim + 1):
            raise ValueError(
                f"Transformation matrix must be {self.dim + 1}x{self.dim + 1}",
            )
        if matrix.dtype != self.dtype:
            matrix = matrix.astype(self.dtype)

        points = np.column_stack(
            (self._coordinates, np.ones(len(self), dtype=self.dtype))
        )
        transformed = points @ matrix.T
        transformed = np.ascontiguousarray(
            transformed[:, : self.dim], dtype=self.dtype
        )

        if in_place:
            self._coordinates = transformed
            return None

        return self.__class__(transformed)

    # ============================
    #       UTILITY METHODS
    # ============================

    def copy(self) -> "PointArrayND":
        """Returns a copy of the current PointArray"""
        return self.__class__(self._coordinates.copy())

    def to_list(self) -> list[list[float]]:
        """Returns a list of lists of coordinates from the current PointArray"""
        return self._coordinates.tolist()


# endregion PointArrayND

# region PointArray2D


class PointArray2D(PointArrayND):
    """PointArray2D, a subclass of PointArray, with two dimensions.

    Attributes
    ----------
    coordinates : np.ndarray
        Array of point coordinates
    dim : int
        Dimension of the points
    dtype : np.dtype
        Data type of the points
    x : NDArray
        Array of x coordinates
    y : NDArray
        Array of y coordinates

    """

    __slots__ = ()

    def __init__(
        self, points: npt.NDArray[np.float64] | Sequence[Sequence[float]]
    ) -> None:
        super().__init__(points)

        if self.dim != 2:
            raise ValueError(f"PointArray2D must have 2 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self._coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._coordinates[:, 1]

    def transform(
        self,
        angle: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        pivot: tuple[float, float] = (0.0, 0.0),
        in_place: bool = False,
        order: str = "rotation_then_translation",
    ) -> Union["PointArray2D", None]:
        """Transformation of the points cluster by rotation and translation,
        either in-place or returning a new PointArray2D

        Parameters
        ----------
        angle : float
            Angle of rotation in radians, default: 0.0
        dx : float
            Translation along x axis, default: 0.0
        dy : float
            Translation along y axis, default: 0.0
        pivot : tuple[float, float]
            Pivot point for rotation, default: (0.0, 0.0)
        in_place : bool
            If True, the transformation is applied in-place and the method returns None.
            If False, a new PointArray2D is returned with the transformed points.
        order : str
            Order of operations, either "rotation_then_translation" or "translation_then_rotation", default: "rotation_then_translation"

        Returns
        -------
        PointArray2D

        """
        if order not in (
            "rotation_then_translation",
            "translation_then_rotation",
        ):
            raise ValueError(
                f"Invalid order: {order}, should be "
                "'rotation_then_translation' or 'translation_then_rotation'"
            )

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        temp_x = self.x - pivot[0]
        temp_y = self.y - pivot[1]

        if order == "rotation_then_translation":
            x = temp_x * cos_a - temp_y * sin_a + dx + pivot[0]
            y = temp_x * sin_a + temp_y * cos_a + dy + pivot[1]
        else:  # translation_then_rotation
            temp_x += dx
            temp_y += dy
            x = temp_x * cos_a - temp_y * sin_a + pivot[0]
            y = temp_x * sin_a + temp_y * cos_a + pivot[1]

        if in_place:
            self._coordinates[:, 0] = x
            self._coordinates[:, 1] = y
            return None

        return PointArray2D.from_named_dims(x=x, y=y)


# endregion PointArray2D
# region PointArray3D


class PointArray3D(PointArrayND):
    """PointArray3D, a subclass of PointArray, with three dimensions"""

    __slots__ = ()

    def __init__(
        self, points: npt.NDArray[np.float64] | Sequence[Sequence[float]]
    ) -> None:
        super().__init__(points)

        if self.dim != 3:
            raise ValueError(f"PointArray3D must have 3 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self._coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._coordinates[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self._coordinates[:, 2]

    def transform(
        self,
        angle: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
        in_place: bool = False,
        order: str = "RT",
    ) -> Union["PointArray3D", None]:
        raise NotImplementedError
