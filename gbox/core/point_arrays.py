
class PointArrayND:
    """PointArrayND, a base class for representing a collection of points
    in N-dimensional space
    """

    __slots__ = ("_cycle", "coor")

    def __init__(self, points: np.ndarray):
        """Constructs a PointArray from a NumpyArray of points"""
        self._validate_points(points)
        self.coor = np.ascontiguousarray(points, dtype=DEFAULT_FLOAT)
        self._cycle = False  # For open curves

    # ============================
    # Private methods
    # ============================
    @staticmethod
    def _validate_points(points: np.ndarray) -> bool:
        """Validates the points in the array"""
        if not isinstance(points, np.ndarray):
            raise TypeError("PointArray construction requires a NumpyArray")
        if points.ndim != 2:
            raise ValueError("Points must be 2D array (n_points x n_dims)")
        if points.size == 0:
            raise ValueError("PointArray must have at least one point")
        return True

    @classmethod
    def from_points(
        cls,
        points: Sequence[PointND] | Sequence[Sequence[float]],
    ) -> "PointArrayND":
        """Constructs a PointArray from a sequence of Point objects or
        sequences of sequences of coordinates
        """
        if not points:
            raise ValueError("PointArray must have at least one point")
        if not isinstance(points, Sequence):
            raise TypeError(
                "Points must be a sequence of Point objects or "
                "sequences of sequence of coordinates",
            )
        _dim_ = len(points[0])
        if any(len(p) != _dim_ for p in points):
            raise ValueError("All points must have same dimension")
        return cls(np.array(points, dtype=DEFAULT_FLOAT))

    @classmethod
    def from_dims(cls, dims: Sequence[Sequence[FloatType]]):
        """
        Constructs a PointArray from a sequence of sequences of coordinates
        """
        if not dims:
            raise ValueError(
                "PointArray must have coordinates along at least one dimension"
            )
        if not all(len(d) == len(dims[0]) for d in dims):
            raise ValueError("All dimensions must have same length")
        return cls(np.array(dims, dtype=DEFAULT_FLOAT).T)

    # ============================
    #       MAGIC METHODS
    # ============================
    def __len__(self):
        """Returns the number of points in the PointArray"""
        return len(self.coor)

    def __getitem__(self, idx: int | slice | tuple) -> np.ndarray:
        """Returns the point(s) at the given index or slice"""
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError("Incorrect number of indices for PointArray")
            return self.coor[idx[0], idx[1]]
        return self.coor[idx]

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter(self.coor)

    def __array__(self, dtype=None, copy=True) -> np.ndarray:
        """Returns the coordinates of the point array as a numpy array"""
        arr = np.array(self.coor, dtype=dtype, copy=copy)
        return arr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({len(self)} points; dim={self.dim};"
            f" dtype={self.dtype})"
        )

    # ============================
    #       POINT PROPERTIES
    # ============================
    @property
    def dim(self):
        return self.coor.shape[1]

    @property
    def dtype(self):
        return self.coor.dtype

    @property
    def coordinates(self) -> np.ndarray:
        """Returns the coordinates of the point array"""
        return self.coor.copy()

    @property
    def cycle(self) -> bool:
        """Returns True if the points are cyclic"""
        return self._cycle

    @cycle.setter
    def cycle(self, val: bool):
        """Sets the cyclic property of the points"""
        if not isinstance(val, bool):
            raise TypeError("cycle take a boolean value")
        self._cycle = val

    # ============================
    #       GEOMETRY OPERATIONS
    # ============================
    def bounding_box(self) -> "BoundingBox":
        """Returns the bounding box of the current PointArray"""
        return BoundingBox(
            np.min(self.coor, axis=0),
            np.max(self.coor, axis=0),
        )

    def transform(
        self,
        matrix: np.ndarray,
        in_place: bool = False,
    ) -> Union["PointArrayND", None]:
        """Transform using a transformation matrix"""
        if matrix.shape != (self.dim + 1, self.dim + 1):
            raise ValueError(
                f"Transformation matrix must be {self.dim + 1}x{self.dim + 1}",
            )
        if matrix.dtype != self.dtype:
            matrix = matrix.astype(self.dtype)
        points = np.column_stack([self.coor, np.ones(len(self))])
        if in_place:
            transformed = points @ matrix.T
            if not transformed.dtype == self.dtype:
                transformed = transformed.astype(self.dtype)
            self.coor[:, :] = transformed[:, : self.dim]
            return None
        transformed = points @ matrix.T
        transformed = transformed[:, : self.dim].astype(self.dtype)
        return self.__class__(transformed)

    def reflection(
        self,
        p1: Union[list[float], "PointND"],
        p2: Union[list[float], "PointND"],
    ):
        """Reflects the current points about a line connecting p1 and p2"""
        raise NotImplementedError("Point Array reflection is not implemented")

    def reverse(self, in_place: bool = False) -> Union["PointArrayND", None]:
        """Reverses the order of the points"""
        rev_coor = np.flip(self.coor, axis=0)
        if in_place:
            self.coor = rev_coor
            return None
        return self.__class__(rev_coor)

    # ============================
    #       UTILITY METHODS
    # ============================
    def copy(self):
        """Returns a copy of the current PointArray"""
        return self.__class__(self.coordinates)

    def to_points_list(self) -> list[PointND]:
        """Returns a list of Point objects from the current PointArray"""
        return [PointND(*row) for row in self.coor]


# endregion PointArrayND
# region PointArray1D


class PointArray1D(PointArrayND):
    """PointArray1D, a subclass of PointArray, with one dimension"""

    __slots__ = ()

    def __init__(self, points: np.ndarray):
        """Constructs a PointArray1D from a NumpyArray"""
        if points.ndim == 1:
            points = np.atleast_2d(points).T
        super().__init__(points)
        if self.dim != 1:
            raise ValueError("PointArray1D must have one dimension")

    @property
    def x(self) -> np.ndarray:
        return self.coor[:, 0]

    def transform(
        self,
        dx: float = 0.0,
        in_place: bool = False,
    ) -> Union["PointArray1D", None]:
        """Transformation of the points cluster by rotation and translation"""
        if in_place:
            if dx != 0.0:
                self.coor[:] = self.coor[:] + dx
            return None
        return self.__class__(self.coor + dx)


# endregion PointArray1D
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

    def __init__(self, points: np.ndarray) -> None:
        """Construct a PointArray2D from a NumpyArray."""
        super().__init__(points)
        if self.dim != 2:
            raise ValueError("PointArray2D must have 2 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]

    def transform(
        self,
        angle: FloatType = 0.0,
        dx: FloatType = 0.0,
        dy: FloatType = 0.0,
        pivot: Point2D | Tuple[FloatType, FloatType] = (0.0, 0.0),
        in_place: bool = False,
        order: str = "RT",
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

        Returns
        -------
        PointArray2D

        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        temp_x = self.x - pivot[0]
        temp_y = self.y - pivot[1]
        if order == "RT":
            x = temp_x * cos_a - temp_y * sin_a + dx + pivot[0]
            y = temp_x * sin_a + temp_y * cos_a + dy + pivot[1]
        elif order == "TR":
            temp_x += dx
            temp_y += dy
            x = temp_x * cos_a - temp_y * sin_a + pivot[0]
            y = temp_x * sin_a + temp_y * cos_a + pivot[1]
        else:
            raise ValueError(f"Invalid order: {order}, should be 'RT' or 'TR'")

        if in_place:
            self.coor[:, 0] = x
            self.coor[:, 1] = y
            return None
        return PointArray2D(np.column_stack([x, y]))

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """Returns tiled copy of the points about the current position"""
        raise NotImplementedError("make_periodic_tiles is not implemented")

    def sort(self) -> "PointArray2D":
        raise NotImplementedError("sort is not implemented")

    def plot(
        self,
        axs,
        points_plt_opt: dict | None = None,
        b_box: bool = False,
        box_plt_opt: dict | None = None,
    ):
        """Plots the points"""

        points_plt_options = {
            **_DEFAULT_POINT_PLOT_OPTIONS,
            **(points_plt_opt or {}),
        }
        axs.plot(
            np.append(self.x, self.x[0]) if self.cycle else self.x,
            np.append(self.y, self.y[0]) if self.cycle else self.y,
            **points_plt_options,
        )

        if b_box:
            bbox_plt_options = {
                **_DEFAULT_LINE_PLOT_OPTIONS,
                **(box_plt_opt or {}),
            }
            self.bounding_box.plot(axs, **bbox_plt_options)

        return axs


# endregion PointArray2D
# region PointArray3D


class PointArray3D(PointArrayND):
    """PointArray3D, a subclass of PointArray, with three dimensions"""

    __slots__ = ()

    def __init__(self, points):
        super().__init__(points)
        if self.dim != 3:
            raise ValueError("PointArray3D must have 3 dims, got {self.dim}D")

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.coordinates[:, 2]

    def make_periodic_tiles(self, bounds: list | None = None, order: int = 1):
        """ """
        raise NotImplementedError("make_periodic_tiles is not implemented")

