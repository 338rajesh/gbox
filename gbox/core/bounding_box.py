
class BoundingBox(PlotMixin):  # TODO: Review this class
    """A class for performing n-dimensional bounding box operations

    It is expexcted that the number of elements in the lower and
    upper bound are same. Also, the lower bound must be less
    than the upper bound.

    Parameters
    ----------
    lower_bound : list or tuple
        Lower bounds of the box
    upper_bound : list or tuple
        Upper bounds of the box

    Raises
    ------
    ValueError
        If lower bound and upper bound have different length
        If lower bounds are greater than upper bounds

    Examples
    --------
    >>> import numpy as np
    >>> lower_bound = [0, 0]
    >>> upper_bound = [10, 10]
    >>> bounding_box = BoundingBox(lower_bound, upper_bound)
    >>> bounding_box.lb
    array([ 0.,  0.])
    >>> bounding_box.ub
    array([10., 10.])

    """

    __slots__ = ("_vertices", "_volume", "p_max", "p_min")

    def __init__(
        self,
        lower_bound: np.ndarray | Sequence[float],
        upper_bound: np.ndarray | Sequence[float],
    ):
        self.p_min = PointND(*lower_bound)
        self.p_max = PointND(*upper_bound)
        self._validate_bounds()

    def _validate_bounds(self) -> bool:
        if self.p_min.dim != self.p_max.dim:
            raise ValueError("lower and upper bounds must have same dimension")

        if np.any(self.p_min.coor > self.p_max.coor):
            raise ValueError(
                "lower bounds must be less than upper bounds"
                f"lower bounds: {self.p_min}, upper bounds: {self.p_max}",
            )
        return True

    def __eq__(self, bb_2: "BoundingBox") -> bool:
        """Checks if two bounding boxes are equal

        Parameters
        ----------
        bb_2 : BoundingBox
            Bounding box to be compared

        Returns
        -------
        bool
            True if bounding boxes are equal

        Raises
        ------
        TypeError
            If bb_2 is not of type BoundingBox

        """
        if not isinstance(bb_2, BoundingBox):
            raise TypeError("bb_2 must be of type BoundingBox")

        lb_equality = np.array_equal(self.p_min.coor, bb_2.p_min.coor)
        ub_equality = np.array_equal(self.p_max.coor, bb_2.p_max.coor)
        return lb_equality and ub_equality

    def __repr__(self) -> str:
        return (
            f"BoundingBox:\n\tlower_bound: {self.p_min.coor}"
            f"\n\tupper_bound: {self.p_max.coor}"
        )

    def has_point(self, p: PointND | Sequence[float]) -> bool:
        """Checks if the point 'p' is within the bounding box

        Returns
        -------
        bool
            True if 'p' is within the bounding box

        """
        p = PointND._from_(p)
        if not (self.p_min.dim == self.p_max.dim == p.dim):
            raise ValueError(
                f"point 'p' dimension {p.dim} does not match with "
                f"bounding box dimension {self.p_min.dim}",
            )
        return bool(
            np.all((self.p_min.coor <= p.coor) & (p.coor <= self.p_max.coor)),
        )

    def overlaps(self, bb: "BoundingBox", include_bounds=False) -> bool:
        """Returns True, if two bounding boxes overlap

        Returns
        -------
        bool
            True if two bounding boxes overlap

        """
        bb1_p1, bb1_p2 = self.p_min.coor, self.p_max.coor
        bb2_p1, bb2_p2 = bb.p_min.coor, bb.p_max.coor
        return bool(
            np.all(
                (bb1_p2 >= bb2_p1 if include_bounds else bb1_p2 > bb2_p1)
                & (bb1_p1 <= bb2_p2 if include_bounds else bb1_p1 < bb2_p2),
            ),
        )

    @property
    def volume(self) -> DEFAULT_FLOAT:
        """Returns the volume of the bounding box"""
        if not hasattr(self, "_volume"):
            self._volume = np.prod(self.p_max.coor - self.p_min.coor).astype(
                DEFAULT_FLOAT,
            )
        return self._volume

    @property
    def perimeter(self) -> DEFAULT_FLOAT:
        return DEFAULT_FLOAT(2 * np.sum(self.p_max.coor - self.p_min.coor))

    @property
    def vertices(self) -> PointArrayND:
        if not hasattr(self, "_vertices"):
            vertices = list(
                product(*zip(self.p_min.coor, self.p_max.coor, strict=True))
            )
            self._vertices = PointArrayND(
                np.asarray(vertices, dtype=DEFAULT_FLOAT)
            )
        return self._vertices

    def get_patch(self, **rect_patch_kwargs) -> Patch:
        if self.p_min.dim != 2:
            raise ValueError("For plotting, bounding box must be 2D")
        (xl, yl), (xu, yu) = self.p_min.coor, self.p_max.coor
        return Rectangle((xl, yl), xu - xl, yu - yl, **rect_patch_kwargs)
