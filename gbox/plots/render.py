from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import matplotlib

# Important for headless/server/dataset-generation environments.
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse as MplEllipse, Circle as MplCircle

import numpy as np
import numpy.typing as npt

from ..core.utils import Validator, Bounds2DRectangular
from ..shapes.shapes_2d import (
    Circle as GBCircle,
    Ellipse as GBEllipse,
    CirclesArray as GBCirclesArray,
)


class ShapesPlotter:
    """
    Plot 2D geometric shapes into raster images.

    Parameters
    ----------
    size : Sequence[int, int], default=(256, 256)
        Output image size as ``(width, height)`` in pixels.
    background : int or float, default=0
        Background pixel value.
    foreground : int or float, default=255
        Shape pixel value.
    dpi : int, default=100
        Matplotlib rendering DPI.

    Notes
    -----
    ``plot()`` always returns a NumPy array. If ``path`` is provided, the
    rendered image is additionally written to the given location.

    The current implementation produces a single-channel ``uint8`` image.
    """

    __slots__ = (
        "_size",
        "_background",
        "_foreground",
        "_dpi",
    )

    def __init__(
        self,
        *,
        size: Sequence[float, float] = (256, 256),
        background: int | float = 0,
        foreground: int | float = 255,
        dpi: int = 100,
    ):
        Validator.sequence(size, ele_type=int, length=2, name="Image size")
        Validator.int(background, low=0, high=255, name="Background")
        Validator.int(foreground, low=0, high=255, name="Foreground")
        Validator.int(dpi, low=0, name="DPI")

        self._size = size
        self._background = background
        self._foreground = foreground
        self._dpi = dpi

    @property
    def size(self) -> tuple[float, float]:
        """Output image size as ``(width, height)`` in pixels."""
        return self._size

    @property
    def background(self) -> int | float:
        """Background pixel value."""
        return self._background

    @property
    def foreground(self) -> int | float:
        """Shape pixel value."""
        return self._foreground

    @property
    def dpi(self) -> int:
        """Rendering DPI."""
        return self._dpi

    def plot(
        self,
        shapes: Any,
        *,
        bounds: Sequence[float] | Bounds2DRectangular,
        path: str | Path | None = None,
    ) -> npt.NDArray[np.uint8]:
        """
        Plot shapes and return the resulting image as a NumPy array.

        Parameters
        ----------
        shapes :
            A single supported shape or a sequence of shapes.

        bounds : Bounds2DRectangular
            Physical plotting bounds in the form
            ``(min_x, min_y, max_x, max_y)``.

        path : str or pathlib.Path, optional
            If provided, save the rendered image to this location.

        Returns
        -------
        numpy.ndarray
            A two-dimensional ``uint8`` array with shape
            ``(height, width)``.

        Examples
        --------
        Plot an RVE containing multiple shapes:

        >>> image = plotter.plot(
        ...     shapes,
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ... )

        Plot and save simultaneously:

        >>> image = plotter.plot(
        ...     shapes,
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ...     path="rve.png",
        ... )
        """

        bounds = Bounds2DRectangular.from_sequence(bounds)
        self._check_aspect_ratio(bounds)

        width, height = self._size

        fig = Figure(
            figsize=(width / self._dpi, height / self._dpi),
            dpi=self._dpi,
            facecolor=self._mpl_color(self._background),
        )

        canvas = FigureCanvasAgg(fig)

        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

        ax.set_xlim(bounds.x_min, bounds.x_max)
        ax.set_ylim(bounds.y_min, bounds.y_max)
        ax.set_aspect("equal")
        ax.axis("off")

        self._add_shapes(ax, shapes)

        canvas.draw()

        # RGBA buffer -> NumPy array.
        rgba = np.asarray(canvas.buffer_rgba())

        # Use the red channel because the renderer currently produces a
        # monochrome image.
        image = rgba[..., 0].copy()

        # Matplotlib/Agg can occasionally produce a size differing by one
        # pixel because of backend rounding. Enforce the requested size.
        if image.shape != (height, width):
            image = self._resize_nearest(
                image,
                width=width,
                height=height,
            )

        if path is not None:
            self._save(image, path)

        return image

    def _check_aspect_ratio(
        self,
        bounds: Bounds2DRectangular,
        *,
        rel_tol: float = 1e-6,
    ) -> None:
        """
        Ensure the requested bounds have the same aspect ratio as the
        output image size, since ``set_aspect("equal")`` requires this
        to fill the canvas without distortion or letterboxing.
        """
        width, height = self._size
        image_ratio = width / height
        bounds_ratio = (bounds.x_max - bounds.x_min) / (
            bounds.y_max - bounds.y_min
        )

        if not math.isclose(image_ratio, bounds_ratio, rel_tol=rel_tol):
            raise ValueError(
                f"Image size {self._size} has aspect ratio {image_ratio:.6g} "
                f"(width/height), but bounds {bounds} have aspect ratio "
                f"{bounds_ratio:.6g} ((x_max-x_min)/(y_max-y_min)). These "
                "must match so that 'equal' aspect scaling fills the full "
                "canvas without letterboxing."
            )

    def _add_shapes(self, ax, shapes) -> None:
        """
        Convert supported geometry objects into Matplotlib collections.
        """
        from ..shapes.shapes_2d import (
            Circle,
            CirclesArray,
            Ellipse,
        )

        if isinstance(shapes, CirclesArray):
            self._add_circles_array(ax, shapes)
            return

        if isinstance(shapes, Circle):
            self._add_circle(ax, shapes)
            return

        if isinstance(shapes, Ellipse):
            self._add_ellipse(ax, shapes)
            return

        # A sequence is useful for heterogeneous collections such as:
        #
        # [Circle(...), Ellipse(...), Circle(...)]
        #
        # Avoid treating strings/bytes as shape sequences.
        if isinstance(shapes, Sequence) and not isinstance(
            shapes, (str, bytes, np.ndarray)
        ):
            self._add_shape_sequence(ax, shapes)
            return

        raise TypeError(f"Unsupported shape type: {type(shapes).__name__}")

    def _add_shape_sequence(self, ax, shapes) -> None:
        """
        Add a sequence of individual shapes.

        Individual shapes are grouped into a single
        PatchCollection where possible.
        """
        patches = []

        for shape in shapes:
            if isinstance(shape, GBCircle):
                patches.append(self._circle_to_patch(shape))
            elif isinstance(shape, GBEllipse):
                patches.append(self._ellipse_to_patch(shape))
            elif isinstance(shape, GBCirclesArray):
                # Keep vectorized circle arrays on their optimized path.
                if patches:
                    self._add_patch_collection(ax, patches)
                    patches = []

                self._add_circles_array(ax, shape)
            else:
                raise TypeError(
                    f"Unsupported shape in sequence: {type(shape).__name__}"
                )

        if patches:
            self._add_patch_collection(ax, patches)

    def _add_circle(self, ax, circle: GBCircle) -> None:
        self._add_patch_collection(
            ax,
            [self._circle_to_patch(circle)],
        )

    def _add_ellipse(self, ax, ellipse: GBEllipse) -> None:
        self._add_patch_collection(
            ax,
            [self._ellipse_to_patch(ellipse)],
        )

    def _add_circles_array(self, ax, circles: GBCirclesArray) -> None:
        """
        Add a CirclesArray as a single PatchCollection.
        """
        # from matplotlib.patches import Circle as MplCircle

        patches = [
            MplCircle(
                (float(x), float(y)),
                radius=float(radius),
            )
            for x, y, radius in zip(
                circles.centres.x,
                circles.centres.y,
                circles.radii,
            )
        ]

        if patches:
            self._add_patch_collection(ax, patches)

    def _circle_to_patch(self, circle: GBCircle) -> MplCircle:
        """
        Convert a Circle geometry object to a Matplotlib circle patch
        """
        from matplotlib.patches import Circle as MplCircle

        return MplCircle(
            xy=(
                float(circle.position.x),
                float(circle.position.y),
            ),
            radius=float(circle.radius),
        )

    def _ellipse_to_patch(self, ellipse: GBEllipse):
        """
        Convert an Ellipse geometry object to a Matplotlib ellipse patch.
        """
        return MplEllipse(
            xy=(
                float(ellipse.position.x),
                float(ellipse.position.y),
            ),
            width=2.0 * float(ellipse.semi_major_length),
            height=2.0 * float(ellipse.semi_minor_length),
            angle=ellipse.position.orientation.degrees,
        )

    def _add_patch_collection(self, ax, patches) -> None:
        collection = PatchCollection(
            patches,
            facecolor=self._mpl_color(self._foreground),
            edgecolor="none",
            antialiased=True,
        )

        ax.add_collection(collection)

    def _save(
        self,
        image: npt.NDArray[np.uint8],
        path: str | Path,
    ) -> None:
        """
        Save an already-rendered image.

        PIL is intentionally only used at the persistence boundary.
        """
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(image, mode="L").save(path)

    @staticmethod
    def _mpl_color(value: int | float) -> tuple[float, float, float]:
        """
        Convert a grayscale pixel value into a Matplotlib grayscale color.
        """
        value = float(value)

        if not 0.0 <= value <= 255.0:
            raise ValueError("Pixel values must be between 0 and 255.")

        normalized = value / 255.0
        return normalized, normalized, normalized

    @staticmethod
    def _resize_nearest(
        image: npt.NDArray[np.uint8],
        *,
        width: int,
        height: int,
    ) -> npt.NDArray[np.uint8]:
        """
        Fallback resize used only if the backend returns an unexpected
        raster size.
        """
        y_indices = np.linspace(
            0,
            image.shape[0] - 1,
            height,
        ).astype(int)

        x_indices = np.linspace(
            0,
            image.shape[1] - 1,
            width,
        ).astype(int)

        return image[np.ix_(y_indices, x_indices)].copy()
