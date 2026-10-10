from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Ellipse as MplEllipse

from ..core.utils import Bounds2DRectangular, Validator
from ..shapes.shapes_2d import Circle as GBCircle
from ..shapes.shapes_2d import CirclesArray as GBCirclesArray
from ..shapes.shapes_2d import Ellipse as GBEllipse

_SHAPE_TYPES = (GBCircle, GBEllipse, GBCirclesArray)

RGB = tuple[int, int, int]
ColorLike = int | float | str | Sequence[float]


def _to_rgb(value: ColorLike) -> RGB:
    """
    Normalize a color specification to an ``(r, g, b)`` tuple of ints in 0-255.

    Accepted forms:
      * grayscale scalar: ``0`` ... ``255``
      * RGB triple: ``(r, g, b)`` with each channel in 0-255
      * any name or hex string understood by Matplotlib, e.g. ``"red"``, ``"#1f77b4"``
    """
    if isinstance(value, str):
        r, g, b = to_rgb(value)
        return (round(r * 255), round(g * 255), round(b * 255))

    if isinstance(value, (int, float, np.number)) and not isinstance(
        value, bool
    ):
        v = float(value)
        if not 0.0 <= v <= 255.0:
            raise ValueError("Grayscale values must be between 0 and 255.")
        g = round(v)
        return (g, g, g)

    if isinstance(value, Sequence) and len(value) == 3:
        channels = [float(c) for c in value]
        if any(not 0.0 <= c <= 255.0 for c in channels):
            raise ValueError("RGB channel values must be between 0 and 255.")
        return tuple(round(c) for c in channels)  # type: ignore[return-value]

    raise TypeError(
        "Color must be a grayscale value, an (r, g, b) triple, or a "
        f"Matplotlib color string; got {type(value).__name__}."
    )


class ShapesPlotter:
    """
    Plot 2D geometric shapes into raster images.

    Parameters
    ----------
    size : Sequence[int, int], default=(256, 256)
        Output image size as ``(width, height)`` in pixels.
    background : color, default=0
        Background color. Accepts a grayscale value (0-255), an
        ``(r, g, b)`` triple, or a Matplotlib color string.
    foreground : color, default=255
        Default shape color, with the same accepted forms as ``background``.
    dpi : int, default=100
        Matplotlib rendering DPI.

    Notes
    -----
    Shapes are painted in the order given, so later shapes appear on top of
    earlier ones. To make a shape visible inside another, give it a different
    color. To cut a hole, paint a shape with the background color.

    ``plot()`` returns a NumPy array:

    * shape ``(height, width)``, ``uint8`` if every color used in the call is
      grayscale (``r == g == b``);
    * shape ``(height, width, 3)``, ``uint8`` (RGB) otherwise.
    """

    __slots__ = (
        "_background",
        "_dpi",
        "_foreground",
        "_size",
    )

    def __init__(
        self,
        *,
        size: Sequence[float, float] = (256, 256),
        background: ColorLike = 0,
        foreground: ColorLike = 255,
        dpi: int = 100,
    ):
        Validator.as_sequence(size, ele_type=int, length=2, name="Image size")
        Validator.as_int(dpi, low=0, name="DPI")

        self._size = size
        self._background: RGB = _to_rgb(background)
        self._foreground: RGB = _to_rgb(foreground)
        self._dpi = dpi

    @property
    def size(self) -> tuple[float, float]:
        """Output image size as ``(width, height)`` in pixels."""
        return self._size

    @property
    def background(self) -> RGB:
        """Background color as an ``(r, g, b)`` tuple."""
        return self._background

    @property
    def foreground(self) -> RGB:
        """Default shape color as an ``(r, g, b)`` tuple."""
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
            A single shape, a sequence of shapes, or a sequence whose items
            are either shapes (drawn in the default foreground color) or
            ``(shape, color)`` pairs.

        bounds : Bounds2DRectangular
            Physical plotting bounds in the form
            ``(min_x, min_y, max_x, max_y)``.

        path : str or pathlib.Path, optional
            If provided, save the rendered image to this location.

        Returns
        -------
        numpy.ndarray
            ``uint8`` array of shape ``(height, width)`` for grayscale output,
            or ``(height, width, 3)`` for RGB output.

        Examples
        --------
        Inner circle visible on top of an outer one:

        >>> image = plotter.plot(
        ...     [(outer, "lightblue"), (inner, "red")],
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ... )

        Hole cut into a shape, saved to disk:

        >>> image = plotter.plot(
        ...     [outer, (inner, plotter.background)],
        ...     bounds=(0.0, 0.0, 100.0, 100.0),
        ...     path="ring.png",
        ... )
        """
        bounds = Bounds2DRectangular.from_sequence(bounds)
        self._check_aspect_ratio(bounds)

        items = self._normalize(shapes)
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

        self._add_items(ax, items)

        canvas.draw()

        rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()

        # Matplotlib/Agg can occasionally produce a size differing by one
        # pixel because of backend rounding. Enforce the requested size.
        if rgb.shape[:2] != (height, width):
            rgb = self._resize_nearest(rgb, width=width, height=height)

        colors_used = {self._background, *(color for _, color in items)}
        if all(r == g == b for r, g, b in colors_used):
            image = rgb[..., 0].copy()
        else:
            image = rgb

        if path is not None:
            self._save(image, path)

        return image

    def show(
        self,
        shapes: Any,
        *,
        bounds: Sequence[float] | Bounds2DRectangular,
        title: str | None = None,
    ) -> npt.NDArray[np.uint8]:
        """
        Plot shapes, display the image, and return the NumPy array.

        Displays inline in Jupyter/IPython notebooks and in interactive
        Matplotlib windows. Uses Matplotlib's ``pyplot``, so no PIL import
        is needed.
        """
        import matplotlib.pyplot as plt

        image = self.plot(shapes, bounds=bounds)

        fig, ax = plt.subplots()
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)

        plt.show()
        plt.close(fig)
        return image

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize(self, shapes: Any) -> list[tuple[Any, RGB]]:
        """
        Convert the user's input into a list of ``(shape, rgb_color)`` pairs.
        """
        if isinstance(shapes, _SHAPE_TYPES):
            return [(shapes, self._foreground)]

        # Avoid treating strings/bytes/arrays as shape sequences.
        if isinstance(shapes, Sequence) and not isinstance(
            shapes, (str, bytes, np.ndarray)
        ):
            items: list[tuple[Any, RGB]] = []
            for item in shapes:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], _SHAPE_TYPES)
                ):
                    shape, color = item
                    items.append((shape, _to_rgb(color)))
                elif isinstance(item, _SHAPE_TYPES):
                    items.append((item, self._foreground))
                else:
                    raise TypeError(
                        f"Unsupported shape in sequence: {type(item).__name__}"
                    )
            return items

        raise TypeError(f"Unsupported shape type: {type(shapes).__name__}")

    def _add_items(self, ax, items: list[tuple[Any, RGB]]) -> None:
        """
        Add shapes to the axes in order.

        Consecutive shapes with the same color are batched into one
        PatchCollection. The order of collections is preserved, so later
        shapes are painted over earlier ones.
        """
        run_color: RGB | None = None
        run: list[MplCircle | MplEllipse] = []

        for shape, color in items:
            if run and color != run_color:
                self._add_patch_collection(ax, run, run_color)
                run = []
            run_color = color
            run.extend(self._to_patches(shape))

        if run:
            self._add_patch_collection(ax, run, run_color)

    def _to_patches(self, shape: Any) -> list[MplCircle | MplEllipse]:
        if isinstance(shape, GBCircle):
            return [self._circle_to_patch(shape)]
        if isinstance(shape, GBEllipse):
            return [self._ellipse_to_patch(shape)]
        if isinstance(shape, GBCirclesArray):
            return [
                MplCircle((float(x), float(y)), radius=float(r))
                for x, y, r in zip(
                    shape.centres.x,
                    shape.centres.y,
                    shape.radii,
                )
            ]
        raise TypeError(f"Unsupported shape type: {type(shape).__name__}")

    def _circle_to_patch(self, circle: GBCircle) -> MplCircle:
        return MplCircle(
            xy=(float(circle.position.x), float(circle.position.y)),
            radius=float(circle.radius),
        )

    def _ellipse_to_patch(self, ellipse: GBEllipse) -> MplEllipse:
        return MplEllipse(
            xy=(float(ellipse.position.x), float(ellipse.position.y)),
            width=2.0 * float(ellipse.semi_major_length),
            height=2.0 * float(ellipse.semi_minor_length),
            angle=ellipse.position.orientation.degrees,
        )

    def _add_patch_collection(self, ax, patches, color: RGB) -> None:
        if not patches:
            return
        collection = PatchCollection(
            patches,
            facecolor=self._mpl_color(color),
            edgecolor="none",
            antialiased=True,
        )
        ax.add_collection(collection)

    def _check_aspect_ratio(
        self,
        bounds: Bounds2DRectangular,
        *,
        rel_tol: float = 1e-6,
    ) -> None:
        """
        Ensure the requested bounds match the output image's aspect ratio,
        so ``set_aspect("equal")`` fills the canvas without letterboxing.
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

    def _save(self, image: npt.NDArray[np.uint8], path: str | Path) -> None:
        """
        Save an already-rendered image. PIL is used only at this boundary.
        """
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)

    @staticmethod
    def _mpl_color(color: RGB) -> tuple[float, float, float]:
        """Convert an 0-255 RGB tuple into Matplotlib's 0-1 floats."""
        r, g, b = color
        return (r / 255.0, g / 255.0, b / 255.0)

    @staticmethod
    def _resize_nearest(
        image: npt.NDArray[np.uint8],
        *,
        width: int,
        height: int,
    ) -> npt.NDArray[np.uint8]:
        """
        Fallback resize used only if the backend returns an unexpected raster
        size. Works for both (H, W) and (H, W, C) arrays.
        """
        y_indices = np.linspace(0, image.shape[0] - 1, height).astype(int)
        x_indices = np.linspace(0, image.shape[1] - 1, width).astype(int)
        return image[np.ix_(y_indices, x_indices)].copy()
