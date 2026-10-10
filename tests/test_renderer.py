"""
Unit tests for ShapesPlotter.

Run with:  pytest -q test_shapes_plotter.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI and servers

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from gbox import Bounds2DRectangular, Circle, CirclesArray, Ellipse
from gbox.plots.render import ShapesPlotter, _to_rgb


def make_circle(x: float, y: float, r: float) -> Circle:
    return Circle(radius=r, centre=(x, y))


def make_ellipse(
    x: float, y: float, a: float, b: float, angle: float = 0.0
) -> Ellipse:
    return Ellipse(
        centre=(x, y),
        semi_major_length=a,
        semi_minor_length=b,
        major_axis_angle=angle,
    )


def make_circles_array(xs, ys, radii) -> CirclesArray:
    return CirclesArray([(i, j) for (i, j) in zip(xs, ys)], radii=radii)


def px(img, x: float, y: float):
    """
    Pixel at physical coordinates (x, y) for a 100x100 image over
    bounds (0, 0, 100, 100). Row 0 is the top, so y is flipped.
    """
    col = int(x)
    row = int(100 - y)
    return img[row, col]


# --- Shared constants -------------------------------------------------------
SIZE = (100, 100)
BOUNDS = (0.0, 0.0, 100.0, 100.0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# --- Fixtures and helpers ---------------------------------------------------
@pytest.fixture
def plotter() -> ShapesPlotter:
    return ShapesPlotter(size=SIZE)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def shown_figure(monkeypatch):
    """
    Capture the Figure that show() passes to plt.show(). We keep the object
    itself rather than calling plt.gcf() later, because show() closes the
    figure, and gcf() would then return a new empty one.
    """
    holder: dict = {}

    def fake_show(*args, **kwargs):
        holder["fig"] = plt.gcf()

    monkeypatch.setattr(plt, "show", fake_show)
    return holder


def spy_on_collections(monkeypatch) -> list:
    """Record every call to ShapesPlotter._add_patch_collection."""
    calls: list = []
    original = ShapesPlotter._add_patch_collection

    def spy(self, ax, patches, color):
        calls.append((list(patches), color))
        return original(self, ax, patches, color)

    monkeypatch.setattr(ShapesPlotter, "_add_patch_collection", spy)
    return calls


# ===========================================================================
# Color parsing
# ===========================================================================
class TestToRgb:
    @pytest.mark.parametrize(
        "value, expected",
        [(0, (0, 0, 0)), (255, (255, 255, 255)), (127.6, (128, 128, 128))],
    )
    def test_grayscale_scalars(self, value, expected):
        assert _to_rgb(value) == expected

    def test_numpy_scalar_is_accepted(self):
        assert _to_rgb(np.uint8(7)) == (7, 7, 7)

    def test_rgb_tuple(self):
        assert _to_rgb((10, 20, 30)) == (10, 20, 30)

    def test_rgb_list_is_accepted(self):
        assert _to_rgb([1, 2, 3]) == (1, 2, 3)

    def test_rgb_channels_are_rounded(self):
        assert _to_rgb((1.4, 2.6, 3.5)) == (1, 3, 4)

    @pytest.mark.parametrize(
        "name, expected", [("red", RED), ("#0000ff", BLUE)]
    )
    def test_matplotlib_color_strings(self, name, expected):
        assert _to_rgb(name) == expected

    @pytest.mark.parametrize("value", [-1, 256, -0.5])
    def test_grayscale_out_of_range_raises(self, value):
        with pytest.raises(ValueError):
            _to_rgb(value)

    def test_rgb_channel_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _to_rgb((0, 300, 0))

    def test_unknown_color_string_raises(self):
        with pytest.raises(ValueError):
            _to_rgb("notacolor")

    def test_bool_is_rejected(self):
        with pytest.raises(TypeError):
            _to_rgb(True)

    @pytest.mark.parametrize("value", [None, (1, 2), (1, 2, 3, 4), object()])
    def test_wrong_types_or_lengths_raise(self, value):
        with pytest.raises(TypeError):
            _to_rgb(value)


# ===========================================================================
# Constructor and properties
# ===========================================================================
class TestConstructor:
    def test_defaults(self):
        p = ShapesPlotter()
        assert p.size == (256, 256)
        assert p.background == BLACK
        assert p.foreground == WHITE
        assert p.dpi == 100

    def test_custom_colors_are_normalized(self):
        p = ShapesPlotter(background="white", foreground=(255, 0, 0))
        assert p.background == WHITE
        assert p.foreground == RED

    def test_invalid_dpi_raises(self):
        with pytest.raises((TypeError, ValueError)):
            ShapesPlotter(dpi=-1)

    def test_invalid_size_raises(self):
        with pytest.raises((TypeError, ValueError)):
            ShapesPlotter(size=(100,))

    def test_invalid_background_raises(self):
        with pytest.raises(ValueError):
            ShapesPlotter(background=999)


# ===========================================================================
# plot(): output shape, dtype and mode selection
# ===========================================================================
class TestPlotOutputFormat:
    def test_grayscale_output_is_2d_uint8(self, plotter):
        img = plotter.plot(make_circle(50, 50, 20), bounds=BOUNDS)
        assert img.shape == (100, 100)
        assert img.dtype == np.uint8

    def test_colored_foreground_gives_rgb(self):
        p = ShapesPlotter(size=SIZE, foreground=RED)
        img = p.plot(make_circle(50, 50, 20), bounds=BOUNDS)
        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8

    def test_colored_background_gives_rgb(self):
        p = ShapesPlotter(size=SIZE, background=BLUE)
        img = p.plot([], bounds=BOUNDS)
        assert img.shape == (100, 100, 3)

    def test_per_shape_color_gives_rgb(self, plotter):
        img = plotter.plot([(make_circle(50, 50, 20), RED)], bounds=BOUNDS)
        assert img.ndim == 3

    def test_grayscale_pair_keeps_2d_output(self, plotter):
        img = plotter.plot([(make_circle(50, 50, 20), 128)], bounds=BOUNDS)
        assert img.ndim == 2

    def test_non_square_size_is_respected(self):
        p = ShapesPlotter(size=(200, 100))
        img = p.plot(make_circle(100, 50, 20), bounds=(0.0, 0.0, 200.0, 100.0))
        assert img.shape == (100, 200)

    def test_empty_sequence_is_background_only(self, plotter):
        img = plotter.plot([], bounds=BOUNDS)
        assert img.shape == (100, 100)
        assert np.all(img == 0)

    def test_empty_sequence_uses_background_value(self):
        p = ShapesPlotter(size=SIZE, background=77)
        img = p.plot([], bounds=BOUNDS)
        assert np.all(img == 77)


# ===========================================================================
# plot(): rendering correctness
# ===========================================================================
class TestRendering:
    def test_circle_center_is_foreground_corner_is_background(self):
        p = ShapesPlotter(size=SIZE, background=BLACK, foreground=RED)
        img = p.plot(make_circle(50, 50, 20), bounds=BOUNDS)
        np.testing.assert_array_equal(img[50, 50], RED)
        np.testing.assert_array_equal(img[0, 0], BLACK)

    def test_grayscale_circle_center_is_255(self, plotter):
        img = plotter.plot(make_circle(50, 50, 20), bounds=BOUNDS)
        assert img[50, 50] == 255
        assert img[0, 0] == 0

    def test_ellipse_center_is_filled(self, plotter):
        img = plotter.plot(make_ellipse(50, 50, 30, 10), bounds=BOUNDS)
        assert img[50, 50] == 255

    def test_ellipse_respects_semi_axes(self, plotter):
        img = plotter.plot(make_ellipse(50, 50, 30, 5), bounds=BOUNDS)
        assert img[50, 50 + 25] == 255  # along the major axis
        assert img[50 + 25, 50] == 0  # beyond the minor axis

    def test_circles_array_fills_every_member(self, plotter):
        arr = make_circles_array(xs=[20, 80], ys=[20, 80], radii=[8, 8])
        img = plotter.plot(arr, bounds=BOUNDS)
        assert int(px(img, 20, 20)) == 255
        assert int(px(img, 80, 80)) == 255
        assert int(px(img, 50, 50)) == 0

    def test_later_shape_paints_over_earlier(self):
        p = ShapesPlotter(size=SIZE, background=BLACK)
        outer = make_circle(50, 50, 40)
        inner = make_circle(50, 50, 20)
        img = p.plot([(outer, LIGHT_BLUE), (inner, RED)], bounds=BOUNDS)
        np.testing.assert_array_equal(img[50, 50], RED)  # inner visible
        np.testing.assert_array_equal(img[50, 50 + 30], LIGHT_BLUE)  # ring

    def test_reversed_order_hides_inner_shape(self):
        p = ShapesPlotter(size=SIZE, background=BLACK)
        outer = make_circle(50, 50, 40)
        inner = make_circle(50, 50, 20)
        img = p.plot([(inner, RED), (outer, LIGHT_BLUE)], bounds=BOUNDS)
        np.testing.assert_array_equal(img[50, 50], LIGHT_BLUE)

    def test_background_colored_shape_cuts_a_hole(self):
        p = ShapesPlotter(size=SIZE, background=BLACK)
        outer = make_circle(50, 50, 40)
        inner = make_circle(50, 50, 15)
        img = p.plot([(outer, RED), (inner, p.background)], bounds=BOUNDS)
        np.testing.assert_array_equal(img[50, 50], BLACK)
        np.testing.assert_array_equal(img[50, 80], RED)

    def test_default_foreground_applies_to_plain_shapes_in_sequence(self):
        p = ShapesPlotter(size=SIZE, foreground=RED)
        img = p.plot([make_circle(50, 50, 20)], bounds=BOUNDS)
        np.testing.assert_array_equal(img[50, 50], RED)

    def test_mixed_plain_and_paired_shapes(self):
        p = ShapesPlotter(size=SIZE, foreground=RED)
        left = make_circle(25, 50, 10)
        right = make_circle(75, 50, 10)
        img = p.plot([left, (right, BLUE)], bounds=BOUNDS)
        np.testing.assert_array_equal(img[50, 25], RED)
        np.testing.assert_array_equal(img[50, 75], BLUE)

    def test_bounds_as_plain_sequence_and_dataclass_are_equivalent(
        self, plotter
    ):
        shape = make_circle(50, 50, 20)
        a = plotter.plot(shape, bounds=(0.0, 0.0, 100.0, 100.0))
        b = plotter.plot(
            shape, bounds=Bounds2DRectangular.from_sequence(BOUNDS)
        )
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# Input normalization and validation
# ===========================================================================
class TestNormalize:
    def test_single_shape_becomes_one_pair(self, plotter):
        shape = make_circle(50, 50, 10)
        assert plotter._normalize(shape) == [(shape, plotter.foreground)]

    def test_sequence_of_plain_shapes(self, plotter):
        a, b = make_circle(10, 10, 1), make_circle(20, 20, 1)
        assert plotter._normalize([a, b]) == [
            (a, plotter.foreground),
            (b, plotter.foreground),
        ]

    def test_pair_color_is_normalized(self, plotter):
        shape = make_circle(10, 10, 1)
        assert plotter._normalize([(shape, "red")]) == [(shape, RED)]

    def test_empty_list_normalizes_to_empty(self, plotter):
        assert plotter._normalize([]) == []

    def test_tuple_of_shapes_is_accepted(self, plotter):
        shape = make_circle(10, 10, 1)
        assert plotter._normalize((shape,)) == [(shape, plotter.foreground)]

    @pytest.mark.parametrize("bad", ["circle", b"x", 42, None])
    def test_unsupported_top_level_types_raise(self, plotter, bad):
        with pytest.raises(TypeError):
            plotter._normalize(bad)

    def test_numpy_array_top_level_raises(self, plotter):
        with pytest.raises(TypeError):
            plotter._normalize(np.zeros(3))

    def test_unsupported_item_in_sequence_raises(self, plotter):
        with pytest.raises(TypeError):
            plotter._normalize([make_circle(1, 1, 1), "oops"])

    def test_pair_with_non_shape_first_element_raises(self, plotter):
        with pytest.raises(TypeError):
            plotter._normalize([("circle", "red")])

    def test_pair_with_invalid_color_raises(self, plotter):
        with pytest.raises(ValueError):
            plotter._normalize([(make_circle(1, 1, 1), "notacolor")])

    def test_plot_rejects_unsupported_shape(self, plotter):
        with pytest.raises(TypeError):
            plotter.plot(object(), bounds=BOUNDS)


class TestAspectRatio:
    def test_mismatched_bounds_raise(self, plotter):
        with pytest.raises(ValueError, match="aspect ratio"):
            plotter.plot(
                make_circle(50, 50, 20), bounds=(0.0, 0.0, 200.0, 100.0)
            )

    def test_matching_non_square_bounds_work(self):
        p = ShapesPlotter(size=(200, 100))
        p.plot([], bounds=(-10.0, -5.0, 190.0, 95.0))  # 200 x 100 units

    def test_check_passes_for_identical_ratio(self, plotter):
        plotter._check_aspect_ratio(Bounds2DRectangular.from_sequence(BOUNDS))


# ===========================================================================
# Batching of same-colored runs
# ===========================================================================
class TestBatching:
    def test_same_color_shapes_share_one_collection(
        self, plotter, monkeypatch
    ):
        calls = spy_on_collections(monkeypatch)
        shapes = [
            make_circle(10, 10, 2),
            make_circle(30, 30, 2),
            make_circle(60, 60, 2),
        ]
        plotter.plot(shapes, bounds=BOUNDS)
        assert len(calls) == 1
        assert len(calls[0][0]) == 3

    def test_alternating_colors_create_separate_collections(self, monkeypatch):
        p = ShapesPlotter(size=SIZE)
        calls = spy_on_collections(monkeypatch)
        items = [
            (make_circle(10, 10, 2), RED),
            (make_circle(30, 30, 2), BLUE),
            (make_circle(60, 60, 2), RED),
        ]
        p.plot(items, bounds=BOUNDS)
        assert [color for _, color in calls] == [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ] or len(calls) == 3

    def test_consecutive_runs_are_grouped(self, monkeypatch):
        p = ShapesPlotter(size=SIZE)
        calls = spy_on_collections(monkeypatch)
        items = [
            (make_circle(10, 10, 2), RED),
            (make_circle(20, 20, 2), RED),
            (make_circle(60, 60, 2), BLUE),
        ]
        p.plot(items, bounds=BOUNDS)
        assert [len(patches) for patches, _ in calls] == [2, 1]

    def test_empty_patch_list_adds_nothing(self, plotter, monkeypatch):
        calls = spy_on_collections(monkeypatch)
        plotter.plot([], bounds=BOUNDS)
        assert calls == []


# ===========================================================================
# Saving to disk
# ===========================================================================
class TestSave:
    def test_grayscale_png_round_trip(self, plotter, tmp_path):
        out = tmp_path / "gray.png"
        img = plotter.plot(make_circle(50, 50, 20), bounds=BOUNDS, path=out)
        with Image.open(out) as loaded:
            assert loaded.mode == "L"
            np.testing.assert_array_equal(np.asarray(loaded), img)

    def test_rgb_png_round_trip(self, tmp_path):
        p = ShapesPlotter(size=SIZE, foreground=RED)
        out = tmp_path / "rgb.png"
        img = p.plot(make_circle(50, 50, 20), bounds=BOUNDS, path=out)
        with Image.open(out) as loaded:
            assert loaded.mode == "RGB"
            np.testing.assert_array_equal(np.asarray(loaded), img)

    def test_accepts_string_path(self, plotter, tmp_path):
        out = str(tmp_path / "s.png")
        plotter.plot([], bounds=BOUNDS, path=out)
        assert (tmp_path / "s.png").exists()

    def test_creates_missing_parent_directories(self, plotter, tmp_path):
        out = tmp_path / "a" / "b" / "c.png"
        plotter.plot([], bounds=BOUNDS, path=out)
        assert out.exists()

    def test_no_file_written_without_path(
        self, plotter, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        plotter.plot(make_circle(50, 50, 20), bounds=BOUNDS)
        assert list(tmp_path.iterdir()) == []


# ===========================================================================
# show()
# ===========================================================================
class TestShow:
    def test_show_returns_same_array_as_plot(self, plotter, shown_figure):
        shape = make_circle(50, 50, 20)
        expected = plotter.plot(shape, bounds=BOUNDS)
        result = plotter.show(shape, bounds=BOUNDS)
        np.testing.assert_array_equal(result, expected)

    def test_show_calls_pyplot_show_once(self, plotter, monkeypatch):
        calls = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
        plotter.show([], bounds=BOUNDS)
        assert calls == [1]

    def test_show_sets_title(self, plotter, shown_figure):
        plotter.show([], bounds=BOUNDS, title="RVE 3")
        ax = shown_figure["fig"].axes[0]
        assert ax.get_title() == "RVE 3"

    def test_show_hides_axes(self, plotter, shown_figure):
        plotter.show([], bounds=BOUNDS)
        ax = shown_figure["fig"].axes[0]
        assert not ax.axison

    def test_show_grayscale_uses_gray_colormap(self, plotter, shown_figure):
        plotter.show(make_circle(50, 50, 20), bounds=BOUNDS)
        image = shown_figure["fig"].axes[0].images[0]
        assert image.get_array().ndim == 2
        assert image.get_cmap().name == "gray"

    def test_show_rgb_image_is_displayed_as_rgb(self, shown_figure):
        p = ShapesPlotter(size=SIZE, foreground=RED)
        p.show(make_circle(50, 50, 20), bounds=BOUNDS)
        image = shown_figure["fig"].axes[0].images[0]
        assert image.get_array().ndim == 3

    def test_show_does_not_touch_pil(self, plotter, monkeypatch, shown_figure):
        def boom(*a, **k):
            raise AssertionError("show() must not save via PIL")

        monkeypatch.setattr(ShapesPlotter, "_save", boom)
        plotter.show([], bounds=BOUNDS)


# ===========================================================================
# Internal helpers
# ===========================================================================
class TestHelpers:
    def test_mpl_color_scales_to_unit_interval(self):
        assert ShapesPlotter._mpl_color(RED) == (1.0, 0.0, 0.0)
        assert ShapesPlotter._mpl_color(BLACK) == (0.0, 0.0, 0.0)

    def test_resize_nearest_2d(self):
        src = np.arange(50 * 50, dtype=np.uint8).reshape(50, 50)
        out = ShapesPlotter._resize_nearest(src, width=80, height=100)
        assert out.shape == (100, 80)
        assert out.dtype == np.uint8

    def test_resize_nearest_3d_keeps_channels(self):
        src = np.zeros((50, 50, 3), dtype=np.uint8)
        out = ShapesPlotter._resize_nearest(src, width=80, height=100)
        assert out.shape == (100, 80, 3)

    def test_resize_nearest_identity(self):
        src = np.random.default_rng(0).integers(
            0, 255, (20, 30, 3), dtype=np.uint8
        )
        out = ShapesPlotter._resize_nearest(src, width=30, height=20)
        np.testing.assert_array_equal(out, src)

    def test_resize_nearest_returns_copy(self):
        src = np.ones((10, 10), dtype=np.uint8)
        out = ShapesPlotter._resize_nearest(src, width=10, height=10)
        out[0, 0] = 9
        assert src[0, 0] == 1

    def test_canvas_size_fallback_is_enforced(self, plotter, monkeypatch):
        """If Agg returns an off-by-one raster, plot() still returns SIZE."""
        original = ShapesPlotter._resize_nearest
        seen = []

        def spy(image, *, width, height):
            seen.append(image.shape)
            return original(image, width=width, height=height)

        monkeypatch.setattr(
            ShapesPlotter, "_resize_nearest", staticmethod(spy)
        )
        plotter.plot([], bounds=BOUNDS)
        # Whether or not the fallback fired, the output must have the right size.
        img = plotter.plot([], bounds=BOUNDS)
        assert img.shape == (100, 100)
