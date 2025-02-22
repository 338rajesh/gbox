from hypothesis import given, strategies as st, settings as hypothesis_settings
from gbox import BoundingBox, TypeConfig
import pytest
import numpy as np
from utils import get_output_dir, gb_plotter
import pathlib


OUTPUT_DIR = get_output_dir(
    pathlib.Path(__file__).parent / "__output" / "test_base_point"
)


class TestBoundingBox:
    @hypothesis_settings(max_examples=10)
    @given(
        n=st.integers(min_value=2, max_value=10),
        lb=st.floats(min_value=-10.0, max_value=0.0),
        ub=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_b_box(self, n, lb, ub):
        lbl, ubl = [lb] * n, [ub] * n
        bb = BoundingBox(lower_bound=lbl, upper_bound=ubl)
        #
        assert bb.dim == n
        assert bb.vertices is not None

        assert bb.has_point([(lb + ub) * 0.5] * n)
        assert not bb.has_point([lb - 1.0] * n)

        with pytest.raises(ValueError):
            BoundingBox(lower_bound=[lb] * (n + 1), upper_bound=[ub] * n)

        with pytest.raises(ValueError):
            BoundingBox(lower_bound=ubl, upper_bound=lbl)

    def test_bounds_ele_type(self):
        with pytest.raises(ValueError):
            BoundingBox(
                lower_bound=[1.0, 1.0],
                upper_bound=[0.0, '0.0']  # type: ignore
            )

    def test_volume(self):
        bb = BoundingBox(lower_bound=[0.2, 0.1], upper_bound=[1.0, 1.6])
        bb_vol = 1.2
        assert bb.volume == pytest.approx(bb_vol)

        for i in [np.float64, np.float32, np.float16]:
            TypeConfig.set_float_type(i)
            bb = BoundingBox(lower_bound=[0.2, 0.1], upper_bound=[1.0, 1.6])
            assert type(bb.volume) is i

    def test_full_overlap(self):
        # Test case where the boxes are identical, so they fully overlap
        bb1 = BoundingBox([0, 0], [5, 5])
        bb2 = BoundingBox([0, 0], [5, 5])
        bb3 = BoundingBox([1, 1], [3, 3])
        assert bb1.overlaps_with(bb2)
        assert bb1.overlaps_with(bb3)

    def test_partial_overlap(self):
        # Test case where the boxes partially overlap
        bb1 = BoundingBox([0, 0], [5, 5])
        bb2 = BoundingBox([3, 3], [7, 7])
        bb3 = BoundingBox([-3, -3], [1, 1])
        bb4 = BoundingBox([-3, 2], [1, 7])
        assert bb1.overlaps_with(bb2)
        assert bb1.overlaps_with(bb3)
        assert bb1.overlaps_with(bb4)

    def test_no_overlap(self):
        # Test case where the boxes don't overlap at all
        bb1 = BoundingBox([0, 0], [5, 5])
        bb2 = BoundingBox([6, 6], [10, 10])
        assert not bb1.overlaps_with(bb2)

    def test_edge_touching_overlap(self):
        # Test case where boxes touch at the edge but don't overlap
        bb1 = BoundingBox([0, 0], [5, 5])
        bb2 = BoundingBox([5, 0], [10, 5])
        assert bb1.overlaps_with(bb2, incl_bounds=True)
        assert not bb1.overlaps_with(bb2, incl_bounds=False)

    def test_bbox_plots(self, test_plots):
        if not test_plots:
            pytest.skip()

        with gb_plotter(OUTPUT_DIR / "bbox.png") as (fig, axs):
            bb = BoundingBox([0, 0], [5, 5])
            bb.plot(axs)
            axs.set_title("Bounding Box")

        with pytest.raises(ValueError):
            with gb_plotter(OUTPUT_DIR / "bbox_1.png") as (fig, axs):
                BoundingBox([0, 0, 1], [5, 5, 10]).plot(axs)
