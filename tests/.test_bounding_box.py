import pytest
import numpy as np
from gbox.base import BoundingBox, PointND, DEFAULT_FLOAT, PointArrayND

# =============================================
#               TEST FIXTURES
# =============================================


@pytest.fixture
def sample_2d_bbox():
    return BoundingBox([0.0, 0.0], [10.0, 10.0])


@pytest.fixture
def sample_3d_bbox():
    return BoundingBox([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])


@pytest.fixture
def overlapping_2d_bbox():
    return BoundingBox([5.0, 5.0], [15.0, 15.0])


@pytest.fixture
def non_overlapping_2d_bbox():
    return BoundingBox([11.0, 11.0], [15.0, 15.0])


# =============================================
#               BOUNDINGBOX TESTS
# =============================================


class TestBoundingBox:
    def test_initialization(self):
        # Test valid 2D initialization
        bb = BoundingBox([0.0, 0.0], [10.0, 10.0])
        assert isinstance(bb.p_min, PointND)
        assert isinstance(bb.p_max, PointND)
        assert np.array_equal(
            bb.p_min.coor, np.array([0.0, 0.0], dtype=DEFAULT_FLOAT)
        )
        assert np.array_equal(
            bb.p_max.coor, np.array([10.0, 10.0], dtype=DEFAULT_FLOAT)
        )

        # Test valid 3D initialization
        bb3d = BoundingBox([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        assert bb3d.p_min.dim == 3
        assert bb3d.p_max.dim == 3

        # Test initialization with numpy arrays
        bb_np = BoundingBox(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert isinstance(bb_np.p_min, PointND)

        # Test invalid initialization
        with pytest.raises(ValueError):
            BoundingBox([0.0], [10.0, 10.0])  # Dimension mismatch

        with pytest.raises(ValueError):
            BoundingBox([5.0, 5.0], [0.0, 0.0])  # Lower > upper

        with pytest.raises(TypeError):
            BoundingBox("invalid", [10.0, 10.0])  # type: ignore # Invalid type

    def test_properties(self, sample_2d_bbox, sample_3d_bbox):
        # Test volume property
        assert sample_2d_bbox.volume == 100.0
        assert sample_3d_bbox.volume == 1000.0

        # Test vertices property
        vertices_2d = sample_2d_bbox.vertices
        assert isinstance(vertices_2d, PointArrayND)
        assert vertices_2d.dim == 2
        assert len(vertices_2d) == 4  # 2^2 vertices for 2D box

        vertices_3d = sample_3d_bbox.vertices
        assert vertices_3d.dim == 3
        assert len(vertices_3d) == 8  # 2^3 vertices for 3D box

        # Verify some vertices
        expected_2d_vertices = np.array(
            [[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]],
            dtype=DEFAULT_FLOAT,
        )
        assert np.array_equal(
            np.sort(vertices_2d.coordinates, axis=0),
            np.sort(expected_2d_vertices, axis=0),
        )

    def test_has_point(self, sample_2d_bbox):
        # Test points inside the bounding box
        assert sample_2d_bbox.has_point([5.0, 5.0])
        assert sample_2d_bbox.has_point(PointND(5.0, 5.0))
        assert sample_2d_bbox.has_point([0.0, 0.0])  # On lower bound
        assert sample_2d_bbox.has_point([10.0, 10.0])  # On upper bound

        # Test points outside the bounding box
        assert not sample_2d_bbox.has_point([-1.0, 5.0])
        assert not sample_2d_bbox.has_point([11.0, 5.0])
        assert not sample_2d_bbox.has_point([5.0, -1.0])
        assert not sample_2d_bbox.has_point([5.0, 11.0])

        # Test dimension mismatch
        with pytest.raises(ValueError):
            sample_2d_bbox.has_point([5.0])  # 1D point

        with pytest.raises(ValueError):
            sample_2d_bbox.has_point([5.0, 5.0, 5.0])  # 3D point

    def test_overlaps(
        self, sample_2d_bbox, overlapping_2d_bbox, non_overlapping_2d_bbox
    ):
        # Test overlapping boxes
        assert sample_2d_bbox.overlaps(overlapping_2d_bbox)
        assert overlapping_2d_bbox.overlaps(sample_2d_bbox)

        # Test non-overlapping boxes
        assert not sample_2d_bbox.overlaps(non_overlapping_2d_bbox)
        assert not non_overlapping_2d_bbox.overlaps(sample_2d_bbox)

        # Test edge cases
        touching_bbox = BoundingBox([10.0, 10.0], [15.0, 15.0])
        assert not sample_2d_bbox.overlaps(
            touching_bbox
        )  # Not overlapping by default
        assert sample_2d_bbox.overlaps(
            touching_bbox, include_bounds=True
        )  # Overlapping when including bounds

        # Test with self
        assert sample_2d_bbox.overlaps(sample_2d_bbox)

        # Test dimension mismatch
        with pytest.raises(ValueError):
            sample_2d_bbox.overlaps(BoundingBox([0.0], [10.0, 10.0]))

    def test_equality(self, sample_2d_bbox):
        # Test equality with same bounds
        same_bbox = BoundingBox([0.0, 0.0], [10.0, 10.0])
        assert sample_2d_bbox == same_bbox

        # Test inequality with different bounds
        different_bbox = BoundingBox([0.0, 0.0], [5.0, 5.0])
        assert sample_2d_bbox != different_bbox

        # Test invalid comparison
        with pytest.raises(TypeError):
            _ = sample_2d_bbox == "not a bbox"
