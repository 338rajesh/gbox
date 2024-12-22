import gbox as gb
import pytest
import numpy as np

# ==================== FIXTURES ====================


@pytest.fixture
def bounding_box():
    def _bbox(a):
        return gb.BoundingBox(
            lower_bound=[0.0] * a,
            upper_bound=[1.0] * a,
        )

    return _bbox


@pytest.fixture
def bounding_box_two_dim():
    return gb.BoundingBox(
        lower_bound=[0.0, 0.0],
        upper_bound=[1.0, 1.0],
    )


@pytest.fixture
def bounding_box_three_dim():
    return gb.BoundingBox(
        lower_bound=[-1.0, -1.0, -1.0],
        upper_bound=[1.0, 1.0, 1.0],
    )


@pytest.fixture
def bounding_box_five_dim():
    return gb.BoundingBox(
        lower_bound=[0.0, 0.0, 0.0, 0.0, 0.0],
        upper_bound=[1.0, 1.0, 1.0, 1.0, 1.0],
    )


# ==================== TESTS ====================


class TestBoundingBox:
    def test_dim(self, bounding_box):
        for i in range(1, 6):
            assert bounding_box(i).dim == i

    def test_vertices_2(self, bounding_box_two_dim):
        vertices = bounding_box_two_dim.vertices
        assert np.array_equal(
            vertices, np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
        )

    def test_vertices_3(self, bounding_box_three_dim):
        vertices = bounding_box_three_dim.vertices
        expected_vertices = np.array(
            [
                (-1.0, -1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0),
            ]
        )
        assert np.array_equal(vertices, expected_vertices)

    def test_has_point_2(self, bounding_box_two_dim):
        assert bounding_box_two_dim.has_point((0.5, 0.5))

    def test_has_no_point_2(self, bounding_box_two_dim):
        assert not bounding_box_two_dim.has_point((1.5, 1.5))

    def test_has_no_point_3(self, bounding_box_three_dim):
        assert not bounding_box_three_dim.has_point((1.5, 1.5, 1.5))

    def test_bounds_length_equality(self):
        with pytest.raises(AssertionError):
            gb.BoundingBox(
                lower_bound=[0.0, 0.0],
                upper_bound=[1.0, 1.0, 1.0],
            )

    def test_bounds_order(self):
        with pytest.raises(AssertionError):
            gb.BoundingBox(
                lower_bound=[1.0, 1.0],
                upper_bound=[0.0, 0.0],
            )
