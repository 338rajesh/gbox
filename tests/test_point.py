import math

import numpy as np
import pytest

from gbox.core.points import (
    ORIGIN_2D,
    ORIGIN_3D,
    Point2D,
    Point3D,
    PointArray2D,
    PointArrayND,
)
from gbox.core.utils import Angle, TransformationOrder


# ============================================================================
# PointND / Point2D / Point3D construction
# ============================================================================


class TestPointND:
    def test_constructs_from_coordinates(self):
        p = Point2D(1, 2)

        assert p.coordinates == (1.0, 2.0)

    def test_coordinates_are_converted_to_float(self):
        p = Point2D(1, 2)

        assert all(isinstance(x, float) for x in p.coordinates)

    def test_dimension(self):
        assert Point2D(1, 2).dim == 2
        assert Point3D(1, 2, 3).dim == 3

    def test_len(self):
        assert len(Point2D(1, 2)) == 2
        assert len(Point3D(1, 2, 3)) == 3

    def test_indexing(self):
        p = Point3D(1, 2, 3)

        assert p[0] == 1.0
        assert p[2] == 3.0

    def test_slicing(self):
        p = Point3D(1, 2, 3)

        assert p[1:] == (2.0, 3.0)

    def test_iteration(self):
        assert list(Point3D(1, 2, 3)) == [1.0, 2.0, 3.0]

    def test_repr_point2d(self):
        assert repr(Point2D(1, 2)) == "Point2D(x=1.0, y=2.0)"

    def test_repr_point3d(self):
        assert repr(Point3D(1, 2, 3)) == ("Point3D(x=1.0, y=2.0, z=3.0)")

    def test_equality(self):
        assert Point2D(1, 2) == Point2D(1, 2)

    def test_inequality(self):
        assert Point2D(1, 2) != Point2D(1, 3)

    def test_equality_with_other_type(self):
        assert Point2D(1, 2) != (1, 2)

    def test_from_sequence(self):
        p = Point2D.from_sequence([1, 2])

        assert p == Point2D(1, 2)

    def test_from_sequence_returns_same_point(self):
        p = Point2D(1, 2)

        assert Point2D.from_sequence(p) is p

    def test_from_sequence_wrong_dimension(self):
        with pytest.raises(TypeError):
            Point2D.from_sequence(1)

    def test_distance(self):
        p = Point2D(0, 0)
        q = Point2D(3, 4)

        assert p.distance_to(q) == pytest.approx(5.0)

    def test_distance_accepts_sequence(self):
        assert Point2D(0, 0).distance_to((3, 4)) == pytest.approx(5.0)

    def test_distance_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            Point2D(0, 0).distance_to(Point3D(0, 0, 0))

    def test_in_bounds(self):
        p = Point2D(5, 5)

        assert p.in_bounds((0, 0), (10, 10))
        assert not p.in_bounds((6, 0), (10, 10))

    def test_in_bounds_is_inclusive(self):
        p = Point2D(0, 10)

        assert p.in_bounds((0, 0), (10, 10))

    def test_in_bounds_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            Point2D(0, 0).in_bounds(
                Point3D(0, 0, 0),
                Point3D(1, 1, 1),
            )

    def test_is_close_to(self):
        p = Point2D(1, 2)

        assert p.is_close_to((1 + 1e-9, 2 + 1e-9))

    def test_is_close_to_detects_large_difference(self):
        p = Point2D(1, 2)

        assert not p.is_close_to((2, 3))

    def test_is_close_to_respects_tolerances(self):
        p = Point2D(1, 2)

        assert p.is_close_to((1.01, 2), atol=0.02, rtol=0)
        assert not p.is_close_to((1.01, 2), atol=0.001, rtol=0)

    def test_is_close_to_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            Point2D(1, 2).is_close_to((1, 2, 3))


# ============================================================================
# Point2D
# ============================================================================


class TestPoint2D:
    def test_x_y_properties(self):
        p = Point2D(3, 4)

        assert p.x == 3.0
        assert p.y == 4.0

    def test_slope(self):
        assert Point2D(0, 0).slope((2, 4)) == pytest.approx(2.0)

    def test_negative_slope(self):
        assert Point2D(0, 0).slope((2, -4)) == pytest.approx(-2.0)

    def test_vertical_slope(self):
        assert math.isinf(Point2D(1, 2).slope((1, 10)))

    def test_slope_uses_epsilon(self):
        result = Point2D(0, 0).slope((1e-7, 1))

        assert math.isinf(result)

    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            ((1, 0), 0),
            ((0, 1), 90),
            ((-1, 0), 180),
            ((0, -1), 270),
        ],
    )
    def test_angle_cardinal_directions(self, q, expected):
        result = Point2D(0, 0).angle(q, degrees=True)

        assert result.degrees == pytest.approx(expected)

    def test_angle_is_normalized_to_0_2pi(self):
        result = Point2D(0, 0).angle((0, -1))

        assert 0 <= result.radians < 2 * math.pi
        assert result.radians == pytest.approx(3 * math.pi / 2)

    def test_angle_defaults_to_radians(self):
        result = Point2D(0, 0).angle((1, 1))

        assert result.unit == "rad"
        assert result.radians == pytest.approx(math.pi / 4)

    def test_angle_degrees_option(self):
        result = Point2D(0, 0).angle((1, 1), degrees=True)

        assert result.unit == "deg"
        assert result.degrees == pytest.approx(45)

    def test_transform_rotation_about_origin(self):
        p = Point2D(1, 0)

        result = p.transform(angle=Angle.deg(90))

        assert result.is_close_to((0, 1))

    def test_transform_translation(self):
        p = Point2D(1, 2)

        result = p.transform(dx=10, dy=-5)

        assert result == Point2D(11, -3)

    def test_transform_rotation_about_pivot(self):
        p = Point2D(2, 1)

        result = p.transform(
            angle=Angle.deg(90),
            pivot=(1, 1),
        )

        assert result.is_close_to((1, 2))

    def test_transform_does_not_mutate_original(self):
        p = Point2D(1, 0)

        result = p.transform(angle=Angle.deg(90))

        assert p == Point2D(1, 0)
        assert result != p

    def test_transform_rotate_then_translate(self):
        p = Point2D(1, 0)

        result = p.transform(
            dx=10,
            dy=20,
            angle=Angle.deg(90),
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )

        assert result.is_close_to((10, 21))

    def test_transform_translate_then_rotate(self):
        p = Point2D(1, 0)

        result = p.transform(
            dx=10,
            dy=20,
            angle=Angle.deg(90),
            order=TransformationOrder.TRANSLATE_THEN_ROTATE,
        )

        assert result.is_close_to((-20, 11))


# ============================================================================
# Point3D
# ============================================================================


class TestPoint3D:
    def test_properties(self):
        p = Point3D(1, 2, 3)

        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_distance(self):
        assert Point3D(0, 0, 0).distance_to((1, 2, 2)) == pytest.approx(3.0)


# ============================================================================
# Origins
# ============================================================================


def test_origin_2d():
    assert ORIGIN_2D == Point2D(0, 0)


def test_origin_3d():
    assert ORIGIN_3D == Point3D(0, 0, 0)


# ============================================================================
# PointArrayND
# ============================================================================


class TestPointArrayND:
    def test_construct_from_nested_sequence(self):
        points = PointArrayND(
            [
                [1, 2],
                [3, 4],
            ]
        )

        assert len(points) == 2
        assert points.dim == 2
        assert points.dtype == np.float64

    def test_construct_from_numpy_array(self):
        array = np.array(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=np.float64,
        )

        points = PointArrayND(array)

        np.testing.assert_array_equal(points.coordinates, array)

    def test_coordinates_are_contiguous(self):
        points = PointArrayND([[1, 2], [3, 4]])

        assert points.coordinates.flags["C_CONTIGUOUS"]

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="2D array"):
            PointArrayND([1, 2, 3])

    def test_rejects_three_dimensional_input(self):
        with pytest.raises(ValueError, match="2D array"):
            PointArrayND(np.zeros((2, 3, 4)))

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match="at least one point"):
            PointArrayND(np.empty((0, 2)))

    def test_getitem_integer(self):
        points = PointArrayND([[1, 2], [3, 4]])

        np.testing.assert_array_equal(points[0], [1, 2])

    def test_getitem_slice(self):
        points = PointArrayND([[1, 2], [3, 4], [5, 6]])

        np.testing.assert_array_equal(
            points[1:],
            [[3, 4], [5, 6]],
        )

    def test_iteration(self):
        points = PointArrayND([[1, 2], [3, 4]])

        assert [row.tolist() for row in points] == [[1, 2], [3, 4]]

    def test_bounding_box(self):
        points = PointArrayND(
            [
                [3, 10],
                [-2, 5],
                [4, 7],
            ]
        )

        assert points.bounding_box() == [-2.0, 5.0, 4.0, 10.0]

    def test_to_list(self):
        points = PointArrayND([[1, 2], [3, 4]])

        assert points.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_copy_is_independent(self):
        points = PointArrayND([[1, 2], [3, 4]])
        copied = points.copy()

        copied.coordinates[0, 0] = 999

        assert points.coordinates[0, 0] == 1.0

    def test_repr(self):
        points = PointArrayND([[1, 2]])

        text = repr(points)

        assert "PointArrayND" in text
        assert "dim=2" in text
        assert "dtype=float64" in text

    def test_str(self):
        points = PointArrayND([[1, 2], [3, 4]])

        assert str(points) == "PointArrayND with 2 points in 2D"

    def test_from_dim_sequences(self):
        result = PointArrayND.from_dim_sequences(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

        np.testing.assert_array_equal(
            result.coordinates,
            [
                [1, 4],
                [2, 5],
                [3, 6],
            ],
        )

    def test_from_dim_sequences_with_names(self):
        result = PointArrayND.from_dim_sequences(
            [
                [1, 2],
                [3, 4],
            ],
            names=["x", "y"],
        )

        np.testing.assert_array_equal(
            result.coordinates,
            [
                [1, 3],
                [2, 4],
            ],
        )

    def test_from_dim_sequences_rejects_name_count_mismatch(self):
        with pytest.raises(ValueError, match="must have length 2"):
            PointArrayND.from_dim_sequences(
                [[1, 2], [3, 4]],
                names=["x"],
            )

    def test_from_dim_sequences_rejects_non_numeric_dimension(self):
        with pytest.raises(TypeError, match="All elements"):
            PointArrayND.from_dim_sequences(
                [[1, "x"], [3, 4]],
            )

    def test_from_dim_sequences_rejects_different_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            PointArrayND.from_dim_sequences(
                [[1, 2], [3]],
            )

    def test_from_named_dims(self):
        result = PointArrayND.from_named_dims(
            x=[1, 2],
            y=[3, 4],
            z=[5, 6],
        )

        np.testing.assert_array_equal(
            result.coordinates,
            [
                [1, 3, 5],
                [2, 4, 6],
            ],
        )

    def test_from_named_dims_requires_data(self):
        with pytest.raises(ValueError, match="empty"):
            PointArrayND.from_named_dims()


# ============================================================================
# PointArray2D
# ============================================================================


class TestPointArray2D:
    def test_requires_two_dimensions(self):
        with pytest.raises(ValueError, match="must have 2 dims"):
            PointArray2D(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            )

    def test_x_property(self):
        points = PointArray2D(
            [
                [1, 2],
                [3, 4],
            ]
        )

        np.testing.assert_array_equal(points.x, [1, 3])

    def test_y_property(self):
        points = PointArray2D(
            [
                [1, 2],
                [3, 4],
            ]
        )

        np.testing.assert_array_equal(points.y, [2, 4])

    def test_transform_translation(self):
        points = PointArray2D(
            [
                [0, 0],
                [1, 2],
                [-3, 4],
            ]
        )

        result = points.transform(dx=10, dy=20)

        np.testing.assert_allclose(
            result.coordinates,
            [
                [10, 20],
                [11, 22],
                [7, 24],
            ],
        )

    def test_transform_rotation(self):
        points = PointArray2D(
            [
                [1, 0],
                [0, 1],
            ]
        )

        result = points.transform(angle=Angle.deg(90))

        np.testing.assert_allclose(
            result.coordinates,
            [
                [0, 1],
                [-1, 0],
            ],
            atol=1e-15,
        )

    def test_transform_rotation_about_pivot(self):
        points = PointArray2D(
            [
                [2, 1],
                [1, 2],
            ]
        )

        result = points.transform(
            angle=Angle.deg(90),
            pivot=(1, 1),
        )

        np.testing.assert_allclose(
            result.coordinates,
            [
                [1, 2],
                [0, 1],
            ],
            atol=1e-15,
        )

    def test_transform_returns_new_object_by_default(self):
        points = PointArray2D([[1, 2]])

        result = points.transform(dx=10)

        assert isinstance(result, PointArray2D)
        assert result is not points

    def test_transform_does_not_mutate_by_default(self):
        points = PointArray2D([[1, 2]])

        points.transform(dx=10)

        np.testing.assert_array_equal(
            points.coordinates,
            [[1, 2]],
        )

    def test_transform_in_place_returns_none(self):
        points = PointArray2D([[1, 2]])

        result = points.transform(dx=10, in_place=True)

        assert result is None

    def test_transform_in_place_mutates(self):
        points = PointArray2D([[1, 2]])

        points.transform(dx=10, dy=20, in_place=True)

        np.testing.assert_array_equal(
            points.coordinates,
            [[11, 22]],
        )

    def test_transform_in_place_keeps_contiguous_storage(self):
        points = PointArray2D(
            [
                [1, 2],
                [3, 4],
            ]
        )

        points.transform(angle=Angle.deg(30), in_place=True)

        assert points.coordinates.flags["C_CONTIGUOUS"]

    def test_transform_accepts_point2d_pivot(self):
        points = PointArray2D([[2, 1]])

        result = points.transform(
            angle=Angle.deg(90),
            pivot=Point2D(1, 1),
        )

        np.testing.assert_allclose(
            result.coordinates,
            [[1, 2]],
            atol=1e-15,
        )

    def test_array_transform_matches_individual_point_transform(self):
        raw_points = [
            [1.0, 2.0],
            [-3.0, 4.0],
            [5.0, -2.0],
            [0.0, 0.0],
        ]

        points = PointArray2D(raw_points)

        kwargs = dict(
            dx=7.0,
            dy=-4.0,
            angle=Angle.deg(37),
            pivot=(2.0, -1.0),
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )

        transformed_array = points.transform(**kwargs)

        expected = np.array(
            [Point2D(*p).transform(**kwargs).coordinates for p in raw_points]
        )

        np.testing.assert_allclose(
            transformed_array.coordinates,
            expected,
        )

    def test_array_transform_translate_then_rotate_matches_points(self):
        raw_points = [
            [1.0, 2.0],
            [-3.0, 4.0],
            [5.0, -2.0],
        ]

        points = PointArray2D(raw_points)

        kwargs = dict(
            dx=7.0,
            dy=-4.0,
            angle=Angle.deg(37),
            pivot=(2.0, -1.0),
            order=TransformationOrder.TRANSLATE_THEN_ROTATE,
        )

        transformed_array = points.transform(**kwargs)

        expected = np.array(
            [Point2D(*p).transform(**kwargs).coordinates for p in raw_points]
        )

        np.testing.assert_allclose(
            transformed_array.coordinates,
            expected,
        )

    def test_array_rotation_preserves_distances_from_pivot(self):
        raw_points = [
            [1.0, 2.0],
            [-3.0, 4.0],
            [5.0, -2.0],
        ]

        pivot = np.array([2.0, -1.0])

        points = PointArray2D(raw_points)

        result = points.transform(
            angle=Angle.deg(73),
            pivot=pivot.tolist(),
        )

        before = np.linalg.norm(
            points.coordinates - pivot,
            axis=1,
        )

        after = np.linalg.norm(
            result.coordinates - pivot,
            axis=1,
        )

        np.testing.assert_allclose(after, before)

    def test_array_identity_transformation(self):
        raw = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

        points = PointArray2D(raw)

        result = points.transform()

        np.testing.assert_array_equal(result.coordinates, raw)
