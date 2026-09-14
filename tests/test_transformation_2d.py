import numpy as np
import pytest

from gbox.core.transformation import (
    rotation_matrix_2d,
    transformation_matrix_2d,
    transform_point_2d,
    translation_matrix_2d,
)
from gbox.core.utils import Angle, TransformationOrder


class TestRotationMatrix2D:
    def test_zero_rotation(self):
        matrix = rotation_matrix_2d(Angle.rad(0))

        np.testing.assert_allclose(
            matrix,
            np.eye(2),
        )

    def test_ninety_degree_rotation(self):
        matrix = rotation_matrix_2d(Angle.deg(90))

        np.testing.assert_allclose(
            matrix,
            [[0, -1], [1, 0]],
            atol=1e-12,
        )

    def test_180_degree_rotation(self):
        matrix = rotation_matrix_2d(Angle.deg(180))

        np.testing.assert_allclose(
            matrix,
            [[-1, 0], [0, -1]],
            atol=1e-12,
        )

    def test_270_degree_rotation(self):
        matrix = rotation_matrix_2d(Angle.deg(270))

        np.testing.assert_allclose(
            matrix,
            [[0, 1], [-1, 0]],
            atol=1e-12,
        )

    def test_rotation_matrix_is_orthogonal(self):
        matrix = rotation_matrix_2d(Angle.deg(37))

        np.testing.assert_allclose(
            matrix.T @ matrix,
            np.eye(2),
            atol=1e-12,
        )

    def test_rotation_matrix_has_determinant_one(self):
        matrix = rotation_matrix_2d(Angle.deg(37))

        assert np.linalg.det(matrix) == pytest.approx(1.0)


class TestTranslationMatrix2D:
    def test_identity_translation(self):
        matrix = translation_matrix_2d()

        np.testing.assert_allclose(matrix, np.eye(3))

    def test_translation(self):
        matrix = translation_matrix_2d(3, -4)

        np.testing.assert_allclose(
            matrix,
            [
                [1, 0, 3],
                [0, 1, -4],
                [0, 0, 1],
            ],
        )


class TestTransformationMatrix2D:
    def test_identity(self):
        matrix = transformation_matrix_2d()

        np.testing.assert_allclose(matrix, np.eye(3))

    def test_rotation_about_origin(self):
        matrix = transformation_matrix_2d(
            angle=Angle.deg(90),
        )

        np.testing.assert_allclose(
            matrix,
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            atol=1e-12,
        )

    def test_rotation_about_non_origin_pivot(self):
        matrix = transformation_matrix_2d(
            angle=Angle.deg(90),
            pivot=(1, 1),
        )

        result = matrix @ np.array([2, 1, 1])

        np.testing.assert_allclose(
            result[:2],
            [1, 2],
            atol=1e-12,
        )

    def test_point_at_pivot_is_unchanged_by_rotation(self):
        matrix = transformation_matrix_2d(
            angle=Angle.deg(123),
            pivot=(5, -7),
        )

        result = matrix @ np.array([5, -7, 1])

        np.testing.assert_allclose(result[:2], [5, -7])

    def test_rotate_then_translate(self):
        matrix = transformation_matrix_2d(
            angle=Angle.deg(90),
            dx=10,
            dy=20,
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )

        result = matrix @ np.array([1, 0, 1])

        np.testing.assert_allclose(
            result[:2],
            [10, 21],
            atol=1e-12,
        )

    def test_translate_then_rotate(self):
        matrix = transformation_matrix_2d(
            angle=Angle.deg(90),
            dx=10,
            dy=20,
            order=TransformationOrder.TRANSLATE_THEN_ROTATE,
        )

        result = matrix @ np.array([1, 0, 1])

        np.testing.assert_allclose(
            result[:2],
            [-20, 11],
            atol=1e-12,
        )

    def test_invalid_pivot_length(self):
        with pytest.raises(ValueError, match="exactly 2"):
            transformation_matrix_2d(pivot=(1, 2, 3))

    def test_invalid_pivot_short(self):
        with pytest.raises(ValueError, match="exactly 2"):
            transformation_matrix_2d(pivot=(1,))

    def test_invalid_order(self):
        with pytest.raises(TypeError, match="TransformationOrder"):
            transformation_matrix_2d(order="rotate_then_translate")


class TestTransformPoint2D:
    def test_identity(self):
        assert transform_point_2d(1, 2) == pytest.approx((1, 2))

    def test_translation(self):
        assert transform_point_2d(
            1,
            2,
            dx=3,
            dy=-4,
        ) == pytest.approx((4, -2))

    def test_rotation_about_origin(self):
        assert transform_point_2d(
            1,
            0,
            angle=Angle.deg(90),
        ) == pytest.approx((0, 1))

    def test_rotation_about_pivot(self):
        assert transform_point_2d(
            2,
            1,
            angle=Angle.deg(90),
            pivot=(1, 1),
        ) == pytest.approx((1, 2))

    def test_rotate_then_translate(self):
        assert transform_point_2d(
            1,
            0,
            angle=Angle.deg(90),
            dx=10,
            dy=20,
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        ) == pytest.approx((10, 21))

    def test_translate_then_rotate(self):
        assert transform_point_2d(
            1,
            0,
            angle=Angle.deg(90),
            dx=10,
            dy=20,
            order=TransformationOrder.TRANSLATE_THEN_ROTATE,
        ) == pytest.approx((-20, 11))

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (Angle.deg(0), (2, 3)),
            (Angle.deg(90), (-3, 2)),
            (Angle.deg(180), (-2, -3)),
            (Angle.deg(270), (3, -2)),
        ],
    )
    def test_cardinal_rotations(self, angle, expected):
        assert transform_point_2d(2, 3, angle=angle) == pytest.approx(expected)
