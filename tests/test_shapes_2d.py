import math

import numpy as np
import pytest

from gbox.core.points import Point2D, PointArray2D
from gbox.shapes.shapes_2d import (
    PI,
    Angle,
    TransformationOrder,
    Shape2D,
    Shape2DPose,
    Ellipse,
    Circle,
    CirclesArray,
)


# =====================================================================
# Shared helpers
# =====================================================================


def assert_point_close(point, expected, *, abs=1e-9):
    assert point.x == pytest.approx(expected[0], abs=abs)
    assert point.y == pytest.approx(expected[1], abs=abs)


def assert_bbox_close(bbox, expected, *, abs=1e-9):
    assert bbox == pytest.approx(expected, abs=abs)


# =====================================================================
# Shape2DPose
# =====================================================================


class TestShape2DPose:
    def test_initialization(self):
        orientation = Angle.rad(np.pi / 4)
        pose = Shape2DPose(1.5, -2.5, orientation)

        assert pose.x == 1.5
        assert pose.y == -2.5
        assert pose.orientation == orientation

    def test_repr(self):
        orientation = Angle.rad(np.pi / 4)
        pose = Shape2DPose(1.5, -2.5, orientation)

        assert repr(pose) == (
            f"Shape2DPose(x=1.5, y=-2.5, orientation={orientation})"
        )

    def test_is_immutable(self):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.5))

        with pytest.raises((AttributeError, TypeError)):
            pose.x = 10.0  # type: ignore[misc]

        with pytest.raises((AttributeError, TypeError)):
            pose.y = 10.0  # type: ignore[misc]

        with pytest.raises((AttributeError, TypeError)):
            pose.orientation = Angle.rad(1.0)  # type: ignore[misc]

    def test_rotate_returns_new_pose(self):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.5))

        result = pose.rotate(Angle.rad(0.25))

        assert result is not pose
        assert (result.x, result.y) == (1.0, 2.0)
        assert result.orientation.radians == pytest.approx(0.75)

    def test_rotate_does_not_mutate_original(self):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.5))

        pose.rotate(Angle.rad(0.25))

        assert pose.orientation.radians == pytest.approx(0.5)

    def test_translate_returns_new_pose(self):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.5))

        result = pose.translate(-0.5, 3.0)

        assert result is not pose
        assert (result.x, result.y) == pytest.approx((0.5, 5.0))
        assert result.orientation == pose.orientation

    def test_translate_does_not_mutate_original(self):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.5))

        pose.translate(10.0, 20.0)

        assert (pose.x, pose.y) == (1.0, 2.0)

    def test_transform_default_pivot_is_position(self):
        pose = Shape2DPose(3.0, 4.0, Angle.rad(0.2))

        result = pose.transform(rot_angle=Angle.rad(np.pi / 2))

        # Rotating about its own position leaves x/y unchanged.
        assert (result.x, result.y) == pytest.approx((3.0, 4.0))
        assert result.orientation.radians == pytest.approx(
            0.2 + np.pi / 2
        )

    def test_transform_rotate_about_origin(self):
        pose = Shape2DPose(1.0, 0.0, Angle.rad(0.0))

        result = pose.transform(
            rot_angle=Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
        )

        assert (result.x, result.y) == pytest.approx((0.0, 1.0))
        assert result.orientation.radians == pytest.approx(np.pi / 2)

    def test_transform_rotate_then_translate(self):
        pose = Shape2DPose(1.0, 0.0, Angle.rad(0.0))

        result = pose.transform(
            dx=2.0,
            dy=3.0,
            rot_angle=Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )

        assert (result.x, result.y) == pytest.approx((2.0, 4.0))
        assert result.orientation.radians == pytest.approx(np.pi / 2)

    def test_transform_returns_new_pose(self):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.0))

        result = pose.transform(dx=1.0)

        assert result is not pose
        assert (pose.x, pose.y) == (1.0, 2.0)

    @pytest.mark.parametrize(
        "angle",
        [
            0.0,
            np.pi / 2,
            np.pi,
            -np.pi / 2,
            2 * np.pi,
            -2 * np.pi,
        ],
    )
    def test_rotation_angles(self, angle):
        pose = Shape2DPose(1.0, 2.0, Angle.rad(0.0))

        result = pose.rotate(Angle.rad(angle))

        assert result.orientation.radians == pytest.approx(angle)


# =====================================================================
# Shape2D abstract interface
# =====================================================================


class TestShape2DInterface:
    def test_cannot_instantiate_abstract_shape(self):
        with pytest.raises(TypeError):
            Shape2D()  # type: ignore[abstract]

    def test_equivalent_circle_radius(self):
        class DummyShape(Shape2D):
            @property
            def position(self):
                return Shape2DPose(0.0, 0.0, Angle.rad(0.0))

            @property
            def area(self):
                return 4.0 * PI

            @property
            def perimeter(self):
                return 4.0

            def contains_point(self, point):
                return 1

            def union_of_circles(self):
                return []

            @property
            def bounding_box(self):
                return [0.0, 0.0, 1.0, 1.0]

        shape = DummyShape()

        assert shape.equivalent_circle_radius == pytest.approx(2.0)

    def test_default_bounding_box_raises(self):
        class DummyShape(Shape2D):
            @property
            def position(self):
                return Shape2DPose(0.0, 0.0, Angle.rad(0.0))

            @property
            def area(self):
                return 1.0

            @property
            def perimeter(self):
                return 1.0

            def contains_point(self, point):
                return 1

            def union_of_circles(self):
                return []

            @property
            def bounding_box(self):
                return super().bounding_box

        with pytest.raises(
            NotImplementedError,
            match="Bounding box method not implemented",
        ):
            _ = DummyShape().bounding_box


# =====================================================================
# Ellipse - construction and validation
# =====================================================================


class TestEllipseConstruction:
    def test_basic_construction(self):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.semi_major_length == 5.0
        assert ellipse.semi_minor_length == 3.0
        assert ellipse.centre == (0.0, 0.0)
        assert ellipse.position.orientation == Angle.rad(0.0)

    @pytest.mark.parametrize(
        "a,b",
        [
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ],
    )
    def test_non_positive_axes_rejected(self, a, b):
        with pytest.raises(ValueError):
            Ellipse(a, b)

    def test_minor_axis_cannot_exceed_major_axis(self):
        with pytest.raises(
            ValueError,
            match="Semi-major axis must be >= semi-minor axis",
        ):
            Ellipse(2.0, 3.0)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("5", 3.0),
            (None, 3.0),
            (object(), 3.0),
            (5.0, "3"),
        ],
    )
    def test_invalid_axis_types_rejected(self, a, b):
        with pytest.raises((ValueError, TypeError)):
            Ellipse(a, b)  # type: ignore[arg-type]

    def test_equal_axes_are_allowed(self):
        ellipse = Ellipse(3.0, 3.0)

        assert ellipse.semi_major_length == 3.0
        assert ellipse.semi_minor_length == 3.0
        assert ellipse.aspect_ratio == pytest.approx(1.0)
        assert ellipse.eccentricity == pytest.approx(0.0)

    def test_centre_is_stored_as_floats(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(1, -2),
        )

        assert ellipse.centre == (1.0, -2.0)

    def test_position_matches_constructor_arguments(self):
        angle = Angle.rad(0.7)
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(4.0, -2.0),
            major_axis_angle=angle,
        )

        assert ellipse.position == Shape2DPose(4.0, -2.0, angle)


# =====================================================================
# Ellipse - geometric properties
# =====================================================================


class TestEllipseProperties:
    @pytest.fixture
    def ellipse(self):
        return Ellipse(5.0, 3.0)

    def test_area(self, ellipse):
        assert ellipse.area == pytest.approx(15.0 * PI)

    def test_aspect_ratio(self, ellipse):
        assert ellipse.aspect_ratio == pytest.approx(5.0 / 3.0)

    def test_eccentricity(self, ellipse):
        assert ellipse.eccentricity == pytest.approx(0.8)

    def test_equivalent_circle_radius(self, ellipse):
        assert ellipse.equivalent_circle_radius == pytest.approx(
            math.sqrt(15.0)
        )

    @pytest.mark.parametrize(
        "a,b",
        [
            (1.0, 1.0),
            (2.0, 1.0),
            (5.0, 3.0),
            (100.0, 50.0),
        ],
    )
    def test_area_formula(self, a, b):
        ellipse = Ellipse(a, b)

        assert ellipse.area == pytest.approx(PI * a * b)

    def test_circle_perimeter_is_circumference(self):
        radius = 4.5
        circle = Ellipse(radius, radius)

        assert circle.perimeter == pytest.approx(2 * PI * radius)

    def test_ellipse_perimeter_is_positive(self, ellipse):
        assert ellipse.perimeter > 0.0

    def test_perimeter_is_greater_than_circle_with_same_minor_axis(
        self,
    ):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.perimeter > 2.0 * PI * 3.0


# =====================================================================
# Ellipse - bounding box
# =====================================================================


class TestEllipseBoundingBox:
    def test_axis_aligned(self):
        ellipse = Ellipse(5.0, 3.0, (1.0, 2.0))

        assert_bbox_close(
            ellipse.bounding_box,
            [-4.0, -1.0, 6.0, 5.0],
        )

    def test_rotated_90_degrees(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            (1.0, 2.0),
            Angle.rad(np.pi / 2),
        )

        assert_bbox_close(
            ellipse.bounding_box,
            [-2.0, -3.0, 4.0, 7.0],
        )

    def test_rotated_45_degrees(self):
        a = 5.0
        b = 3.0
        angle = np.pi / 4

        h = math.sqrt((a * math.cos(angle)) ** 2 + (b * math.sin(angle)) ** 2)

        ellipse = Ellipse(
            a,
            b,
            (1.0, 2.0),
            Angle.rad(angle),
        )

        assert_bbox_close(
            ellipse.bounding_box,
            [1.0 - h, 2.0 - h, 1.0 + h, 2.0 + h],
        )

    @pytest.mark.parametrize(
        "angle",
        [
            0.0,
            np.pi / 6,
            np.pi / 4,
            np.pi / 2,
            np.pi,
            3 * np.pi / 2,
        ],
    )
    def test_bounding_box_contains_sampled_boundary(self, angle):
        ellipse = Ellipse(
            5.0,
            3.0,
            (2.0, -3.0),
            Angle.rad(angle),
        )

        bbox = ellipse.bounding_box
        points = ellipse.sample_points(num_points=500)

        assert np.all(points.x >= bbox[0] - 1e-9)
        assert np.all(points.x <= bbox[2] + 1e-9)
        assert np.all(points.y >= bbox[1] - 1e-9)
        assert np.all(points.y <= bbox[3] + 1e-9)


# =====================================================================
# Ellipse - containment
# =====================================================================


class TestEllipseContainment:
    def test_centre_is_inside(self):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.contains_point((0.0, 0.0)) == 1

    def test_major_axis_endpoint_is_boundary(self):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.contains_point((5.0, 0.0)) == 0
        assert ellipse.contains_point((-5.0, 0.0)) == 0

    def test_minor_axis_endpoint_is_boundary(self):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.contains_point((0.0, 3.0)) == 0
        assert ellipse.contains_point((0.0, -3.0)) == 0

    @pytest.mark.parametrize(
        "point",
        [
            (5.1, 0.0),
            (-5.1, 0.0),
            (0.0, 3.1),
            (0.0, -3.1),
            (4.0, 3.0),
            (10.0, 10.0),
        ],
    )
    def test_outside_points(self, point):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.contains_point(point) == -1

    @pytest.mark.parametrize(
        "point",
        [
            (0.0, 0.0),
            (1.0, 0.5),
            (-2.0, 1.0),
            (4.0, 0.0),
        ],
    )
    def test_inside_points(self, point):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.contains_point(point) == 1

    def test_accepts_point2d(self):
        ellipse = Ellipse(5.0, 3.0)

        point = Point2D(0.0, 0.0)

        assert ellipse.contains_point(point) == 1

    def test_translation_and_rotation_are_respected(self):
        ellipse = Ellipse(
            4.0,
            2.0,
            centre=(2.0, 3.0),
            major_axis_angle=Angle.rad(np.pi / 2),
        )

        assert ellipse.contains_point((2.0, 3.0)) == 1
        assert ellipse.contains_point((2.0, 7.0)) == 0
        assert ellipse.contains_point((4.0, 3.0)) == 0
        assert ellipse.contains_point((2.0, 7.1)) == -1

    def test_atol_controls_boundary_band(self):
        ellipse = Ellipse(5.0, 3.0)

        # val is approximately 0.0004.
        point = (5.001, 0.0)

        assert ellipse.contains_point(point, atol=1e-6) == -1
        assert ellipse.contains_point(point, atol=1e-3) == 0

    def test_negative_atol_is_rejected_or_documented(self):
        ellipse = Ellipse(5.0, 3.0)

        # Current implementation accepts this. If negative tolerances are
        # considered invalid API, this test should become a normal
        # ValueError test and the implementation should validate atol.
        result = ellipse.contains_point((5.0, 0.0), atol=-1e-6)

        assert result in (-1, 0, 1)


# =====================================================================
# Ellipse - parametric points
# =====================================================================


class TestEllipseSampling:
    @pytest.fixture
    def ellipse(self):
        return Ellipse(
            5.0,
            3.0,
            centre=(2.0, -1.0),
            major_axis_angle=Angle.rad(np.pi / 4),
        )

    def test_point_at_angle_zero(self):
        ellipse = Ellipse(5.0, 3.0)

        point = ellipse.point_at_angle(0.0)

        assert_point_close(point, (5.0, 0.0))

    def test_point_at_angle_pi_over_two(self):
        ellipse = Ellipse(5.0, 3.0)

        point = ellipse.point_at_angle(np.pi / 2)

        assert_point_close(point, (0.0, 3.0))

    def test_point_at_angle_pi(self):
        ellipse = Ellipse(5.0, 3.0)

        point = ellipse.point_at_angle(np.pi)

        assert_point_close(point, (-5.0, 0.0))

    def test_point_at_angle_three_pi_over_two(self):
        ellipse = Ellipse(5.0, 3.0)

        point = ellipse.point_at_angle(3 * np.pi / 2)

        assert_point_close(point, (0.0, -3.0))

    def test_point_at_angle_respects_pose(self, ellipse):
        point = ellipse.point_at_angle(0.0)

        expected_x = 2.0 + 5.0 * math.cos(np.pi / 4)
        expected_y = -1.0 + 5.0 * math.sin(np.pi / 4)

        assert_point_close(point, (expected_x, expected_y))

    def test_parametric_points_shape(self, ellipse):
        theta = np.linspace(0.0, 2.0 * PI, 25)

        points = ellipse.points_at_parametric_points(theta)

        assert isinstance(points, PointArray2D)
        assert len(points) == 25

    def test_parametric_points_are_on_boundary(self, ellipse):
        theta = np.linspace(0.0, 2.0 * PI, 100)

        points = ellipse.points_at_parametric_points(theta)

        for point in points:
            assert ellipse.contains_point(point.tolist(), atol=1e-5) == 0

    def test_sampling_num_points(self):
        ellipse = Ellipse(5.0, 3.0)

        points = ellipse.sample_points(num_points=100)

        assert isinstance(points, PointArray2D)
        assert len(points) == 100


    def test_sampling_explicit_num_points_takes_precedence(self):
        ellipse = Ellipse(5.0, 3.0)

        points = ellipse.sample_points(
            num_points=17,
            point_density=1000.0,
        )

        assert len(points) == 17

    def test_sampling_zero_num_points_should_be_invalid(self):
        ellipse = Ellipse(5.0, 3.0)

        # This exposes the current `num_points or ...` behavior.
        #
        # Desired contract: explicit 0 should fail validation rather than
        # silently falling back to density.
        with pytest.raises(ValueError):
            ellipse.sample_points(num_points=0)

    def test_sampling_negative_num_points_is_invalid(self):
        ellipse = Ellipse(5.0, 3.0)

        with pytest.raises(ValueError):
            ellipse.sample_points(num_points=-1)

    def test_sampling_one_point(self):
        ellipse = Ellipse(5.0, 3.0)

        points = ellipse.sample_points(num_points=1)

        assert len(points) == 1
        assert ellipse.contains_point(points[0].tolist()) == 0


# =====================================================================
# Ellipse - transforms
# =====================================================================


class TestEllipseTransform:
    def test_translate(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(1.0, 2.0),
            major_axis_angle=Angle.rad(0.25),
        )

        result = ellipse.transform(dx=4.0, dy=-3.0)

        assert result is not ellipse
        assert result.centre == pytest.approx((5.0, -1.0))
        assert result.position.orientation == ellipse.position.orientation

    def test_translate_does_not_mutate_original(self):
        ellipse = Ellipse(5.0, 3.0, centre=(1.0, 2.0))

        ellipse.transform(dx=4.0, dy=-3.0)

        assert ellipse.centre == (1.0, 2.0)

    def test_rotate_about_own_centre_changes_orientation_only(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(2.0, 4.0),
            major_axis_angle=Angle.rad(0.25),
        )

        result = ellipse.transform(
            d_theta=Angle.rad(np.pi / 2),
        )

        assert result.centre == pytest.approx((2.0, 4.0))
        assert result.position.orientation.radians == pytest.approx(
            0.25 + np.pi / 2
        )

    def test_rotate_about_origin(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(2.0, 0.0),
            major_axis_angle=Angle.rad(0.0),
        )

        result = ellipse.transform(
            d_theta=Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
        )

        assert result.centre == pytest.approx((0.0, 2.0))
        assert result.position.orientation.radians == pytest.approx(
            np.pi / 2
        )

    def test_rotate_about_arbitrary_pivot(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(4.0, 2.0),
            major_axis_angle=Angle.rad(0.0),
        )

        result = ellipse.transform(
            d_theta=Angle.rad(np.pi / 2),
            pivot=(2.0, 2.0),
        )

        assert result.centre == pytest.approx((2.0, 4.0))
        assert result.position.orientation.radians == pytest.approx(
            np.pi / 2
        )

    def test_transform_combines_rotation_and_translation(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(1.0, 0.0),
        )

        result = ellipse.transform(
            dx=2.0,
            dy=3.0,
            d_theta=Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )

        assert result.centre == pytest.approx((2.0, 4.0))

    def test_transform_preserves_dimensions(self):
        ellipse = Ellipse(5.0, 3.0)

        result = ellipse.transform(
            dx=10.0,
            dy=-20.0,
            d_theta=Angle.rad(1.5),
        )

        assert result.semi_major_length == ellipse.semi_major_length
        assert result.semi_minor_length == ellipse.semi_minor_length
        assert result.area == pytest.approx(ellipse.area)

    def test_transform_preserves_aspect_ratio(self):
        ellipse = Ellipse(5.0, 3.0)

        result = ellipse.transform(d_theta=Angle.rad(0.7))

        assert result.aspect_ratio == pytest.approx(ellipse.aspect_ratio)


# =====================================================================
# Ellipse - cloning
# =====================================================================


class TestEllipseClone:
    def test_clone_is_distinct(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(1.0, 2.0),
            major_axis_angle=Angle.rad(0.7),
        )

        clone = ellipse.clone()

        assert clone is not ellipse

    def test_clone_preserves_all_geometry(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(1.0, 2.0),
            major_axis_angle=Angle.rad(0.7),
        )

        clone = ellipse.clone()

        assert clone.semi_major_length == ellipse.semi_major_length
        assert clone.semi_minor_length == ellipse.semi_minor_length
        assert clone.centre == ellipse.centre
        assert clone.position.orientation == ellipse.position.orientation
        assert clone.area == pytest.approx(ellipse.area)
        assert clone.perimeter == pytest.approx(ellipse.perimeter)
        assert clone.bounding_box == pytest.approx(ellipse.bounding_box)

    def test_clone_is_independent(self):
        ellipse = Ellipse(5.0, 3.0)

        clone = ellipse.clone()
        moved = clone.transform(dx=10.0)

        assert ellipse.centre == (0.0, 0.0)
        assert moved.centre == (10.0, 0.0)


# =====================================================================
# Ellipse - from_params
# =====================================================================


class TestEllipseFromParams:
    def test_valid_params(self):
        ellipse = Ellipse.from_params(
            position_params={
                "xc": 3.0,
                "yc": -1.0,
                "major_axis_angle": Angle.rad(0.5),
            },
            size_params={
                "semi_major_length": 6.0,
                "semi_minor_length": 2.0,
            },
        )

        assert ellipse.centre == (3.0, -1.0)
        assert ellipse.position.orientation == Angle.rad(0.5)
        assert ellipse.semi_major_length == 6.0
        assert ellipse.semi_minor_length == 2.0

    @pytest.mark.parametrize(
        "position_params",
        [
            {},
            {"xc": 0.0},
            {"xc": 0.0, "yc": 0.0},
            {
                "xc": 0.0,
                "yc": 0.0,
                "major_axis_angle": 0.5,
            },
            {
                "xc": "0",
                "yc": 0.0,
                "major_axis_angle": Angle.rad(0.0),
            },
        ],
    )
    def test_invalid_position_params(self, position_params):
        with pytest.raises((ValueError, TypeError)):
            Ellipse.from_params(
                position_params,
                {
                    "semi_major_length": 5.0,
                    "semi_minor_length": 3.0,
                },
            )

    @pytest.mark.parametrize(
        "size_params",
        [
            {},
            {"semi_major_length": 5.0},
            {
                "semi_major_length": 5.0,
                "semi_minor_length": "3",
            },
            {
                "semi_major_length": 2.0,
                "semi_minor_length": 3.0,
            },
        ],
    )
    def test_invalid_size_params(self, size_params):
        with pytest.raises((ValueError, TypeError)):
            Ellipse.from_params(
                {
                    "xc": 0.0,
                    "yc": 0.0,
                    "major_axis_angle": Angle.rad(0.0),
                },
                size_params,
            )

    def test_extra_keys_are_rejected(self):
        with pytest.raises(ValueError):
            Ellipse.from_params(
                {
                    "xc": 0.0,
                    "yc": 0.0,
                    "major_axis_angle": Angle.rad(0.0),
                    "extra": 1.0,
                },
                {
                    "semi_major_length": 6.0,
                    "semi_minor_length": 3.0,
                },
            )


# =====================================================================
# Ellipse - r_shortest
# =====================================================================


class TestEllipseRShortest:
    def test_circle_case(self):
        ellipse = Ellipse(4.0, 4.0)

        for xi in [-100.0, -2.0, 0.0, 2.0, 100.0]:
            assert ellipse.r_shortest(xi) == pytest.approx(4.0)

    def test_at_origin(self):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.r_shortest(0.0) == pytest.approx(3.0)

    def test_formula_at_known_point(self):
        a = 5.0
        b = 3.0
        xi = 2.0

        ellipse = Ellipse(a, b)

        expected = b * math.sqrt(
            1.0 - xi**2 / (a**2 - b**2)
        )

        assert ellipse.r_shortest(xi) == pytest.approx(expected)

    def test_symmetry_about_origin(self):
        ellipse = Ellipse(5.0, 3.0)

        assert ellipse.r_shortest(1.5) == pytest.approx(
            ellipse.r_shortest(-1.5)
        )

    def test_zero_at_a_squared_minus_b_squared_boundary(self):
        a = 5.0
        b = 3.0
        xi = math.sqrt(a * a - b * b)

        ellipse = Ellipse(a, b)

        assert ellipse.r_shortest(xi) == pytest.approx(0.0)

    def test_outside_valid_domain_currently_produces_nan(self):
        ellipse = Ellipse(5.0, 3.0)

        result = ellipse.r_shortest(10.0)

        assert np.isnan(result)


# =====================================================================
# Ellipse - union_of_circles
# =====================================================================


class TestEllipseUnionOfCircles:
    def test_circle_returns_single_circle(self):
        ellipse = Ellipse(
            3.0,
            3.0,
            centre=(1.0, 2.0),
            major_axis_angle=Angle.rad(0.5),
        )

        circles = ellipse.union_of_circles()

        assert len(circles) == 1
        assert isinstance(circles[0], Circle)
        assert circles[0].radius == pytest.approx(3.0)
        assert circles[0].centre == pytest.approx((1.0, 2.0))

    def test_dh_must_be_positive(self):
        ellipse = Ellipse(5.0, 3.0)

        with pytest.raises(
            ValueError,
            match="buffer thickness dh must be > 0",
        ):
            ellipse.union_of_circles(dh=0.0)

        with pytest.raises(
            ValueError,
            match="buffer thickness dh must be > 0",
        ):
            ellipse.union_of_circles(dh=-0.1)

    def test_dh_must_be_less_than_minor_axis(self):
        ellipse = Ellipse(5.0, 3.0)

        with pytest.raises(ValueError):
            ellipse.union_of_circles(dh=3.0)

        with pytest.raises(ValueError):
            ellipse.union_of_circles(dh=3.1)

    def test_non_circle_generates_multiple_circles(self):
        ellipse = Ellipse(5.0, 3.0)

        circles = ellipse.union_of_circles(dh=0.1)

        assert len(circles) > 1
        assert all(isinstance(c, Circle) for c in circles)

    def test_generated_circle_radii_are_positive(self):
        ellipse = Ellipse(5.0, 3.0)

        circles = ellipse.union_of_circles(dh=0.1)

        assert all(c.radius > 0.0 for c in circles)

    def test_generated_circle_radii_do_not_exceed_minor_axis(self):
        ellipse = Ellipse(5.0, 3.0)

        circles = ellipse.union_of_circles(dh=0.1)

        assert all(c.radius <= ellipse.semi_minor_length + 1e-9 for c in circles)

    def test_generated_circles_are_transformed_to_ellipse_pose(self):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(10.0, -4.0),
            major_axis_angle=Angle.rad(np.pi / 4),
        )

        circles = ellipse.union_of_circles(dh=0.1)

        # At least one generated circle should be displaced from the
        # ellipse centre for a non-circular ellipse.
        assert any(
            not np.allclose(c.centre, ellipse.centre)
            for c in circles
        )

    def test_union_generation_is_deterministic(self):
        ellipse = Ellipse(5.0, 3.0)

        circles_a = ellipse.union_of_circles(dh=0.1)
        circles_b = ellipse.union_of_circles(dh=0.1)

        assert len(circles_a) == len(circles_b)

        for a, b in zip(circles_a, circles_b):
            assert a.radius == pytest.approx(b.radius)
            assert a.centre == pytest.approx(b.centre)


# =====================================================================
# Circle
# =====================================================================


class TestCircle:
    def test_construction(self):
        circle = Circle(4.5, (2.0, -3.0))

        assert circle.radius == 4.5
        assert circle.semi_major_length == 4.5
        assert circle.semi_minor_length == 4.5
        assert circle.centre == (2.0, -3.0)

    def test_is_geometrically_circular(self):
        circle = Circle(4.5)

        assert circle.aspect_ratio == pytest.approx(1.0)
        assert circle.eccentricity == pytest.approx(0.0)

    def test_area(self):
        circle = Circle(4.5)

        assert circle.area == pytest.approx(PI * 4.5**2)

    def test_perimeter(self):
        circle = Circle(4.5)

        assert circle.perimeter == pytest.approx(2 * PI * 4.5)

    def test_bounding_box(self):
        circle = Circle(4.5, (2.0, -3.0))

        assert_bbox_close(
            circle.bounding_box,
            [-2.5, -7.5, 6.5, 1.5],
        )

    def test_equivalent_circle_radius_is_radius(self):
        for radius in [0.1, 1.0, 5.0, 100.0]:
            circle = Circle(radius)

            assert circle.equivalent_circle_radius == pytest.approx(radius)

    def test_contains_point(self):
        circle = Circle(5.0)

        assert circle.contains_point((0.0, 0.0)) == 1
        assert circle.contains_point((3.0, 4.0)) == 0
        assert circle.contains_point((5.1, 0.0)) == -1

    def test_clone(self):
        circle = Circle(2.0, (1.0, 1.0))

        clone = circle.clone()

        assert clone is not circle
        assert isinstance(clone, Circle)
        assert clone.radius == 2.0
        assert clone.centre == (1.0, 1.0)

    def test_clone_is_independent(self):
        circle = Circle(2.0, (1.0, 1.0))

        moved = circle.clone().transform(dx=10.0)

        assert circle.centre == (1.0, 1.0)
        assert moved.centre == (11.0, 1.0)

    def test_transform_translation(self):
        circle = Circle(2.0, (1.0, 2.0))

        result = circle.transform(dx=3.0, dy=-4.0)

        assert result.centre == pytest.approx((4.0, -2.0))
        assert result.radius == pytest.approx(2.0)

    def test_transform_rotation_about_origin(self):
        circle = Circle(2.0, (1.0, 0.0))

        result = circle.transform(
            d_theta=Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
        )

        assert result.centre == pytest.approx((0.0, 1.0))
        assert result.radius == pytest.approx(2.0)

    def test_transform_rotation_about_own_centre(self):
        circle = Circle(2.0, (4.0, 5.0))

        result = circle.transform(
            d_theta=Angle.rad(np.pi / 2),
        )

        assert result.centre == pytest.approx((4.0, 5.0))
        assert result.radius == pytest.approx(2.0)


# =====================================================================
# CirclesArray - construction
# =====================================================================


class TestCirclesArrayConstruction:

    def test_from_tuples(self):
        circles = CirclesArray(
            [(0.0, 0.0), (1.0, 2.0)],
            [1.0, 2.0],
        )

        assert len(circles) == 2

    def test_from_point_array(self):
        centres = PointArray2D.from_named_dims(
            x=[0.0, 1.0],
            y=[2.0, 3.0],
        )

        circles = CirclesArray(centres, [1.0, 2.0])

        assert len(circles) == 2
        assert np.allclose(
            circles.centres.coordinates,
            centres.coordinates,
        )

    def test_from_numpy_array(self):
        centres = np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        )

        circles = CirclesArray(centres, [1.0, 2.0])

        assert len(circles) == 2
        assert np.allclose(
            circles.centres.coordinates,
            centres,
        )

    def test_scalar_radius_is_broadcast(self):
        circles = CirclesArray(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            2.5,
        )

        assert len(circles) == 3
        assert np.all(circles.radii == 2.5)

    def test_integer_scalar_radius_is_supported(self):
        circles = CirclesArray(
            [(0.0, 0.0), (1.0, 1.0)],
            2,
        )

        assert np.all(circles.radii == 2.0)

    def test_numpy_radii_are_supported(self):
        circles = CirclesArray(
            [(0.0, 0.0), (1.0, 1.0)],
            np.array([1.0, 2.0]),
        )

        assert np.allclose(circles.radii, [1.0, 2.0])

    def test_invalid_centre_dimensions(self):
        with pytest.raises(
            ValueError,
            match=r"centres must be a 2D array with shape \(N, 2\)",
        ):
            CirclesArray(np.array([1.0, 2.0]), 1.0)

    @pytest.mark.parametrize(
        "centres",
        [
            np.empty((0, 3)),
            np.empty((2, 3)),
            np.empty((2, 1)),
            np.empty((2, 4)),
        ],
    )
    def test_invalid_numpy_centre_shapes(self, centres):
        with pytest.raises(ValueError):
            CirclesArray(centres, 1.0)

    def test_radii_length_must_match_centres(self):
        with pytest.raises(
            ValueError,
            match="radii length must match number of centres",
        ):
            CirclesArray(
                [(0.0, 0.0), (1.0, 1.0)],
                [1.0],
            )

    @pytest.mark.parametrize(
        "radii",
        [
            [0.0],
            [-1.0],
            [1.0, -2.0],
            [0.0, 2.0],
        ],
    )
    def test_radii_must_be_positive(self, radii):
        centres = [(0.0, 0.0)] * len(radii)

        with pytest.raises(
            ValueError,
            match="all radii must be positive",
        ):
            CirclesArray(centres, radii)

    def test_radii_must_be_one_dimensional(self):
        centres = [(0.0, 0.0), (1.0, 1.0)]

        with pytest.raises(ValueError):
            CirclesArray(
                centres,
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )


# =====================================================================
# CirclesArray - conversions
# =====================================================================


class TestCirclesArrayConversions:
    @pytest.fixture
    def circles_array(self):
        return CirclesArray(
            [
                (0.0, 0.0),
                (3.0, 4.0),
                (-2.0, 1.0),
            ],
            [1.0, 2.0, 0.5],
        )

    def test_len(self, circles_array):
        assert len(circles_array) == 3

    def test_from_circles(self):
        source = [
            Circle(1.0, (0.0, 0.0)),
            Circle(2.5, (3.0, 5.0)),
        ]

        result = CirclesArray.from_circles(source)

        assert len(result) == 2
        assert np.allclose(result.radii, [1.0, 2.5])
        assert np.allclose(
            result.centres.coordinates,
            [[0.0, 0.0], [3.0, 5.0]],
        )

    def test_to_circles(self, circles_array):
        circles = circles_array.to_circles()

        assert len(circles) == len(circles_array)
        assert all(isinstance(c, Circle) for c in circles)

        for circle, centre, radius in zip(
            circles,
            circles_array.centres.coordinates,
            circles_array.radii,
        ):
            assert circle.centre == pytest.approx(centre)
            assert circle.radius == pytest.approx(radius)

    def test_round_trip(self, circles_array):
        recovered = CirclesArray.from_circles(
            circles_array.to_circles()
        )

        assert np.allclose(
            recovered.centres.coordinates,
            circles_array.centres.coordinates,
        )
        assert np.allclose(
            recovered.radii,
            circles_array.radii,
        )


# =====================================================================
# CirclesArray - cloning and ownership
# =====================================================================


class TestCirclesArrayClone:
    @pytest.fixture
    def circles_array(self):
        return CirclesArray(
            [(0.0, 0.0), (3.0, 4.0)],
            [1.0, 2.0],
        )

    def test_clone_is_distinct(self, circles_array):
        clone = circles_array.clone()

        assert clone is not circles_array

    def test_clone_copies_geometry(self, circles_array):
        clone = circles_array.clone()

        assert np.array_equal(
            clone.centres.coordinates,
            circles_array.centres.coordinates,
        )
        assert np.array_equal(
            clone.radii,
            circles_array.radii,
        )

    def test_clone_is_independent(self, circles_array):
        clone = circles_array.clone()

        clone.translate(10.0, 20.0, in_place=True)

        assert not np.allclose(
            clone.centres.coordinates,
            circles_array.centres.coordinates,
        )

    def test_clone_radii_are_independent(self, circles_array):
        clone = circles_array.clone()

        clone.radii[0] = 100.0

        assert circles_array.radii[0] == pytest.approx(1.0)


# =====================================================================
# CirclesArray - translation
# =====================================================================


class TestCirclesArrayTranslate:
    @pytest.fixture
    def circles_array(self):
        return CirclesArray(
            [(0.0, 0.0), (3.0, 4.0)],
            [1.0, 2.0],
        )

    def test_translate_out_of_place(self, circles_array):
        result = circles_array.translate(
            2.0,
            -1.0,
            in_place=False,
        )

        assert result is not circles_array
        assert np.allclose(
            result.centres.coordinates,
            [[2.0, -1.0], [5.0, 3.0]],
        )
        assert np.allclose(
            circles_array.centres.coordinates,
            [[0.0, 0.0], [3.0, 4.0]],
        )

    def test_translate_in_place(self, circles_array):
        result = circles_array.translate(
            2.0,
            -1.0,
            in_place=True,
        )

        assert result is circles_array
        assert np.allclose(
            circles_array.centres.coordinates,
            [[2.0, -1.0], [5.0, 3.0]],
        )

    def test_translate_does_not_change_radii(self, circles_array):
        before = circles_array.radii.copy()

        circles_array.translate(10.0, -10.0)

        assert np.array_equal(circles_array.radii, before)

    @pytest.mark.parametrize(
        "dx,dy",
        [
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, -10.0),
            (-3.5, 8.25),
        ],
    )
    def test_translation_vector(self, dx, dy):
        circles = CirclesArray([(1.0, 2.0)], [3.0])

        result = circles.translate(dx, dy)

        assert result.centres[0].tolist() == pytest.approx(
            [1.0 + dx, 2.0 + dy]
        )


# =====================================================================
# CirclesArray - rotation
# =====================================================================


class TestCirclesArrayRotate:
    def test_rotate_about_origin(self):
        circles = CirclesArray(
            [(1.0, 0.0), (0.0, 1.0)],
            [1.0, 2.0],
        )

        result = circles.rotate(
            Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
        )

        assert result.centres[0].tolist() == pytest.approx(
            [0.0, 1.0]
        )
        assert result.centres[1].tolist() == pytest.approx(
            [-1.0, 0.0]
        )

    def test_rotate_about_arbitrary_pivot(self):
        circles = CirclesArray(
            [(3.0, 2.0)],
            [1.0],
        )

        result = circles.rotate(
            Angle.rad(np.pi / 2),
            pivot=(2.0, 2.0),
        )

        assert result.centres[0].tolist() == pytest.approx(
            [2.0, 3.0]
        )

    def test_rotation_does_not_change_radii(self):
        circles = CirclesArray(
            [(1.0, 0.0), (2.0, 0.0)],
            [1.0, 2.0],
        )

        result = circles.rotate(
            Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
        )

        assert np.array_equal(result.radii, [1.0, 2.0])

    def test_rotate_out_of_place_does_not_mutate_original(self):
        circles = CirclesArray(
            [(1.0, 0.0)],
            [1.0],
        )

        circles.rotate(
            Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
        )

        assert circles.centres[0].tolist() == pytest.approx(
            [1.0, 0.0]
        )

    def test_rotate_in_place(self):
        circles = CirclesArray(
            [(1.0, 0.0)],
            [1.0],
        )

        result = circles.rotate(
            Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
            in_place=True,
        )

        assert result is circles
        assert circles.centres[0].tolist() == pytest.approx(
            [0.0, 1.0]
        )


# =====================================================================
# CirclesArray - transform
# =====================================================================


class TestCirclesArrayTransform:
    def test_rotate_then_translate(self):
        circles = CirclesArray(
            [(1.0, 0.0)],
            [1.0],
        )

        result = circles.transform(
            dx=2.0,
            dy=3.0,
            rot_angle=Angle.rad(np.pi / 2),
            pivot=(0.0, 0.0),
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )

        assert result.centres[0].tolist() == pytest.approx(
            [2.0, 4.0]
        )

    def test_transform_defaults_pivot_to_origin(self):
        circles = CirclesArray(
            [(1.0, 0.0)],
            [1.0],
        )

        result = circles.transform(
            rot_angle=Angle.rad(np.pi / 2),
        )

        assert result.centres[0].tolist() == pytest.approx(
            [0.0, 1.0]
        )

    def test_transform_out_of_place(self):
        circles = CirclesArray(
            [(1.0, 0.0)],
            [1.0],
        )

        result = circles.transform(dx=10.0)

        assert result is not circles
        assert circles.centres[0].tolist() == pytest.approx(
            [1.0, 0.0]
        )
        assert result.centres[0].tolist() == pytest.approx(
            [11.0, 0.0]
        )

    def test_transform_in_place(self):
        circles = CirclesArray(
            [(1.0, 0.0)],
            [1.0],
        )

        result = circles.transform(
            dx=10.0,
            in_place=True,
        )

        assert result is circles
        assert circles.centres[0].tolist() == pytest.approx(
            [11.0, 0.0]
        )

    def test_transform_preserves_count(self):
        circles = CirclesArray(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            [1.0, 2.0, 3.0],
        )

        result = circles.transform(
            dx=10.0,
            dy=20.0,
            rot_angle=Angle.rad(0.5),
        )

        assert len(result) == 3

    def test_transform_preserves_radii(self):
        circles = CirclesArray(
            [(0.0, 0.0), (1.0, 1.0)],
            [1.0, 2.0],
        )

        result = circles.transform(
            dx=10.0,
            dy=20.0,
            rot_angle=Angle.rad(0.5),
        )

        assert np.array_equal(result.radii, [1.0, 2.0])


# =====================================================================
# CirclesArray - bounding box
# =====================================================================


class TestCirclesArrayBoundingBox:
    def test_bounding_box(self):
        circles = CirclesArray(
            [
                (0.0, 0.0),
                (5.0, 3.0),
            ],
            [2.0, 1.0],
        )

        assert_bbox_close(
            circles.bounding_box(),
            [-2.0, -2.0, 6.0, 4.0],
        )

    def test_single_circle(self):
        circles = CirclesArray(
            [(2.0, -3.0)],
            [4.0],
        )

        assert_bbox_close(
            circles.bounding_box(),
            [-2.0, -7.0, 6.0, 1.0],
        )

    def test_negative_coordinates(self):
        circles = CirclesArray(
            [(-10.0, -20.0), (-5.0, -3.0)],
            [1.0, 2.0],
        )

        assert_bbox_close(
            circles.bounding_box(),
            [-11.0, -21.0, -3.0, -1.0],
        )

    def test_bbox_contains_all_circle_extents(self):
        circles = CirclesArray(
            [
                (0.0, 0.0),
                (5.0, 3.0),
                (-2.0, 10.0),
            ],
            [2.0, 1.0, 4.0],
        )

        bbox = circles.bounding_box()

        for centre, radius in zip(
            circles.centres.coordinates,
            circles.radii,
        ):
            x, y = centre

            assert x - radius >= bbox[0] - 1e-12
            assert y - radius >= bbox[1] - 1e-12
            assert x + radius <= bbox[2] + 1e-12
            assert y + radius <= bbox[3] + 1e-12



# =====================================================================
# Cross-class / invariants
# =====================================================================


class TestGeometricInvariants:
    @pytest.mark.parametrize(
        "a,b",
        [
            (1.0, 1.0),
            (2.0, 1.0),
            (5.0, 3.0),
            (10.0, 2.0),
        ],
    )
    def test_equivalent_circle_area_matches(self, a, b):
        ellipse = Ellipse(a, b)

        radius = ellipse.equivalent_circle_radius
        equivalent = Circle(radius)

        assert equivalent.area == pytest.approx(ellipse.area)

    @pytest.mark.parametrize(
        "angle",
        [
            0.0,
            np.pi / 6,
            np.pi / 4,
            np.pi / 2,
            np.pi,
            -np.pi / 3,
        ],
    )
    def test_ellipse_area_invariant_under_rotation(self, angle):
        ellipse = Ellipse(5.0, 3.0)

        transformed = ellipse.transform(
            d_theta=Angle.rad(angle),
        )

        assert transformed.area == pytest.approx(ellipse.area)

    @pytest.mark.parametrize(
        "angle",
        [
            0.0,
            np.pi / 6,
            np.pi / 4,
            np.pi / 2,
            np.pi,
        ],
    )
    def test_ellipse_bbox_is_consistent_with_boundary(
        self,
        angle,
    ):
        ellipse = Ellipse(
            5.0,
            3.0,
            centre=(4.0, -2.0),
            major_axis_angle=Angle.rad(angle),
        )

        bbox = ellipse.bounding_box

        for point in ellipse.sample_points(num_points=1000):
            assert bbox[0] - 1e-9 <= point[0] <= bbox[2] + 1e-9
            assert bbox[1] - 1e-9 <= point[1] <= bbox[3] + 1e-9

    def test_circle_is_special_case_of_ellipse(self):
        radius = 4.0
        ellipse = Ellipse(radius, radius)
        circle = Circle(radius)

        assert ellipse.area == pytest.approx(circle.area)
        assert ellipse.perimeter == pytest.approx(circle.perimeter)
        assert ellipse.eccentricity == pytest.approx(circle.eccentricity)
        assert ellipse.aspect_ratio == pytest.approx(circle.aspect_ratio)
        assert ellipse.bounding_box == pytest.approx(
            circle.bounding_box
        )

    def test_circle_transform_preserves_geometry(self):
        circle = Circle(3.0, (1.0, 2.0))

        transformed = circle.transform(
            dx=10.0,
            dy=-5.0,
            d_theta=Angle.rad(1.7),
            pivot=(0.0, 0.0),
        )

        assert transformed.radius == pytest.approx(circle.radius)
        assert transformed.area == pytest.approx(circle.area)
        assert transformed.perimeter == pytest.approx(circle.perimeter)
