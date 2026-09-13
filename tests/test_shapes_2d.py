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
# Tests: Shape2DPose
# =====================================================================


class TestShape2DPose:
    def test_initialization_and_repr(self):
        pos = Shape2DPose(1.5, -2.5, Angle.rad(np.pi / 4))
        assert pos.x == 1.5
        assert pos.y == -2.5
        pytest.approx(pos.orientation.radians, float(np.pi / 4))
        assert (
            repr(pos)
            == f"Shape2DPose(x={1.5}, y={-2.5}, orientation={Angle.rad(np.pi / 4)})"
        )

    def test_immutability(self):
        pos = Shape2DPose(1.0, 2.0, Angle.rad(0.0))
        with pytest.raises((AttributeError, TypeError)):
            pos.x = 10.0  # type: ignore

    def test_rotate(self):
        pos = Shape2DPose(1.0, 2.0, Angle.rad(0.5))
        new_pos = pos.rotate(Angle.rad(0.25))
        assert new_pos is not pos
        assert pos.orientation.radians == 0.5
        assert new_pos.orientation.radians == pytest.approx(0.75)
        assert (new_pos.x, new_pos.y) == (pos.x, pos.y)

    def test_translate(self):
        pos = Shape2DPose(1.0, 2.0, Angle.rad(0.5))
        new_pos = pos.translate(-0.5, 3.0)
        assert new_pos is not pos
        assert (new_pos.x, new_pos.y) == (0.5, 5.0)
        assert new_pos.orientation == pos.orientation

    def test_transform(self):
        pos = Shape2DPose(1.0, 2.0, Angle.rad(0.5))
        new_pos = pos.transform(
            dx=1.0,
            dy=-1.0,
            rot_angle=Angle.rad(0.452),
            order=TransformationOrder.ROTATE_THEN_TRANSLATE,
        )
        assert new_pos is not pos
        assert (new_pos.x, new_pos.y) == (2.0, 1.0)
        assert new_pos.orientation.radians == pytest.approx(0.952)


# =====================================================================
# Tests: Shape2D (Abstract Base Interface)
# =====================================================================


class TestShape2DInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Shape2D()  # type: ignore

    def test_unimplemented_bounding_box_error(self):
        class DummyShape(Shape2D):
            @property
            def position(self):
                return Shape2DPose(0, 0, 0)

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
                return super().bounding_box

        dummy = DummyShape()
        assert dummy.equivalent_circle_radius == pytest.approx(2.0)
        with pytest.raises(
            NotImplementedError, match="Bounding box method not implemented"
        ):
            _ = dummy.bounding_box


# =====================================================================
# Tests: Ellipse
# =====================================================================


class TestEllipse:
    @pytest.fixture
    def canonical_ellipse(self):
        # Center at (0, 0), no rotation, a=5, b=3
        return Ellipse(5.0, 3.0, (0.0, 0.0), Angle.rad(0.0))

    # --- Validation ---

    def test_validations_on_init(self):
        # a < b error
        with pytest.raises(
            ValueError, match="Semi-major axis must be >= semi-minor axis"
        ):
            Ellipse(2.0, 4.0)

        # Non-positive parameters
        with pytest.raises(ValueError):
            Ellipse(-1.0, 2.0)
        with pytest.raises(ValueError):
            Ellipse(2.0, 0.0)

        # Invalid types
        with pytest.raises(ValueError):
            Ellipse("5", 3.0)  # type: ignore

    # --- Properties ---

    def test_geometric_properties(self, canonical_ellipse):
        assert canonical_ellipse.semi_major_length == 5.0
        assert canonical_ellipse.semi_minor_length == 3.0
        assert canonical_ellipse.centre == (0.0, 0.0)
        assert canonical_ellipse.aspect_ratio == pytest.approx(5.0 / 3.0)
        # e = sqrt(1 - 9/25) = 4/5 = 0.8
        assert canonical_ellipse.eccentricity == pytest.approx(0.8)
        assert canonical_ellipse.area == pytest.approx(PI * 15.0)
        assert canonical_ellipse.equivalent_circle_radius == pytest.approx(
            math.sqrt(15.0)
        )

    def test_perimeter_numerical_integration(self, canonical_ellipse):
        # Accurate complete elliptic integral approximation
        h = ((5.0 - 3.0) ** 2) / ((5.0 + 3.0) ** 2)
        ramanujan_p = PI * (5.0 + 3.0) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
        assert canonical_ellipse.perimeter == pytest.approx(ramanujan_p, rel=1e-4)

    def test_bounding_box_aligned(self, canonical_ellipse):
        bbox = canonical_ellipse.bounding_box
        assert bbox == pytest.approx([-5.0, -3.0, 5.0, 3.0])

    def test_bounding_box_rotated(self):
        # Rotated 90 degrees: a and b swap in AABB
        rotated = Ellipse(5.0, 3.0, (1.0, 2.0), Angle.rad(np.pi / 2))
        bbox = rotated.bounding_box
        assert bbox == pytest.approx([1.0 - 3.0, 2.0 - 5.0, 1.0 + 3.0, 2.0 + 5.0])

    # --- Point Containment ---

    def test_contains_point(self, canonical_ellipse):
        # Inside
        assert canonical_ellipse.contains_point((0.0, 0.0)) == 1
        assert canonical_ellipse.contains_point(Point2D(2.0, 1.0)) == 1

        # On boundary
        assert canonical_ellipse.contains_point((5.0, 0.0)) == 0
        assert canonical_ellipse.contains_point((0.0, -3.0)) == 0

        # Outside
        assert canonical_ellipse.contains_point((5.1, 0.0)) == -1
        assert canonical_ellipse.contains_point((0.0, 3.01)) == -1

    def test_contains_point_rotated_and_translated(self):
        # Shifted to (2, 3), rotated 90 degrees CCW
        ell = Ellipse(4.0, 2.0, (2.0, 3.0), Angle.rad(np.pi / 2))
        # Major axis is along Y now: extent y in [3 - 4, 3 + 4] = [-1, 7], x in [2 - 2, 2 + 2] = [0, 4]
        assert ell.contains_point((2.0, 3.0)) == 1
        assert ell.contains_point((2.0, 7.0)) == 0  # apex along major axis
        assert ell.contains_point((4.0, 3.0)) == 0  # apex along minor axis
        assert ell.contains_point((2.0, 7.05)) == -1

    # --- Point Sampling & Evaluation ---

    def test_point_at_angle(self, canonical_ellipse):
        p0 = canonical_ellipse.point_at_angle(0.0)
        assert (p0.x, p0.y) == pytest.approx((5.0, 0.0))

        p_pi_2 = canonical_ellipse.point_at_angle(np.pi / 2)
        assert (p_pi_2.x, p_pi_2.y) == pytest.approx((0.0, 3.0))

    def test_sample_points(self, canonical_ellipse):
        pts = canonical_ellipse.sample_points(num_points=100)
        assert isinstance(pts, PointArray2D)
        assert len(pts) == 100
        # Check all points lie on boundary within tolerance
        for p in pts:
            assert canonical_ellipse.contains_point(p.tolist(), atol=1e-3) == 0

    def test_sample_points_default_density(self, canonical_ellipse):
        pts = canonical_ellipse.sample_points()
        expected_len = max(16, int(10.0 * canonical_ellipse.perimeter))
        assert len(pts) == expected_len

    # --- Serialization / Factories / Clones ---

    def test_clone(self, canonical_ellipse):
        cloned = canonical_ellipse.clone()
        assert cloned is not canonical_ellipse
        assert cloned.semi_major_length == canonical_ellipse.semi_major_length
        assert cloned.semi_minor_length == canonical_ellipse.semi_minor_length
        assert cloned.centre == canonical_ellipse.centre
        assert cloned.position.orientation == canonical_ellipse.position.orientation

    def test_from_params(self):
        pos_dict = {"xc": 3.0, "yc": -1.0, "major_axis_angle": Angle.rad(0.5)}
        size_dict = {"semi_major_length": 6.0, "semi_minor_length": 2.0}
        ell = Ellipse.from_params(pos_dict, size_dict)
        assert ell.centre == (3.0, -1.0)
        assert ell.position.orientation == Angle.rad(0.5)
        assert ell.semi_major_length == 6.0
        assert ell.semi_minor_length == 2.0

    def test_from_params_validation_failure(self):
        with pytest.raises(ValueError):
            Ellipse.from_params({"xc": 0.0}, {"semi_major_length": 1.0})

    # --- Union of Circles & Tangent Circles ---

    def test_r_shortest_circle_case(self):
        circ_ell = Ellipse(4.0, 4.0)
        assert circ_ell.r_shortest(0.0) == 4.0
        assert circ_ell.r_shortest(2.0) == 4.0

    def test_r_shortest_tangent_radius(self, canonical_ellipse):
        # a=5, b=3. xi=0 -> r_min = b = 3.0
        assert canonical_ellipse.r_shortest(0.0) == pytest.approx(3.0)

    def test_union_of_circles_aspect_ratio_1(self):
        circ_ell = Ellipse(3.0, 3.0, (1.0, 1.0))
        circles = circ_ell.union_of_circles()
        assert len(circles) == 1
        assert isinstance(circles[0], Circle)
        assert circles[0].radius == 3.0
        assert circles[0].centre == (1.0, 1.0)

    def test_union_of_circles_dh_bounds(self, canonical_ellipse):
        with pytest.raises(ValueError, match="dh must be > 0"):
            canonical_ellipse.union_of_circles(dh=0.0)

        with pytest.raises(ValueError):
            # dh >= semi_minor_length violates closed_bounds=False check
            canonical_ellipse.union_of_circles(
                dh=canonical_ellipse.semi_minor_length + 1.0
            )

    def test_union_of_circles_generation(self):
        ell = Ellipse(5.0, 3.0, (1.0, 2.0), Angle.rad(np.pi / 4))
        circles = ell.union_of_circles(dh=0.1)
        assert len(circles) > 1
        for c in circles:
            assert isinstance(c, Circle)
            assert c.radius <= ell.semi_minor_length + 1e-6


# =====================================================================
# Tests: Circle
# =====================================================================


class TestCircle:
    def test_init_and_properties(self):
        c = Circle(radius=4.5, centre=(2.0, -3.0))
        assert c.radius == 4.5
        assert c.semi_major_length == 4.5
        assert c.semi_minor_length == 4.5
        assert c.centre == (2.0, -3.0)
        assert c.eccentricity == 0.0
        assert c.aspect_ratio == 1.0
        assert c.area == pytest.approx(PI * 4.5**2)
        assert c.perimeter == pytest.approx(2.0 * PI * 4.5)
        assert c.bounding_box == pytest.approx([-2.5, -7.5, 6.5, 1.5])

    def test_clone(self):
        c = Circle(2.0, (1.0, 1.0))
        cloned = c.clone()
        assert cloned is not c
        assert isinstance(cloned, Circle)
        assert cloned.radius == 2.0
        assert cloned.centre == (1.0, 1.0)


# =====================================================================
# Tests: CirclesArray
# =====================================================================


class TestCirclesArray:
    @pytest.fixture
    def sample_circles_array(self):
        centres = [(0.0, 0.0), (3.0, 4.0), (-2.0, 1.0)]
        radii = [1.0, 2.0, 0.5]
        return CirclesArray(centres, radii)

    # --- Initialization & Validation ---

    def test_init_with_pointarray2d(self):
        pts = PointArray2D.from_named_dims(x=[0.0, 1.0], y=[0.0, 1.0])
        ca = CirclesArray(pts, 2.0)
        assert len(ca) == 2
        assert np.all(ca.radii == 2.0)

    def test_init_with_numpy_centres(self):
        centres = np.array([[1.0, 2.0], [3.0, 4.0]])
        ca = CirclesArray(centres, [1.0, 2.0])
        assert len(ca) == 2

    def test_init_validation_errors(self):
        # Dimension shape errors
        with pytest.raises(
            ValueError, match=r"centres must be a 2D array with shape \(N, 2\)"
        ):
            CirclesArray(np.array([1.0, 2.0, 3.0]), 1.0)

        # Length mismatches
        with pytest.raises(
            ValueError, match="radii length must match number of centres"
        ):
            CirclesArray([(0.0, 0.0), (1.0, 1.0)], [1.0])

        # Non-positive radii
        with pytest.raises(ValueError, match="all radii must be positive"):
            CirclesArray([(0.0, 0.0)], [0.0])
        with pytest.raises(ValueError, match="all radii must be positive"):
            CirclesArray([(0.0, 0.0)], [-1.5])

    # --- Conversions & Core Methods ---

    def test_from_and_to_circles(self):
        orig_circles = [Circle(1.0, (0.0, 0.0)), Circle(2.5, (3.0, 5.0))]
        ca = CirclesArray.from_circles(orig_circles)
        assert len(ca) == 2
        assert np.all(ca.radii == [1.0, 2.5])

        recovered = ca.to_circles()
        assert len(recovered) == 2
        assert all(isinstance(c, Circle) for c in recovered)
        assert recovered[0].centre == pytest.approx((0.0, 0.0))
        assert recovered[0].radius == 1.0
        assert recovered[1].centre == pytest.approx((3.0, 5.0))
        assert recovered[1].radius == 2.5

    def test_clone(self, sample_circles_array):
        cloned = sample_circles_array.clone()
        assert cloned is not sample_circles_array
        assert np.array_equal(cloned.radii, sample_circles_array.radii)
        assert np.array_equal(
            cloned.centres.coordinates, sample_circles_array.centres.coordinates
        )

    # --- Transformations ---

    def test_translate(self, sample_circles_array):
        # Out-of-place
        translated = sample_circles_array.translate(2.0, -1.0, in_place=False)
        assert translated is not sample_circles_array
        assert translated.centres[0].tolist() == pytest.approx([2.0, -1.0])
        assert sample_circles_array.centres[0].tolist() == pytest.approx([0.0, 0.0])

        # In-place
        res = sample_circles_array.translate(2.0, -1.0, in_place=True)
        assert res is sample_circles_array
        assert sample_circles_array.centres[0].tolist() == pytest.approx([2.0, -1.0])

    def test_rotate(self):
        ca = CirclesArray([(1.0, 0.0)], [1.0])
        rotated = ca.rotate(Angle.rad(np.pi / 2), pivot=(0.0, 0.0), in_place=False)
        assert rotated.centres[0].tolist() == pytest.approx([0.0, 1.0])
        assert rotated.radii[0] == 1.0  # Radii must remain invariant

    def test_transform(self):
        ca = CirclesArray([(1.0, 0.0)], [1.0])
        # Rotate 90 deg about origin -> (0, 1), then translate dx=2, dy=3 -> (2, 4)
        transformed = ca.transform(
            rot_angle=Angle.rad(np.pi / 2), dx=2.0, dy=3.0, pivot=(0.0, 0.0)
        )
        assert transformed.centres[0].tolist() == pytest.approx([2.0, 4.0])

    # --- Bounding Box ---

    def test_bounding_box(self):
        # Circles:
        # 1: centre=(0, 0), r=2 -> x in [-2, 2], y in [-2, 2]
        # 2: centre=(5, 3), r=1 -> x in [4, 6], y in [2, 4]
        ca = CirclesArray([(0.0, 0.0), (5.0, 3.0)], [2.0, 1.0])
        bbox = ca.bounding_box()
        assert bbox == pytest.approx([-2.0, -2.0, 6.0, 4.0])
