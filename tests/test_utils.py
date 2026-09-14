import math

import pytest

from gbox.core.utils import (
    Angle,
    TransformationOrder,
    _assert_a_sequence,
    _assert_a_sequence_of_numbers,
    _is_a_number,
    _validate_bounds,
    _validate_dict,
    _validate_float,
    _validate_int,
    _validate_positive_float,
    _validate_positive_int,
    _validate_type,
)


# ============================================================
# Angle
# ============================================================


class TestAngle:
    def test_degree_constructor(self):
        angle = Angle.deg(90)

        assert angle.value == 90
        assert angle.unit == "deg"

    def test_radian_constructor(self):
        angle = Angle.rad(math.pi)

        assert angle.value == math.pi
        assert angle.unit == "rad"

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            Angle(90, "gradians")

    def test_radians_from_degrees(self):
        assert Angle.deg(180).radians == pytest.approx(math.pi)

    def test_radians_from_radians(self):
        assert Angle.rad(math.pi).radians == pytest.approx(math.pi)

    def test_degrees_from_degrees(self):
        assert Angle.deg(180).degrees == pytest.approx(180)

    def test_degrees_from_radians(self):
        assert Angle.rad(math.pi).degrees == pytest.approx(180)

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (Angle.deg(0), 1.0),
            (Angle.deg(60), 0.5),
            (Angle.deg(90), 0.0),
            (Angle.deg(180), -1.0),
        ],
    )
    def test_cos(self, angle, expected):
        assert angle.cos == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (Angle.deg(0), 0.0),
            (Angle.deg(30), 0.5),
            (Angle.deg(90), 1.0),
            (
                Angle.deg(180),
                0.0,
            ),
        ],
    )
    def test_sin(self, angle, expected):
        assert angle.sin == pytest.approx(expected, abs=1e-12)

    def test_tan(self):
        assert Angle.deg(45).tan == pytest.approx(1.0)

    def test_add_same_units(self):
        assert Angle.deg(20) + Angle.deg(30) == Angle.deg(50)

    def test_subtract_same_units(self):
        assert Angle.deg(50) - Angle.deg(20) == Angle.deg(30)

    def test_add_different_units_rejected(self):
        with pytest.raises(ValueError, match="different units"):
            Angle.deg(90) + Angle.rad(math.pi / 2)

    def test_subtract_different_units_rejected(self):
        with pytest.raises(ValueError, match="different units"):
            Angle.deg(90) - Angle.rad(math.pi / 2)

    def test_add_non_angle_rejected(self):
        with pytest.raises(TypeError):
            Angle.deg(10) + 5

    def test_subtract_non_angle_rejected(self):
        with pytest.raises(TypeError):
            Angle.deg(10) - 5

    def test_equality_is_unit_independent(self):
        assert Angle.deg(180) == Angle.rad(math.pi)

    def test_equality_is_tolerant(self):
        assert Angle.rad(1.0) == Angle.rad(1.0 + 1e-10)

    def test_equality_rejects_non_angle(self):
        assert Angle.deg(10) != 10

    def test_repr(self):
        assert repr(Angle.deg(30)) == "Angle(30, 'deg')"

    def test_str(self):
        assert str(Angle.deg(30)) == "30 deg"

    def test_is_frozen(self):
        angle = Angle.deg(30)

        with pytest.raises((AttributeError, TypeError)):
            angle.value = 40

    def test_negative_angles_are_allowed(self):
        assert Angle.deg(-90).sin == pytest.approx(-1.0)

    def test_angles_greater_than_full_rotation_are_allowed(self):
        assert Angle.deg(450).sin == pytest.approx(1.0)


# ============================================================
# TransformationOrder
# ============================================================


class TestTransformationOrder:
    def test_members(self):
        assert TransformationOrder.ROTATE_THEN_TRANSLATE.value == (
            "rotate_then_translate"
        )
        assert TransformationOrder.TRANSLATE_THEN_ROTATE.value == (
            "translate_then_rotate"
        )

    def test_is_str_enum(self):
        assert isinstance(TransformationOrder.ROTATE_THEN_TRANSLATE, str)


# ============================================================
# Generic validation helpers
# ============================================================


@pytest.mark.parametrize(
    "value",
    [1, 1.5, 0, -3, 1 + 2j],
)
def test_is_a_number(value):
    assert _is_a_number(value)


def test_is_a_number_rejects_strings():
    assert not _is_a_number("1")


def test_assert_sequence_accepts_list():
    assert _assert_a_sequence([1, 2, 3])


def test_assert_sequence_accepts_tuple():
    assert _assert_a_sequence((1, 2, 3))


def test_assert_sequence_accepts_string():
    # This is technically Sequence behavior.
    assert _assert_a_sequence("abc")


def test_assert_sequence_rejects_scalar():
    with pytest.raises(TypeError, match="must be a sequence"):
        _assert_a_sequence(123)


@pytest.mark.parametrize(
    "seq",
    [
        [],
        [1],
        [1, 2.5, -3],
        (1, 2, 3),
    ],
)
def test_assert_sequence_of_numbers(seq):
    assert _assert_a_sequence_of_numbers(seq)


def test_assert_sequence_of_numbers_rejects_non_number():
    with pytest.raises(TypeError, match="All elements"):
        _assert_a_sequence_of_numbers([1, "x", 3])


def test_assert_sequence_of_numbers_length():
    assert _assert_a_sequence_of_numbers([1, 2], length=2)


def test_assert_sequence_of_numbers_wrong_length():
    with pytest.raises(ValueError, match="length 3"):
        _assert_a_sequence_of_numbers([1, 2], length=3)


def test_assert_sequence_of_numbers_invalid_length_type():
    with pytest.raises(ValueError, match="length argument must be an int"):
        _assert_a_sequence_of_numbers([1, 2], length=2.0)


def test_validate_type_accepts_expected_type():
    _validate_type(1, int)


def test_validate_type_rejects_wrong_type():
    with pytest.raises(ValueError, match="must be of type"):
        _validate_type("1", int)


# ============================================================
# Bounds
# ============================================================


def test_closed_lower_bound():
    _validate_bounds(1, low=1)


def test_closed_upper_bound():
    _validate_bounds(1, high=1)


def test_closed_lower_bound_failure():
    with pytest.raises(ValueError, match=">= 1"):
        _validate_bounds(0, low=1)


def test_closed_upper_bound_failure():
    with pytest.raises(ValueError, match="<= 1"):
        _validate_bounds(2, high=1)


def test_open_lower_bound():
    _validate_bounds(2, low=1, closed_bounds=False)


def test_open_upper_bound():
    _validate_bounds(0, high=1, closed_bounds=False)


def test_open_lower_bound_failure_at_boundary():
    with pytest.raises(ValueError, match="> 1"):
        _validate_bounds(1, low=1, closed_bounds=False)


def test_open_upper_bound_failure_at_boundary():
    with pytest.raises(ValueError, match="< 1"):
        _validate_bounds(1, high=1, closed_bounds=False)


def test_unbounded_value():
    _validate_bounds(123)


# ============================================================
# Float validation
# ============================================================


def test_validate_float():
    assert _validate_float(2) == 2.0
    assert isinstance(_validate_float(2), float)


def test_validate_float_preserves_float():
    value = _validate_float(2.5)

    assert value == 2.5
    assert isinstance(value, float)


def test_validate_float_bounds():
    assert _validate_float(5, low=0, high=10) == 5


def test_validate_float_rejects_string():
    with pytest.raises(ValueError):
        _validate_float("1.5")


def test_validate_float_open_bounds():
    with pytest.raises(ValueError):
        _validate_float(0, low=0, closed_bounds=False)


def test_validate_positive_float():
    assert _validate_positive_float(0.1) == pytest.approx(0.1)


def test_validate_positive_float_rejects_zero():
    with pytest.raises(ValueError):
        _validate_positive_float(0)


def test_validate_positive_float_rejects_negative():
    with pytest.raises(ValueError):
        _validate_positive_float(-1)


# ============================================================
# Integer validation
# ============================================================


def test_validate_int_accepts_int():
    result = _validate_int(3)

    assert result == 3
    assert isinstance(result, int)


def test_validate_int_bounds():
    assert _validate_int(3, low=1, high=5) == 3


def test_validate_int_rejects_string():
    with pytest.raises(ValueError):
        _validate_int("3")


def test_validate_positive_int():
    assert _validate_positive_int(3) == 3


def test_validate_positive_int_rejects_zero():
    with pytest.raises(ValueError):
        _validate_positive_int(0)


def test_validate_positive_int_rejects_negative():
    with pytest.raises(ValueError):
        _validate_positive_int(-1)


def test_validate_int_rejects_fractional_values():
    """
    Intended behavior recommendation.

    This currently exposes a bug because _validate_int(1.5)
    is accepted when coerce_type=False.
    """
    with pytest.raises((ValueError, TypeError)):
        _validate_int(1.5)


# ============================================================
# Dict validation
# ============================================================


def test_validate_dict_accepts_dict():
    _validate_dict({"x": 1})


def test_validate_dict_rejects_non_dict():
    with pytest.raises(ValueError):
        _validate_dict([])


def test_validate_dict_checks_required_keys():
    _validate_dict({"x": 1}, keys=["x"])


def test_validate_dict_missing_key():
    with pytest.raises(ValueError, match="Keys"):
        _validate_dict({"x": 1}, keys=["x", "y"])


def test_validate_dict_checks_types():
    _validate_dict({"x": 1}, keys=["x"], types=[int])


def test_validate_dict_wrong_type():
    with pytest.raises(ValueError):
        _validate_dict({"x": "1"}, keys=["x"], types=[int])


def test_validate_dict_keys_and_types_length_must_match():
    with pytest.raises(ValueError, match="same length"):
        _validate_dict({"x": 1}, keys=["x"], types=[int, float])


def test_validate_dict_types_require_keys():
    with pytest.raises(ValueError, match="keys"):
        _validate_dict({"x": 1}, types=[int])
