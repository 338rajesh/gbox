import logging
import math
from numbers import Number
from pathlib import Path

import pytest

from gbox.core.utils import (
    Angle,
    Bounds2DRectangular,
    TransformationOrder,
    Validator,
    get_logger,
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

    @pytest.mark.parametrize("value", [1, 1.5, -2, math.pi])
    def test_numeric_values_are_accepted(self, value):
        assert Angle.deg(value).value == value

    def test_bool_value_is_rejected(self):
        with pytest.raises(TypeError, match="Angle value must be a number"):
            Angle.deg(True)

    def test_non_numeric_value_is_rejected(self):
        with pytest.raises(TypeError, match="Angle value must be a number"):
            Angle.deg("90")

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
            (Angle.deg(180), 0.0),
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
        with pytest.raises(TypeError, match="Cannot add Angle"):
            Angle.deg(10) + 5

    def test_subtract_non_angle_rejected(self):
        with pytest.raises(TypeError, match="Cannot subtract Angle"):
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
# Bounds2DRectangular
# ============================================================


class TestBounds2DRectangular:
    def test_constructor(self):
        bounds = Bounds2DRectangular(-1, -2, 3, 4)

        assert bounds.x_min == -1
        assert bounds.y_min == -2
        assert bounds.x_max == 3
        assert bounds.y_max == 4

    def test_requires_strictly_increasing_x_bounds(self):
        with pytest.raises(ValueError, match="x_min >= x_max"):
            Bounds2DRectangular(1, 0, 1, 2)

    def test_requires_strictly_increasing_y_bounds(self):
        with pytest.raises(ValueError, match="y_min >= y_max"):
            Bounds2DRectangular(0, 1, 2, 1)

    def test_from_sequence(self):
        bounds = Bounds2DRectangular.from_sequence([-1, -2, 3, 4])

        assert bounds == Bounds2DRectangular(-1, -2, 3, 4)

    def test_from_sequence_returns_existing_instance(self):
        bounds = Bounds2DRectangular(-1, -2, 3, 4)
        assert Bounds2DRectangular.from_sequence(bounds) is bounds

    def test_from_sequence_requires_four_elements(self):
        with pytest.raises(ValueError, match="exactly four elements"):
            Bounds2DRectangular.from_sequence([0, 1, 2])

    def test_is_frozen(self):
        bounds = Bounds2DRectangular(0, 0, 1, 1)

        with pytest.raises((AttributeError, TypeError)):
            bounds.x_min = -1


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
# get_logger
# ============================================================


def test_get_logger_returns_named_logger():
    logger = get_logger("test-utils-logger", logging.DEBUG)

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test-utils-logger"
    assert logger.level == logging.DEBUG


# ============================================================
# Validator.is_type
# ============================================================


class TestValidatorIsType:
    def test_accepts_expected_type(self):
        assert Validator.is_type(1, int) is True

    def test_accepts_tuple_of_types(self):
        assert Validator.is_type(1.5, (int, float)) is True

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.is_type("1", int)

    def test_rejects_excluded_type(self):
        with pytest.raises(TypeError, match="must not be of type"):
            Validator.is_type(True, (int, float), exclusion_types=bool)

    def test_name_is_in_error_message(self):
        with pytest.raises(TypeError, match="temperature"):
            Validator.is_type("hot", float, name="temperature")


# ============================================================
# Validator.bounds
# ============================================================


class TestValidatorBounds:
    def test_closed_lower_bound(self):
        assert Validator.in_bounds(1, low=1) is True

    def test_closed_upper_bound(self):
        assert Validator.in_bounds(1, high=1) is True

    def test_closed_lower_bound_failure(self):
        with pytest.raises(ValueError, match=r">= 1"):
            Validator.in_bounds(0, low=1)

    def test_closed_upper_bound_failure(self):
        with pytest.raises(ValueError, match=r"<= 1"):
            Validator.in_bounds(2, high=1)

    def test_open_lower_bound(self):
        assert Validator.in_bounds(2, low=1, closed_bounds=False) is True

    def test_open_upper_bound(self):
        assert Validator.in_bounds(0, high=1, closed_bounds=False) is True

    def test_open_lower_bound_failure_at_boundary(self):
        with pytest.raises(ValueError, match=r"> 1"):
            Validator.in_bounds(1, low=1, closed_bounds=False)

    def test_open_upper_bound_failure_at_boundary(self):
        with pytest.raises(ValueError, match=r"< 1"):
            Validator.in_bounds(1, high=1, closed_bounds=False)

    def test_unbounded_value(self):
        assert Validator.in_bounds(123) is True

    def test_name_is_in_error_message(self):
        with pytest.raises(ValueError, match="length"):
            Validator.in_bounds(0, low=1, name="length")


# ============================================================
# Validator.sequence
# ============================================================


class TestValidatorSequence:
    @pytest.mark.parametrize(
        "seq",
        [
            [],
            [1, 2, 3],
            (1, 2, 3),
            "abc",
        ],
    )
    def test_accepts_sequences(self, seq):
        assert Validator.as_sequence(seq) is seq

    def test_rejects_non_sequence_without_len(self):
        with pytest.raises(TypeError):
            Validator.as_sequence(123)

    @pytest.mark.parametrize(
        "seq",
        [
            [],
            [1],
            [1, 2.5, -3],
            (1, 2, 3),
        ],
    )
    def test_validates_element_type(self, seq):
        assert Validator.as_sequence(seq, ele_type=Number) is seq

    def test_rejects_invalid_element_type(self):
        with pytest.raises(TypeError, match="All elements"):
            Validator.as_sequence([1, "x", 3], ele_type=Number)

    def test_validates_length(self):
        assert Validator.as_sequence([1, 2], length=2) == [1, 2]

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="length 2"):
            Validator.as_sequence([1, 2], length=3)

    def test_rejects_invalid_length_argument(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_sequence([1, 2], length=2.0)

    def test_validates_required_elements(self):
        seq = ["x", "y", "z"]
        assert Validator.as_sequence(seq, req_elements=["x", "z"]) is seq

    def test_rejects_missing_required_elements(self):
        with pytest.raises(ValueError, match="missing"):
            Validator.as_sequence(["x", "y"], req_elements=["x", "z"])

    def test_allows_none_when_requested(self):
        assert Validator.as_sequence(None, allow_none=True) is None

    def test_rejects_none_by_default(self):
        with pytest.raises(TypeError):
            Validator.as_sequence(None)

    def test_validates_length_argument_even_for_zero(self):
        assert Validator.as_sequence([], length=0) == []


# ============================================================
# Validator.as_float
# ============================================================


class TestValidatorFloat:
    def test_converts_int_to_float(self):
        result = Validator.as_float(2)

        assert result == 2.0
        assert isinstance(result, float)

    def test_preserves_float(self):
        result = Validator.as_float(2.5)

        assert result == 2.5
        assert isinstance(result, float)

    def test_validates_bounds(self):
        assert Validator.as_float(5, low=0, high=10) == 5.0

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_float("1.5")

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="must not be of type"):
            Validator.as_float(True)

    def test_open_lower_bound(self):
        with pytest.raises(ValueError, match=r"> 0"):
            Validator.as_float(0, low=0, closed_bounds=False)

    def test_open_upper_bound(self):
        with pytest.raises(ValueError, match=r"< 1"):
            Validator.as_float(1, high=1, closed_bounds=False)

    def test_zero_is_valid_by_default(self):
        assert Validator.as_float(0) == 0.0

    def test_negative_values_are_valid_by_default(self):
        assert Validator.as_float(-1) == -1.0

    def test_allow_none(self):
        assert Validator.as_float(None, allow_none=True) is None

    def test_none_is_rejected_by_default(self):
        with pytest.raises(TypeError):
            Validator.as_float(None)


# ============================================================
# Validator.as_int
# ============================================================


class TestValidatorInt:
    def test_accepts_int(self):
        result = Validator.as_int(3)

        assert result == 3
        assert isinstance(result, int)

    def test_validates_bounds(self):
        assert Validator.as_int(3, low=1, high=5) == 3

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_int("3")

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="must not be of type"):
            Validator.as_int(True)

    def test_rejects_fractional_values(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_int(1.5)

    def test_open_lower_bound(self):
        with pytest.raises(ValueError, match=r"> 0"):
            Validator.as_int(0, low=0, closed_bounds=False)

    def test_open_upper_bound(self):
        with pytest.raises(ValueError, match=r"< 3"):
            Validator.as_int(3, high=3, closed_bounds=False)

    def test_allow_none(self):
        assert Validator.as_int(None, allow_none=True) is None

    def test_none_is_rejected_by_default(self):
        with pytest.raises(TypeError):
            Validator.as_int(None)


# ============================================================
# Validator.as_dict
# ============================================================


class TestValidatorDict:
    def test_accepts_dict(self):
        value = {"x": 1}
        assert Validator.as_dict(value) is value

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_dict([])

    def test_checks_required_keys(self):
        value = {"x": 1}
        assert Validator.as_dict(value, keys=["x"]) is value

    def test_missing_key(self):
        with pytest.raises(ValueError, match="Missing Keys"):
            Validator.as_dict({"x": 1}, keys=["x", "y"])

    def test_checks_value_types(self):
        value = {"x": 1}
        assert Validator.as_dict(value, keys=["x"], types=[int]) is value

    def test_wrong_value_type(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_dict({"x": "1"}, keys=["x"], types=[int])

    def test_keys_and_types_length_must_match(self):
        with pytest.raises(ValueError, match="same length"):
            Validator.as_dict({"x": 1}, keys=["x"], types=[int, float])

    def test_types_require_keys(self):
        with pytest.raises(ValueError, match="keys"):
            Validator.as_dict({"x": 1}, types=[int])

    def test_keys_must_be_list(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_dict({"x": 1}, keys=("x",))

    def test_types_must_be_list(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.as_dict({"x": 1}, keys=["x"], types=(int,))

    def test_extra_keys_are_allowed_by_default(self):
        value = {"x": 1, "y": 2}
        assert Validator.as_dict(value, keys=["x"]) is value

    def test_extra_keys_can_be_rejected(self):
        with pytest.raises(ValueError, match="Extra Keys"):
            Validator.as_dict(
                {"x": 1, "y": 2},
                keys=["x"],
                reject_extra_keys=True,
            )

    def test_reject_extra_keys_requires_keys(self):
        with pytest.raises(ValueError, match="keys must also be provided"):
            Validator.as_dict({"x": 1}, reject_extra_keys=True)

    def test_allow_none(self):
        assert Validator.as_dict(None, allow_none=True) is None

    def test_none_is_rejected_by_default(self):
        with pytest.raises(TypeError):
            Validator.as_dict(None)


# ============================================================
# Validator.file_path
# ============================================================


class TestValidatorFilePath:
    def test_accepts_string_path(self, tmp_path):
        path = Validator.file_path(tmp_path / "file.txt")

        assert isinstance(path, Path)
        assert path == (tmp_path / "file.txt").resolve()

    def test_accepts_path_object(self, tmp_path):
        expected = (tmp_path / "file.txt").resolve()
        assert Validator.file_path(expected) == expected

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError, match="must be of type"):
            Validator.file_path(123)

    def test_requires_existing_path(self, tmp_path):
        path = tmp_path / "existing.txt"
        path.touch()

        assert Validator.file_path(path, must_exist=True) == path.resolve()

    def test_missing_required_path_is_rejected(self, tmp_path):
        path = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            Validator.file_path(path, must_exist=True)

    def test_rejects_existing_path_when_must_not_exist(self, tmp_path):
        path = tmp_path / "existing.txt"
        path.touch()

        with pytest.raises(FileExistsError, match="already exists"):
            Validator.file_path(path, must_exist=False)

    def test_accepts_non_existing_path_when_must_not_exist(self, tmp_path):
        path = tmp_path / "new.txt"

        assert Validator.file_path(path, must_exist=False) == path.resolve()

    def test_validates_extension(self, tmp_path):
        path = tmp_path / "data.txt"

        assert Validator.file_path(path, extensions=[".txt"]) == path.resolve()

    def test_rejects_invalid_extension(self, tmp_path):
        path = tmp_path / "data.csv"

        with pytest.raises(ValueError, match="missing"):
            Validator.file_path(path, extensions=[".txt"])

    def test_extension_list_must_contain_strings(self, tmp_path):
        path = tmp_path / "data.txt"

        with pytest.raises(TypeError, match="All elements"):
            Validator.file_path(path, extensions=[".txt", 123])

    def test_create_parent(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "data.txt"

        result = Validator.file_path(path, create_parent=True)

        assert result == path.resolve()
        assert path.parent.is_dir()

    def test_does_not_create_parent_by_default(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "data.txt"

        Validator.file_path(path)

        assert not path.parent.exists()

    def test_can_preserve_relative_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        result = Validator.file_path("data.txt", resolve_path=False)

        assert result == Path("data.txt")
        assert not result.is_absolute()
