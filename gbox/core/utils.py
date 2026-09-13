import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Number
from typing import Any, Self


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TransformationOrder(StrEnum):
    ROTATE_THEN_TRANSLATE = "rotate_then_translate"
    TRANSLATE_THEN_ROTATE = "translate_then_rotate"


@dataclass(frozen=True, slots=True)
class Angle:
    """An immutable angle represented in degrees or radians."""

    value: float
    unit: str

    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise TypeError(
                f"Angle value must be a number (int or float), but got {type(self.value)}"
            )
        if isinstance(self.value, bool):
            raise TypeError(
                f"Angle value must be a number (int or float), but got {type(self.value)}"
            )
        if self.unit not in ("deg", "rad"):
            raise ValueError(f"Unknown unit {self.unit!r}. Expected 'deg' or 'rad'.")

    @property
    def radians(self) -> float:
        """Returns the angle in radians"""
        return self.value if self.unit == "rad" else math.radians(self.value)

    @property
    def degrees(self) -> float:
        """Returns the angle in degrees"""
        return self.value if self.unit == "deg" else math.degrees(self.value)

    @classmethod
    def rad(cls, value: float) -> Self:
        """Creates an Angle object from a value in radians"""
        return cls(value, "rad")

    @classmethod
    def deg(cls, value: float) -> Self:
        """Creates an Angle object from a value in degrees"""
        return cls(value, "deg")

    @property
    def cos(self) -> float:
        """Returns the cosine of the angle"""
        return math.cos(self.radians)

    @property
    def sin(self) -> float:
        """Returns the sine of the angle"""
        return math.sin(self.radians)

    @property
    def tan(self) -> float:
        """Returns the tangent of the angle"""
        return math.tan(self.radians)

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Angle):
            raise TypeError(f"Cannot add Angle with {type(other)}")
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot add Angle with different units: {self.unit} and {other.unit}"
            )
        return Angle(self.value + other.value, self.unit)

    def __sub__(self, other: Self) -> Self:
        if not isinstance(other, Angle):
            raise TypeError(f"Cannot subtract Angle with {type(other)}")
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot subtract Angle with different units: {self.unit} and {other.unit}"
            )
        return Angle(self.value - other.value, self.unit)

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, Angle):
            return False
        return math.isclose(self.radians, other.radians, rel_tol=1e-9, abs_tol=1e-9)

    def __repr__(self) -> str:
        return f"Angle({self.value}, '{self.unit}')"

    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Returns a logger with the given name and level"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def _is_a_number(x) -> bool:
    """Checks if the input is a number as defined in the numbers module"""
    return isinstance(x, Number)


def _assert_a_sequence(seq, name: str = "input") -> bool:
    """Checks if the input is a sequence"""
    if not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be a sequence, but got {type(seq)}")
    return True


def _assert_a_sequence_of_numbers(seq, name: str = "input", length=None) -> bool:
    """Checks if the input is a sequence of numbers"""
    if not all(_is_a_number(x) for x in seq):
        raise TypeError(
            f"All elements of {name} must be numbers (int or float), "
            f"but got {[type(x) for x in seq]}",
        )
    if length is not None:
        if not isinstance(length, int):
            raise ValueError(
                f"While asserting {name} to be a sequence of numbers, "
                f"the length argument must be an int, but got {type(length)}"
            )
        if len(seq) != length:
            raise ValueError(f"{name} must have length {length}, but got {len(seq)}")
    return True


def _validate_type(v, types: type | tuple, name: str | None = None):
    name = "" if name is None else name
    if not isinstance(v, types):
        raise ValueError(
            f"Given value '{name}' must be of type {types}, but got {type(v)}"
        )


def _validate_bounds(
    v: Any,
    low: Any = None,
    high: Any = None,
    name: str | None = None,
    closed_bounds: bool = True,
) -> None:
    if low is not None:
        if closed_bounds and v < low:
            raise ValueError(
                f"Given value '{name}' must be >= {low}, but got {v}",
            )
        elif not closed_bounds and v <= low:
            raise ValueError(
                f"Given value '{name}' must be > {low}, but got {v}",
            )

    if high is not None:
        if closed_bounds and v > high:
            raise ValueError(
                f"Given value '{name}' must be <= {high}, but got {v}",
            )
        elif not closed_bounds and v >= high:
            raise ValueError(
                f"Given value '{name}' must be < {high}, but got {v}",
            )


def _validate_float(
    v: Any,
    low: float = None,
    high: float = None,
    name: str | None = None,
    closed_bounds: bool = True,
    coerce_type: bool = True,
) -> float:
    name = "" if name is None else name
    _validate_type(v, (int, float), name)
    v = float(v) if coerce_type else v
    _validate_bounds(v, low, high, name, closed_bounds)
    return v


def _validate_int(
    v: Any,
    low: float = None,
    high: float = None,
    name: str | None = None,
    closed_bounds: bool = True,
) -> int:
    name = "" if name is None else name
    _validate_type(v, int, name)
    _validate_bounds(v, low, high, name, closed_bounds)
    return v


def _validate_positive_float(v: Any, name: str | None = None):
    return _validate_float(v, 0.0, float("inf"), name, False)


def _validate_positive_int(v: Any, name: str | None = None):
    return _validate_int(v, 0, float("inf"), name, False)


def _validate_dict(
    d: Any,
    keys: list[Any] = None,
    types: list[type | tuple] = None,
    name: str | None = None,
):
    _validate_type(d, dict, name)

    if keys is not None:
        _validate_type(keys, list, name)
        missing_keys = [k for k in keys if k not in d]
        if missing_keys:
            raise ValueError(f"Keys {missing_keys} not found in {name}")

        if types is not None:
            _validate_type(types, list, name)

            if len(keys) != len(types):
                raise ValueError(
                    "When types and keys are specified, they must have "
                    f"the same length, but got {len(keys)} and {len(types)}"
                )

            for k, t in zip(keys, types):
                _validate_type(d[k], t, name)
    else:
        if types is not None:
            raise ValueError("When types is specified, keys must also be specified")
