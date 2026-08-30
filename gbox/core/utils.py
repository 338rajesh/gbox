import logging
from collections.abc import Sequence
from numbers import Number
from typing import Any


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_logger(
    name: str = __name__, level: int = logging.INFO
) -> logging.Logger:
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


def _assert_a_sequence_of_numbers(
    seq, name: str = "input", length=None
) -> bool:
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
            raise ValueError(
                f"{name} must have length {length}, but got {len(seq)}"
            )
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
    coerce_type: bool = False,
) -> int:
    name = "" if name is None else name
    _validate_type(v, (int, float), name)
    v = int(v) if coerce_type else v
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
            raise ValueError(
                "When types is specified, keys must also be specified"
            )
