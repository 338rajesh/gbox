import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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
            raise ValueError(
                f"Unknown unit {self.unit!r}. Expected 'deg' or 'rad'."
            )

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
        return math.isclose(
            self.radians, other.radians, rel_tol=1e-9, abs_tol=1e-9
        )

    def __repr__(self) -> str:
        return f"Angle({self.value}, '{self.unit}')"

    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


@dataclass(frozen=True, slots=True)
class Bounds2DRectangular:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if not self.x_min < self.x_max:
            raise ValueError(f"x_min >= x_max: {self.x_min} >= {self.x_max}")
        if not self.y_min < self.y_max:
            raise ValueError(f"y_min >= y_max: {self.y_min} >= {self.y_max}")

    @classmethod
    def from_sequence(cls, s: Sequence) -> Self:
        if isinstance(s, cls):
            return s
        if len(s) != 4:
            raise ValueError(
                f"The sequence used for creating {cls.__name__} must have "
                f"exactly four elements,  but got {len(s)}"
            )
        return cls(*s)

    @classmethod
    def from_mapping(cls, d: Mapping) -> Self:
        if isinstance(d, cls):
            return d
        if not isinstance(d, Mapping) and len(d) != 4:
            raise ValueError(
                f"The dictionary used for creating {cls.__name__} must be a "
                f"mapping of length 4, but got {type(d)} of length {len(d)}."
            )
        return cls(**d)

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @property
    def bounds(self) -> Mapping[str, float]:
        return self.to_dict()

    @property
    def x_len(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_len(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.x_len * self.y_len

    def overlaps(self, bb: Bounds2DRectangular) -> bool:
        Validator.is_type(
            bb,
            Bounds2DRectangular,
            name="Other bounding box of overlap checking",
        )
        return (
            self.x_max >= bb.x_min
            and self.x_min <= bb.x_max
            and self.y_max >= bb.y_min
            and self.y_min <= bb.y_max
        )


def get_logger(
    name: str = __name__, level: int = logging.INFO
) -> logging.Logger:
    """Returns a logger with the given name and level"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


class Validator:
    @staticmethod
    def is_type(
        v,
        types: type | tuple,
        *,
        exclusion_types: type | tuple | None = None,
        name: str | None = None,
    ) -> bool:
        name = "" if name is None else name
        if not isinstance(v, types):
            raise TypeError(
                f"Given value '{name}' must be of type {types}, but got {type(v)}"
            )
        if exclusion_types is not None and isinstance(v, exclusion_types):
            raise TypeError(
                f"Given value '{name}' must not be of type "
                f"{exclusion_types}, but got {type(v)}"
            )
        return True

    @staticmethod
    def has(
        v: Any,
        collection: Collection,
        *,
        name: str | None = None,
        num_repeats: int = 1,
        allow_none: bool = False,
    ) -> bool:
        name = "" if name is None else name
        if allow_none and v is None:
            return True
        actual_repeats = len([x for x in collection if x == v])
        Validator.as_int(num_repeats, low=1, name="num_repeats")
        if actual_repeats != num_repeats:
            raise ValueError(
                f"The given value '{name}' does not have "
                f"the expected number of repeats {num_repeats}"
            )

        return True

    @staticmethod
    def in_bounds(
        v: Any,
        low: Any = None,
        high: Any = None,
        name: str | None = None,
        closed_bounds: bool = True,
    ) -> bool:
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
        return True

    @staticmethod
    def as_sequence(
        seq,
        *,
        name: str = "input",
        ele_type: type | None = None,
        length: int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        allow_none: bool = False,
        req_elements: Collection[Any] | None = None,
    ) -> Sequence[Any] | None:
        """Returns validated sequence"""
        if allow_none and seq is None:
            return None

        try:
            len(seq)
            iter(seq)
        except TypeError:
            raise TypeError(
                f"Given value '{name}' must be a sequence, but got {type(seq)}"
            )

        seq_len = len(seq)
        if min_length is not None:
            Validator.is_type(min_length, int)
            if max_length is not None:
                Validator.is_type(max_length, int)
                if min_length > max_length:
                    raise ValueError("Given minimum length > maximum length")

            if seq_len < min_length:
                raise ValueError(
                    f"Given value '{name}' must have a length of at least "
                    f"{min_length}, but got a sequence of length {seq_len}"
                )

        if max_length is not None:
            Validator.is_type(max_length, int)
            if seq_len > max_length:
                raise ValueError(
                    f"Given value '{name}' must have a length of at most "
                    f"{max_length}, but got a sequence of length {seq_len}"
                )

        if length is not None:
            Validator.is_type(length, int)
            if seq_len != length:
                raise ValueError(
                    f"Given sequence '{name}' must have exact length {length} "
                    f"but got a sequence of length {seq_len}"
                )

        if ele_type is not None:
            invalid_elements = [e for e in seq if not isinstance(e, ele_type)]
            if invalid_elements:
                raise TypeError(
                    f"All elements of {name} must be of type, "
                    f"got different types for {invalid_elements}"
                )

        if isinstance(req_elements, Collection):
            missing_elements = [e for e in req_elements if e not in seq]
            if missing_elements:
                raise ValueError(
                    f"It is expected sequence {name}, contains elements "
                    f"{req_elements}, but '{missing_elements}' are missing."
                )

        return seq

    @staticmethod
    def as_float(
        v: Any,
        low: float | None = None,
        high: float | None = None,
        name: str | None = None,
        *,
        closed_bounds: bool = True,
        allow_none: bool = False,
    ) -> float | None:
        if allow_none and v is None:
            return None
        name = "" if name is None else name
        Validator.is_type(v, (int, float), name=name, exclusion_types=bool)
        v = float(v)
        Validator.in_bounds(v, low, high, name, closed_bounds)
        return v

    @staticmethod
    def as_int(
        v: Any,
        low: float | None = None,
        high: float | None = None,
        name: str | None = None,
        *,
        closed_bounds: bool = True,
        allow_none: bool = False,
    ) -> int | None:
        """Return the validated int"""
        if allow_none and v is None:
            return None
        name = "" if name is None else name
        Validator.is_type(v, int, name=name, exclusion_types=bool)
        Validator.in_bounds(v, low, high, name, closed_bounds)
        return v

    @staticmethod
    def as_string(
        v: Any,
        *,
        name: str | None = None,
        target: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        allow_none: bool = False,
    ) -> str | None:
        """Return the validated string."""
        if allow_none and v is None:
            return None
        name = "" if name is None else name

        Validator.is_type(v, str, name=name)
        if Validator.is_type(target, str) and v != target:
            raise ValueError(
                f"The given string '{name}' does not match the target string {target}"
            )

        v_len = len(v)
        if Validator.is_type(min_length, int) and v_len < min_length:
            raise ValueError(
                f"The given string '{name}' is too short {v_len} "
                f"(minimum length: {min_length})"
            )

        if Validator.is_type(max_length, int) and v_len > max_length:
            raise ValueError(
                f"The given string '{name}' is tool long {v_len} "
                f"(maximum length {max_length})"
            )

        return v

    @staticmethod
    def as_dict(
        d: Any,
        keys: list[Any] | None = None,
        types: list[type | tuple] | None = None,
        name: str | None = None,
        *,
        key_type_map: Mapping | None = None,
        reject_extra_keys: bool = False,
        allow_none: bool = False,
    ) -> dict | None:
        if allow_none and d is None:
            return None
        Validator.is_type(d, dict, name=name)

        if key_type_map is not None and (
            keys is not None or types is not None
        ):
            raise ValueError(
                "When key_type_map is provided, keys and types should not be provided."
            )

        if key_type_map is not None:
            keys = list(key_type_map.keys())
            types = list(key_type_map.values())

        if keys is not None:
            Validator.is_type(keys, list, name=name)
            missing_keys = [k for k in keys if k not in d]
            if missing_keys:
                raise ValueError(f"In {name}, Missing Keys: {missing_keys}")
            if reject_extra_keys:
                extra_keys = [k for k in d if k not in keys]
                if len(extra_keys) > 0:
                    raise ValueError(f"In {name}, Extra Keys: {extra_keys}")

            if types is not None:
                Validator.is_type(types, list, name=name)

                if len(keys) != len(types):
                    raise ValueError(
                        "When types and keys are specified, they must have "
                        f"the same length, but got {len(keys)} and {len(types)}"
                    )

                for k, t in zip(keys, types):
                    if t is not None:
                        Validator.is_type(d[k], t, name=f"{name}.{k}")
        else:
            if types is not None:
                raise ValueError(
                    "When types is specified, keys must also be specified"
                )
            if reject_extra_keys:
                raise ValueError(
                    "When reject_extra_keys is True, keys must also be provided"
                )
        return d

    @staticmethod
    def file_path(
        file_path: Any,
        must_exist: bool | None = None,
        extensions: list[str] | None = None,
        *,
        create_parent: bool = False,
        resolve_path: bool = True,
    ) -> Path:
        """Returns validated file path."""
        Validator.is_type(file_path, (str, Path), name="file_path")
        fp = Path(file_path).resolve() if resolve_path else Path(file_path)
        if must_exist is not None:
            if must_exist and not fp.exists():
                raise FileNotFoundError(f"File '{fp}' does not exist.")
            elif must_exist is False and fp.exists():
                raise FileExistsError(f"File '{fp}' already exists.")
        if extensions is not None:
            Validator.as_sequence(
                extensions,
                ele_type=str,
                req_elements=(fp.suffix,),
            )
        if create_parent:
            fp.parent.mkdir(parents=True, exist_ok=True)
        return fp

    @staticmethod
    def dir_path(
        dir_path: Any,
        must_exist: bool | None = None,
        *,
        mkdir: bool = False,
        resolve_path: bool = True,
    ) -> Path:
        """Returns validated directory path."""
        Validator.is_type(dir_path, (str, Path), name="dir_path")
        dp = Path(dir_path).resolve() if resolve_path else Path(dir_path)

        if must_exist is not None:
            if must_exist and not dp.exists():
                raise FileNotFoundError(f"Directory '{dp}' does not exist.")
            elif must_exist and not dp.is_dir():
                raise NotADirectoryError(f"Path '{dp}' is not a directory.")
            elif must_exist is False and dp.exists():
                raise FileExistsError(f"Directory '{dp}' already exists.")
        if mkdir:
            dp.mkdir(parents=True, exist_ok=True)
        return dp
