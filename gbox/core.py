import numpy as np
import operator
from typing import Literal

# ============================================================================
#                     Module level definitions
# ============================================================================

operators = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "floor_div": operator.floordiv,
    "eq": operator.eq,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "ne": operator.ne,
    "and": operator.and_,
    "or": operator.or_,
    "not": operator.not_,
}


# ============================================================================
#                     Type Class
# ============================================================================


class Type:

    NUMPY_FLOAT = (np.float16, np.float32, np.float64)
    NUMPY_INT = (np.int8, np.int16, np.int32, np.int64)
    NUMPY_UINT = (np.uint8, np.uint16, np.uint32, np.uint64)

    def __init__(self, dtype):

        if dtype not in self._types_:
            raise TypeError(
                f"Unsupported dtype: {dtype}, consider one of {self._types_}"
            )

        self._type_ = dtype

    @property
    def dtype(self):
        return self._type_

    def __eq__(self, other):
        return isinstance(other, Type) and self.dtype == other.dtype

    def _check_validity(self, val):
        if not isinstance(
            val,
            (np.ndarray, list, tuple, int, float)
            + Type.NUMPY_FLOAT
            + Type.NUMPY_INT
            + Type.NUMPY_UINT,
        ):
            raise NotImplementedError(
                f"Unsupported type: {type(val)}. "
                f"Only scalar types {self._types_}, numpy arrays, "
                f"and nested lists/tuples of these types are supported."
            )

    def __call__(self, val):
        self._check_validity(val)
        dtype = self.dtype

        if isinstance(val, np.ndarray):
            return np.asarray(val, dtype=dtype)

        elif isinstance(val, (list, tuple)):

            def convert_nested(obj):

                if isinstance(obj, (list, tuple)):
                    return type(obj)(convert_nested(v) for v in obj)

                return dtype(obj) if obj is not None else None

            return convert_nested(val)

        return dtype(val)


class RealNumType(Type):
    DEFAULT = np.float32
    _types_ = (int, float) + Type.NUMPY_FLOAT + Type.NUMPY_INT + Type.NUMPY_UINT

    def __init__(self, dtype=None):
        super().__init__(dtype or RealNumType.DEFAULT)


class IntType(RealNumType):
    DEFAULT = np.int32
    _types_ = (int,) + Type.NUMPY_INT + Type.NUMPY_UINT

    def __init__(self, dtype=None):

        super().__init__(dtype or IntType.DEFAULT)


class FloatType(RealNumType):
    DEFAULT = np.float32
    _types_ = (float,) + Type.NUMPY_FLOAT

    @property
    def precision(self):
        return np.finfo(self.dtype).resolution

    def __init__(self, dtype=None):
        super().__init__(dtype or FloatType.DEFAULT)


class TypeConfig:
    _float_type_ = FloatType()
    _int_type_ = IntType()
    _real_num_type_ = RealNumType()

    @classmethod
    def float_type(cls):
        return cls._float_type_

    @classmethod
    def int_type(cls):
        return cls._int_type_

    @classmethod
    def real_type(cls):
        return cls._real_num_type_

    @classmethod
    def set_float_type(cls, dtype):
        cls._float_type_ = FloatType(dtype)

    @classmethod
    def set_int_type(cls, dtype):
        cls._int_type_ = IntType(dtype)

    @classmethod
    def set_real_num_type(cls, dtype):
        cls._real_num_type_ = RealNumType(dtype)

    @classmethod
    def float_precision(cls):
        return cls._float_type_.precision


_valid_type_tags = Literal["float", "int", "real", "np_float"]


def get_type(tag: _valid_type_tags) -> type:
    if tag not in _valid_type_tags.__args__:
        raise ValueError(
            f"Invalid tag: {tag}, valid tags are {_valid_type_tags.__args__}"
        )
    if tag == "float":
        return TypeConfig.float_type().dtype
    elif tag == "int":
        return TypeConfig.int_type().dtype
    elif tag == "real":
        return TypeConfig.real_type().dtype
    elif tag == "np_float":
        return np.dtype(TypeConfig.float_type().dtype)


def cast_to(v, tag):
    return get_type(tag)(v)
