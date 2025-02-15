import numpy as np
from typing import Literal, get_type_hints, get_args
import inspect


# ============================================================================
#                     Type Class
# ============================================================================


class _Type:
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
        """The dtype of the type"""
        return self._type_

    def __eq__(self, other):
        return isinstance(other, _Type) and self.dtype == other.dtype

    def _check_validity(self, val):
        if not isinstance(
            val,
            (np.ndarray, list, tuple, int, float)
            + _Type.NUMPY_FLOAT
            + _Type.NUMPY_INT
            + _Type.NUMPY_UINT,
        ):
            raise NotImplementedError(
                f"Unsupported type: {type(val)}. "
                f"Only scalar types {self._types_}, numpy arrays, "
                f"and nested lists/tuples of these types are supported."
            )

    def __call__(self, val):
        """Converts a value to the dtype of the type. The value can be a
        scalar value, a numpy array, or an arbitrarily nested list/tuple.
        """
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


class RealNumType(_Type):
    """Real number type, defaults to np.float32
    Real number can be int, float, numpy int16, numpy float32, numpy float64,
    numpy int8, numpy int16, numpy int32, numpy int64, numpy uint8,
    numpy uint16, numpy uint32, numpy uint64. These list can accessed with
    `RealNumType._types_`.
    """

    DEFAULT = np.float32
    _types_ = (int, float) + _Type.NUMPY_FLOAT + _Type.NUMPY_INT + _Type.NUMPY_UINT

    def __init__(self, dtype=None):
        super().__init__(dtype or RealNumType.DEFAULT)


class IntType(RealNumType):
    """Integer type, defaults to np.int32
    Integer can be int, numpy int8, numpy int16, numpy int32, numpy int64,
    numpy uint8, numpy uint16, numpy uint32, numpy uint64. These list can
    accessed with `IntType._types_`.
    """

    DEFAULT = np.int32
    _types_ = (int,) + _Type.NUMPY_INT + _Type.NUMPY_UINT

    def __init__(self, dtype=None):
        super().__init__(dtype or IntType.DEFAULT)


class FloatType(RealNumType):
    """Float type, defaults to np.float32
    Float can be float, numpy float32, numpy float64. These list can
    accessed with `FloatType._types_`.
    """

    DEFAULT = np.float32
    _types_ = (float,) + _Type.NUMPY_FLOAT

    @property
    def precision(self):
        return np.finfo(self.dtype).resolution

    def __init__(self, dtype=None):
        super().__init__(dtype or FloatType.DEFAULT)


class TypeConfig:
    """This class is used to configure various types.

    Attributes
    ----------
    float_type : FloatType
        Float type, defaults to np.float32
    int_type : IntType
        Integer type, defaults to np.int32
    real_type : RealNumType
        Real number type, defaults to np.float32
    float_precision : float
        Precision of the current float type

    Methods
    -------
    set_float_type(dtype)
        Sets the float type
    set_int_type(dtype)
        Sets the integer type
    set_real_num_type(dtype)
        Sets the real number type

    Examples
    --------
    >>> TypeConfig.set_float_type(np.float64)
    >>> TypeConfig.set_int_type(np.int64)
    >>> TypeConfig.set_real_num_type(np.float64)
    >>> TypeConfig.float_precision()
    1e-15
    >>> TypeConfig.float_type().dtype
    <class 'numpy.float64'>
    """

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


# _valid_type_tags = Literal["float", "int", "real", "np_float"]
def get_current_func_info():
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    module = inspect.getmodule(frame)
    func_object = getattr(module, function_name)
    if func_object is None:
        raise ValueError(f"Function {function_name} not found in module {module}")
    return {
        "function_name": function_name,
        "module": module,
        "type_hints": get_type_hints(func_object),
    }


def get_type(tag: Literal["float", "int", "real", "np_float"]) -> type:
    """Returns the type corresponding to the tag

    Parameters
    ----------
    tag: Literal["float", "int", "real", "np_float"]
        The type tag

    Returns
    -------
    type
        The current configured type corresponding to the tag
    """
    tag_types = get_args(get_current_func_info()["type_hints"]["tag"])
    if tag not in tag_types:
        raise ValueError(f"Invalid tag: {tag}, should be one of {tag_types}")
    if tag == "float":
        return TypeConfig.float_type().dtype
    elif tag == "int":
        return TypeConfig.int_type().dtype
    elif tag == "real":
        return TypeConfig.real_type().dtype
    elif tag == "np_float":
        return np.dtype(TypeConfig.float_type().dtype)


def cast_to(v, tag):
    """
    Casts the value to the type corresponding to the tag

    Parameters
    ----------
    v : Any
        The value to cast
    tag : Literal["float", "int", "real", "np_float"]
        The type tag

    Returns
    -------
    Any
        The casted value
    """
    return get_type(tag)(v)
