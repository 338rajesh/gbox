import pytest
import numpy as np
from gbox import TypeConfig
from gbox.core import IntType, FloatType


def test_construction():

    Int32 = IntType(np.int32)
    Float64 = FloatType(np.float64)

    assert isinstance(Int32, IntType)
    assert isinstance(Float64, FloatType)
    assert Int32.dtype == np.int32
    assert Float64.dtype == np.float64

    with pytest.raises(TypeError):
        IntType(np.float64)
    with pytest.raises(TypeError):
        FloatType(np.int32)


@pytest.mark.parametrize("cls", [IntType, FloatType])
@pytest.mark.parametrize("val_type", [int, float, np.int32, np.float64])
def test_type_conversion(cls, val_type):
    for type_ in cls._types_:
        mn = [val_type(1.0), val_type(6.0)]

        tt = cls(type_)

        for a in (mn[0], mn, np.array(mn, dtype=np.float64), tuple(mn)):
            a = tt(a)
            if isinstance(a, np.ndarray):
                assert a.dtype == type_
            elif isinstance(a, (list, tuple)):
                assert all(isinstance(x, type_) for x in a)
            else:
                assert isinstance(a, type_)


def test_default_type():
    assert IntType().dtype == IntType.DEFAULT, (
        "IntType should have a default type"
    )
    assert (
        FloatType().dtype == FloatType.DEFAULT
    ), "FloatType should have a default type"


@pytest.mark.parametrize("cls", [IntType, FloatType])
@pytest.mark.parametrize(
    "invalid_val",
    [
        "string",  # String input
        {"key": "value"},  # Dictionary
        {1, 2, 3},  # Set
        None,  # NoneType
    ],
)
def test_invalid_input(cls, invalid_val):
    with pytest.raises(NotImplementedError):
        cls()(invalid_val)


def test_type_config():
    TypeConfig.set_int_type(np.int16)
    TypeConfig.set_float_type(np.float64)

    assert TypeConfig.int_type() == IntType(np.int16)
    assert TypeConfig.float_type() == FloatType(np.float64)

    float_type = TypeConfig.float_type()
    a = np.float32(1.0)
    b = float_type(a)
    assert type(b) is float_type.dtype
