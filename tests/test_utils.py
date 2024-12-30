import copy

import pytest
from gbox.utilities import (
    rotation_matrix_2d,
    SEQUENCE,
    REAL_NUMBER,
    validated_num_cores,
    cpu_count,
)
import numpy as np

from hypothesis import given, strategies as st


class TestTypeAssertions:
    def test_seq_type(self):
        assert list in SEQUENCE
        assert tuple in SEQUENCE
        assert isinstance(-1.0, REAL_NUMBER)
        assert isinstance(1.0, REAL_NUMBER)
        assert isinstance(1, REAL_NUMBER)
        assert isinstance(-1, REAL_NUMBER)
        assert isinstance([1.0, 2.0], SEQUENCE)
        assert isinstance((1.0, 2.0), SEQUENCE)
        with pytest.raises(AssertionError):
            assert isinstance("1", REAL_NUMBER)
        with pytest.raises(AssertionError):
            assert isinstance(1.0, SEQUENCE)
        with pytest.raises(AssertionError):
            assert isinstance({1.0, 2.0}, SEQUENCE)
        with pytest.raises(AssertionError):
            assert isinstance({1.0: 2.0}, SEQUENCE)


@given(st.floats(min_value=-np.pi, max_value=np.pi))
def test_rotation_matrix_properties(angle):
    rot_matrix_2d = rotation_matrix_2d(angle)
    identity_matrix = np.eye(2)
    assert np.allclose(rot_matrix_2d.T @ rot_matrix_2d, identity_matrix)
    assert pytest.approx(np.linalg.det(rot_matrix_2d)) == 1.0

    with pytest.raises(AssertionError):
        rotation_matrix_2d(45, unit="degg")


def test_number_of_cores():
    assert cpu_count() > 0
    assert validated_num_cores(1) == 1
    assert validated_num_cores(cpu_count() + 1) == cpu_count()
