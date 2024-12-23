import pytest
from gbox import utilities as utils
import numpy as np

# ==================== FIXTURES =====================


@pytest.fixture(params=[np.pi * i * 0.1 for i in range(6)])
def rotation_matrix_2d(request):
    return utils.rotation_matrix_2d(request.param)


# ==================== TESTS =====================


class TestTypeAssertions:
    def test_seq_type(self):
        assert list in utils.SEQUENCE
        assert tuple in utils.SEQUENCE
        assert isinstance([1.0, 2.0], utils.SEQUENCE)
        assert isinstance((1.0, 2.0), utils.SEQUENCE)
        with pytest.raises(AssertionError):
            assert isinstance(1.0, utils.SEQUENCE)
        with pytest.raises(AssertionError):
            assert isinstance({1.0, 2.0}, utils.SEQUENCE)
        with pytest.raises(AssertionError):
            assert isinstance({1.0: 2.0}, utils.SEQUENCE)


class TestAssertion:
    def test_equality(self):
        utils.Assert(1, 1).equal()
        utils.Assert([2, 3], [2, 3]).equal()
        with pytest.raises(AssertionError):
            utils.Assert(1, 2).equal()

    def test_equal_lenths(self):
        utils.Assert([1, 2], [1, 2]).have_equal_lenths()
        with pytest.raises(AssertionError):
            utils.Assert([1, 2], [1, 2, 3]).have_equal_lenths()

    def test_lt(self):
        utils.Assert([1, 2]).lt([2, 3])
        with pytest.raises(AssertionError):
            utils.Assert([2, 3]).lt([1, 2])

    def test_le(self):
        utils.Assert([1, 2]).le([1, 2])
        with pytest.raises(AssertionError):
            utils.Assert([2, 3]).le([2, 1])

    def test_ge(self):
        utils.Assert([1, 2]).ge([1, 2])
        with pytest.raises(AssertionError):
            utils.Assert([2, 1]).ge([2, 13])

    def test_gt(self):
        utils.Assert([1, 2]).gt([0, 0])
        with pytest.raises(AssertionError):
            utils.Assert([0, 1]).gt([2, 13])

    def test_eq(self):
        utils.Assert([1, 2.0]).eq([1, 2.0])
        with pytest.raises(AssertionError):
            utils.Assert([2, 1]).eq([2, 13])

    def test_between(self):
        utils.Assert(np.pi * 0.5).between(0.0, np.pi, "Failed Between Test")
        utils.Assert(1, 2.0, 3.0, 5.0, 10.0).between(-1.0, 11.0)
        with pytest.raises(AssertionError):
            utils.Assert(0.5).between(4.0, 0.0)  # order of min and max
        with pytest.raises(AssertionError):
            utils.Assert(1, 2.0, 3.0, 5.0, 10.0).between(4.0, 11.0)

    def test_of_type(self):
        utils.Assert(1, 2).of_type(int)
        utils.Assert((1, 2), (3, 4)).of_type(tuple)
        with pytest.raises(AssertionError):
            utils.Assert(1.0, 2).of_type(int)
        with pytest.raises(AssertionError):
            utils.Assert((1, 2), (3, 4)).of_type(list)

    def test_seq_of(self):
        utils.Assert([1, 2]).are_seq(utils.REAL_NUMBER)
        utils.Assert((1.0, 2.0)).are_seq(utils.REAL_NUMBER)
        with pytest.raises(AssertionError):
            utils.Assert([1, 2]).are_seq(float)
        with pytest.raises(AssertionError):
            utils.Assert((1.0, 2.0)).are_seq(int)

    def test_seq_of_seq(self):
        utils.Assert([[1, 2], [3, 4]]).are_seq_of_seq(int)
        with pytest.raises(AssertionError):
            utils.Assert([[1, 2], [3, 4]]).are_seq_of_seq(float)


# ------------------------------------------------


def test_rotation_matrix_properties(rotation_matrix_2d):
    identity_matrix = np.eye(2)
    assert np.allclose(rotation_matrix_2d.T @ rotation_matrix_2d, identity_matrix)
    assert pytest.approx(np.linalg.det(rotation_matrix_2d)) == 1.0

    with pytest.raises(AssertionError):
        utils.rotation_matrix_2d(45, unit="degg")


def test_number_of_cores():
    assert utils.cpu_count() > 0
    assert utils.validated_num_cores(1) == 1
    assert utils.validated_num_cores(utils.cpu_count() + 1) == utils.cpu_count()


def test_make_iterators_if_not():
    assert utils.make_sequence_if_not(1, 2, 3) == [[1], [2], [3]]
    assert utils.make_sequence_if_not([1, 2, 3], [4, 5]) == [[1, 2, 3], [4, 5]]
