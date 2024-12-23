from contextlib import contextmanager
import pathlib

from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

SEQUENCE = (list, tuple)
REAL_NUMBER = (int, float)


def rotation_matrix_2d(angle: float, unit="rad") -> np.ndarray:
    """Returns the rotational matrix for a given angle."""
    if unit == "deg":
        angle *= np.pi / 180
    else:
        assert unit == "rad", "Unit must be 'deg' or 'rad'"

    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def validated_num_cores(n):
    assert isinstance(n, int), "Number of cores must be an integer"
    if n > cpu_count():
        print("Given number of cores greater than available, setting to maximum.")
        n = cpu_count()
    return n


def make_sequence_if_not(*args):
    return [a if isinstance(a, SEQUENCE) else [a] for a in args]


class Assert:
    def __init__(self, *args, err_msg=None):
        self.args = args
        self.types = [type(a) for a in args]
        # self.lens = [len(a) for a in args]
        self.err_msg = err_msg

    # ==================================================================
    # ===                   Assertions for type checking             ===
    # ==================================================================

    @staticmethod
    def assert_same_type_of_a_seq(a, type_):
        assert all(isinstance(i, type_) for i in a)

    def of_type(self, type_, err_msg=None):
        self.assert_same_type_of_a_seq(self.args, type_)

    def are_seq(self, type_=None, err_msg=None):
        self.assert_same_type_of_a_seq(self.args, SEQUENCE)
        if type_ is not None:
            for i in self.args:
                self.assert_same_type_of_a_seq(i, type_)
        return True

    def are_seq_of_seq(self, type_=None, err_msg=None):
        self.are_seq(SEQUENCE, err_msg)
        if type_ is not None:
            for i in self.args:
                for j in i:
                    self.assert_same_type_of_a_seq(j, type_)
        return True

    # -----------------------------------------------------------------

    def equal(self, err_msg=None):
        if not all(i == self.args[0] for i in self.args[1:]):
            raise AssertionError(f"Assertion Error: {self.err_msg or err_msg}")
        return self

    def have_equal_lenths(self, err_msg=None):
        assert self.are_seq(), f"Given elements are not sequences"
        if len(set(len(i) for i in self.args)) > 1:
            raise AssertionError(f"Assertion Error: {self.err_msg or err_msg}")

    def _compare(self, *b, key="eq", err_msg=None):
        a, b = make_sequence_if_not(self.args, b)

        op = {
            "lt": lambda x, y: x < y,
            "le": lambda x, y: x <= y,
            "gt": lambda x, y: x > y,
            "ge": lambda x, y: x >= y,
            "eq": lambda x, y: x == y,
        }.get(key)
        if not all(op(i, j) for i, j in zip(a, b)):
            raise AssertionError(f"Assertion Error: {self.err_msg or err_msg}")

    def lt(self, *b, err_msg=None):
        self._compare(*b, key="lt", err_msg=err_msg)

    def le(self, *b, err_msg=None):
        self._compare(*b, key="le", err_msg=err_msg)

    def gt(self, *b, err_msg=None):
        self._compare(*b, key="gt", err_msg=err_msg)

    def ge(self, *b, err_msg=None):
        self._compare(*b, key="ge", err_msg=err_msg)

    def eq(self, *b, err_msg=None):
        self._compare(*b, key="eq", err_msg=err_msg)

    def between(self, min_, max_, err_msg=None):
        Assert(min_, max_).of_type(REAL_NUMBER)
        Assert(min_).le(max_)
        for i in self.args:
            if i <= min_ or i >= max_:
                raise AssertionError(f"Assertion Error: {self.err_msg or err_msg}")



@contextmanager
def plot_context(file_path=None, **kwargs):

    fig, axs = plt.subplots(1, 1)

    try:
        yield fig, axs
    finally:
        fig.savefig(file_path, **kwargs)
        assert file_path.is_file(), f"File {file_path} is not saved!"
        plt.close(fig)
