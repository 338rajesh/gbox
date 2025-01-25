# gbox:: Geometry Box

[![Documentation Status](https://readthedocs.org/projects/gbox/badge/?version=latest)](https://gbox.readthedocs.io/en/latest/?badge=latest)

A simple Python package for working with geometry related operations.
For the extensive documentation, see [gbox.readthedocs.io](https://gbox.readthedocs.io)

## Notes

* Angles are in radians
* In `m` dimensional space, a group of `n` Points are represented
    as a `m x n` matrix. That is each column is a point in `m` dimensions
    and each row represents a single dimension.
* Starting with np.float32 for all computations.
  * Later it should be made possible to use float or np.float64
* Perform all operations in np.float32 unless otherwise specified
* Perform in-place operations unless otherwise specified
  