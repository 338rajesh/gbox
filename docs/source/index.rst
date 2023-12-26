gbox: Geometry Box documentation
================================

**gbox** is a simple and small python package intended to perform elementary geometry operations.

Contents
--------

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   usage
   gbox


WIP/Goals
^^^^^^^^^

  + At present Only `Circle` and `Ellipse` contain the specified number of points on the locus.
    As, other shapes are made up of multiple elements, needs to find the right number of points
    on each element of the shape.
  + Convert `Points(list)` to `Points(numpy.ndarray)` and see if there is any performance gain
  + Updating the documentation