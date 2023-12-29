gbox :: Geometry Box
=====================

**gbox** is a simple and small python package intended to perform mathematical operations on elementary shapes.

.. image:: ../media/shapes.png

A sample collection of shapes plotted using this library

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   usage
   generated/gbox
   generated/gbox.points
   generated/gbox.curves
   generated/gbox.closed_shapes
   generated/gbox.utils
   generated/unit_cell


WIP/Goals
^^^^^^^^^

  + At present Only `Circle` and `Ellipse` contain the specified number of points on the locus.
    As, other shapes are made up of multiple elements, needs to find the right number of points
    on each element of the shape.
  + Convert `Points(list)` to `Points(numpy.ndarray)` and see if there is any performance gain
  + Updating the documentation