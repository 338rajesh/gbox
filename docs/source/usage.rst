==========
Tutorial
==========

.. _installation:

Installation
------------

To use the gbox, first install it using pip:

.. code-block:: console

   pip install gbox


Creating Closed Shapes
----------------------

.. code-block:: python

    import gbox as gb
    from os import path
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4)
    gb.Circle().plot(axis=axs[0, 0])
    gb.Ellipse().plot(axis=axs[0, 1])
    gb.RegularPolygon(3).plot(axis=axs[0, 2])
    gb.Rectangle().plot(axis=axs[0, 3])
    gb.BoundingBox2D().plot(axis=axs[1, 0])
    gb.CShape().plot(axis=axs[1, 1])
    gb.Capsule().plot(axis=axs[1, 2])
    gb.NLobeShape(3).plot(axis=axs[1, 3])
    plt.tight_layout()
    plt.savefig(path.join(path.dirname(__file__), "shapes.pdf"))
    plt.close()

It produces the following figure

.. image:: _static/shapes.png

Applications
*************

It contains standalone use cases which might be too little to maintain separately and depend heavily on this gbox
package. It currently supports unit cell application and open to add new cases in near future.


.. toctree::
   :maxdepth: 3
   :caption: List of applications

   tutorial/unit_cell_2d
   tutorial/unit_cell_3d

