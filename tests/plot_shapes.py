import gbox as gb
from os import path, makedirs
from numpy import array, pi
from matplotlib.pyplot import subplots, savefig

gb.PLOT_OPTIONS.face_color = 'r'
gb.PLOT_OPTIONS.edge_color = 'b'
gb.PLOT_OPTIONS.linewidth = 2.0
gb.PLOT_OPTIONS.hide_axes = True
gb.PLOT_OPTIONS.show_grid = True

PLOT_DIR = path.join(path.dirname(__file__), "_plots")
makedirs(PLOT_DIR, exist_ok=True)

# Circle
gb.Circle().plot(f_path=path.join(PLOT_DIR, "circle.png"))

# Ellipse
gb.Ellipse().plot(f_path=path.join(PLOT_DIR, "ellipse.png"))

# Rectangle
gb.Rectangle(rc=0.25).plot(f_path=path.join(PLOT_DIR, "rectangle.png"))

# Capsule
gb.Capsule().plot(f_path=path.join(PLOT_DIR, "capsule.png"))

# Regular Polygon
gb.RegularPolygon(3).plot(f_path=path.join(PLOT_DIR, "r_polygon.png"))

# BoundingBox2D
gb.BoundingBox2D(-1, -1, 1, 1).plot(f_path=path.join(PLOT_DIR, "b_box.png"))

# C-SHAPE
gb.CShape(theta_c=1.25 * pi).plot(f_path=path.join(PLOT_DIR, "c_shape.png"))

# N-LOBE-SHAPE
gb.NLobeShape(3, 2.0, ld_factor=0.95).plot(f_path=path.join(PLOT_DIR, "n_lobe.png"))

# N-TIP STAR

gb.PLOT_OPTIONS.face_color = 'g'
gb.PLOT_OPTIONS.edge_color = 'b'

circle_data = array([[0.0, 0.0, 2.0], [5.0, 5.0, 2.5]])
capsule_data = array([[-5.0, 5.0, 0.3 * pi, 3.0, 1.0]])
nrp_data = array([[5.0, -5.0, 0.0 * pi, 3.0, 0.5, 5]])

fig, axs = subplots()
#
shapes = gb.ShapesList()
shapes.extend(gb.Circles(circle_data))
shapes.extend(gb.Capsules(capsule_data))
shapes.extend(gb.RegularPolygons(nrp_data))
shapes.plot(axis=axs, linewidth=1.0)
#
savefig(path.join(PLOT_DIR, "shapes.png"))
