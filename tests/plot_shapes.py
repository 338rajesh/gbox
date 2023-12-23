import geometry_box as gb
from os import path, makedirs
from math import pi

gb.PLOT_OPTIONS.face_color = 'b'
gb.PLOT_OPTIONS.edge_color = 'r'
gb.PLOT_OPTIONS.linewidth = 2.0
gb.PLOT_OPTIONS.hide_axes = True
gb.PLOT_OPTIONS.show_grid = True

PLOT_DIR = path.join(path.dirname(__file__), "_plots")
makedirs(PLOT_DIR, exist_ok=True)

# Circle
gb.Circle().eval_locus().plot(f_path=path.join(PLOT_DIR, "circle.png"))

# Ellipse
gb.Ellipse().eval_locus().plot(f_path=path.join(PLOT_DIR, "ellipse.png"))

# Rectangle
gb.Rectangle().eval_locus().plot(f_path=path.join(PLOT_DIR, "rectangle.png"))

# Capsule
gb.Capsule().eval_locus(num_points=10).plot(f_path=path.join(PLOT_DIR, "capsule.png"))

# Regular Polygon
gb.RegularPolygon(5).eval_locus().plot(f_path=path.join(PLOT_DIR, "r_polygon.png"))

# BoundingBox2D
gb.BoundingBox2D(-1, -1, 1, 1).plot(f_path=path.join(PLOT_DIR, "b_box.png"))

# C-SHAPE
gb.CShape(theta_c=1.25 * pi).eval_locus(50, pivot_angle=1.5 * pi).plot(f_path=path.join(PLOT_DIR, "c_shape.png"))

# N-LOBE-SHAPE
gb.NLobeShape(3, 2.0, ld_factor=0.95).eval_locus().plot(f_path=path.join(PLOT_DIR, "n_lobe.png"))

# N-TIP STAR
