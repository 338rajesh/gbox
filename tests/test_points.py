import numpy as np
import matplotlib.pyplot as plt
import os
from geometry_box.points import Points
from geometry_box.shapes import BoundingBox

HOME_DIR = os.path.expanduser("~")
RES_DIR = os.path.join(HOME_DIR, "geometry_box_test")
os.makedirs(RES_DIR, exist_ok=True)

xy_points = np.random.rand(100, 3)
pc = Points(xy_points)
bb = BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
print(bb.dim)
tiled_pc = pc.make_periodic_tiles(bb)
print(tiled_pc.shape)
