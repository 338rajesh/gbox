import unittest

from numpy import pi

import gbox as gb


class TestEllipticalArc(unittest.TestCase):
    e1 = gb.EllipticalArc(2.0, 1.0, 1.5 * pi)

    def test_default_num_locus_points(self):
        assert self.e1.locus.points.shape == (100, 2)

    def test_setting_num_locus_points(self):
        self.e1.num_locus_points = 20
        assert self.e1.locus.points.shape == (20, 2)
