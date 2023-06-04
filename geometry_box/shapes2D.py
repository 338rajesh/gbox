import numpy as np


class Shape2D:
    def __init__(self, locus=None):
        self.locus = locus
        return

    # @property
    # def area(self):
    #     return
    #
    # @area.setter
    # def area(self, value):
    #     self.area = value
    #
    # @property
    # def perimeter(self):
    #     return 1.0
    #
    # @perimeter.setter
    # def perimeter(self, value):
    #     self.perimeter = value
    #
    # def shape_factor(self):
    #     return self.perimeter / (np.sqrt(4.0 * np.pi * self.area))


class Ellipse(Shape2D):
    def __init__(self, smj=2.0, smn=1.0, cent=(0.0, 0.0), locus=None):
        assert smj >= smn, f"Requires semi major axis > semi minor axis but found {smj} < {smn}"
        self.smj = smj
        self.smn = smn
        self.cent = cent
        self.locus = locus
        super(Ellipse, self).__init__(locus)
        return

    def perimeter(self, method="Ramanujan"):
        if method == "Ramanujan":
            return np.pi * (
                    (3.0 * (self.smj + self.smn)) - np.sqrt(
                ((3.0 * self.smj) + self.smn) * (self.smj + (3.0 * self.smn))))
            # super(Ellipse, self).perimeter

    def area(self):
        return np.pi * self.smj * self.smn
        # return super(Ellipse, self).area

    def shape_factor(self, p=None, a=None):
        if a is None and p is None:
            p, a = self.perimeter(), self.area()
        return p / np.sqrt(4.0 * np.pi * a)

    def make_sector(self,
                    theta_1=0.0,
                    theta_2=np.pi,
                    num_sec_points=100,
                    direct_loop_closure=False):
        """

        :param float theta_1: starting angle of the sector, in radians
        :param float theta_2: ending angle of the sector, in radians
        :param num_sec_points:
        :param bool direct_loop_closure: True,
        :return: xy
        :rtype np.ndarray:
        """
        #
        theta_i = np.linspace(start=theta_1, stop=theta_2, num=num_sec_points)
        x_y = np.column_stack([self.smj * np.cos(theta_i), self.smn * np.sin(theta_i)])
        if direct_loop_closure:
            x_y = np.append(x_y, [x_y[0]], axis=0)
        else:
            x_y = np.append(x_y, [[0.0, 0.0], x_y[0]], axis=0)
        #
        self.locus = x_y
        # returning rotated and translated reference-elliptical sector
        return self


class Circle(Ellipse):
    def __init__(self, radius=2.0, cent=(0.0, 0.0)):
        super().__init__(radius, radius, cent)
        return


class RegularPolygon(Shape2D):
    def __init__(self, num_sides: int, crr: float,
                 side_len: float = None,
                 eqr: float = None,
                 cent=(0.0, 0.0),
                 locus=None,
                 ):
        assert 0.0 <= crr <= 1.0, "Corner radius ratio must be between 0[Perfect Triangle] and 1[Perfect circle]"
        assert num_sides > 2, "Number of sides should be integer and greater than 2"
        #
        self.num_sides = num_sides
        self.alpha = np.pi / self.num_sides
        self.crr = crr

        if side_len is None:  # finding the side_length from equivalent radius, if not provided.
            if eqr is None:
                raise ValueError("Side length and Equivalent radii cannot be None, simultaneously.")
            else:
                side_len = eqr * np.sqrt((4.0 * self.alpha * np.tan(self.alpha)) / (
                        1.0 - ((self.crr ** 2) * (1.0 - self.alpha / np.tan(self.alpha)))))
        self.side_len = side_len
        self.cr = self.crr * self.side_len / (2.0 * np.tan(self.alpha))
        #
        self.locus = None
        super(RegularPolygon, self).__init__(locus)
        return

    def perimeter(self):
        return self.num_sides * self.side_len * (
                1.0 - self.crr + (self.crr * self.alpha * np.cos(self.alpha) / np.sin(self.alpha))
        )

    def area(self):
        cot_alpha = np.cos(self.alpha) / np.sin(self.alpha)
        return 0.25 * self.num_sides * self.side_len * self.side_len * cot_alpha * (
                1.0 - ((self.crr * self.crr) * (1.0 - (self.alpha * cot_alpha)))
        )

    def shape_factor(self, p=None, a=None):
        if a is None and p is None:
            p, a = self.perimeter(), self.area()
        return p / np.sqrt(4.0 * np.pi * a)


class Rectangle(Shape2D):
    def __init__(self, smj=2.0, smn=1.0, cent=(0.0, 0.0), rc: float = 0.0, locus=None):
        assert smj >= smn, f"Requires semi major axis > semi minor axis but found {smj} < {smn}"
        self.smj = smj
        self.smn = smn
        self.rc = rc
        self.cent = cent
        self.locus = locus
        super(Rectangle, self).__init__(locus)
        return

    def perimeter(self):
        return 4 * (self.smj + self.smn) - (2.0 * (4.0 - np.pi) * self.rc)

    def area(self):
        return (4.0 * self.smj * self.smn) - ((4.0 - np.pi) * self.rc * self.rc)
        # return super(Ellipse, self).area

    def shape_factor(self, p=None, a=None):
        if a is None and p is None:
            p, a = self.perimeter(), self.area()
        return p / np.sqrt(4.0 * np.pi * a)

    def make_locus(self):
        raise NotImplementedError("YET TO IMPLEMENT")


class CShape(Shape2D):
    def __init__(self, ri=2.0, ro=1.0, theta_c: float = 0.5 * np.pi, cent=(0.0, 0.0), locus=None):
        assert ro >= ri, f"Requires outer radius > inner radius but found {ro} < {ri}"
        self.ri = ri
        self.ro = ro
        self.r = (ro - ri) * 0.5
        self.rm = (ro + ri) * 0.5
        self.theta_c = theta_c
        self.locus = locus
        super(CShape, self).__init__(locus)
        return

    def perimeter(self):
        return (2.0 * np.pi * self.r) + (2.0 * self.theta_c * self.rm)

    def area(self):
        return (np.pi * self.r * self.r) + (2.0 * self.theta_c * self.r * self.rm)

    def shape_factor(self, p=None, a=None):
        if a is None and p is None:
            p, a = self.perimeter(), self.area()
        return p / np.sqrt(4.0 * np.pi * a)


class NLobeShape(Shape2D):

    def __init__(self,
                 num_lobes: int,
                 ldf: float,
                 eq_radius: float,
                 # lobe_radius: float = None,
                 # outer_radius: float = None,
                 locus=None
                 ):
        super(NLobeShape, self).__init__(locus)
        #
        assert 0.0 < ldf < 1.0, f"Invalid lobe distance factor {ldf} is encountered, it must be in (0.0, 1.0)"
        #
        self.num_lobes = num_lobes
        self.eq_radius = eq_radius
        self.ldf = ldf
        self.lobe_radius = None
        self.outer_radius = None
        self.alpha = np.pi / self.num_lobes
        self.theta = np.arcsin(0.5 * (1.0 + self.ldf))

    def set_lobe_radius(self, l_df: float = None, eq_radius: float = None):
        if l_df is None:
            l_df = self.ldf
        if eq_radius is None:
            eq_radius = self.eq_radius
        #
        k1 = self.alpha * np.sin(self.alpha)
        self.lobe_radius = eq_radius * np.sqrt(k1 / (k1 + (2.0 * (1.0 + l_df) * np.sin(self.alpha + self.theta))))
        return self

    def set_outer_radius(self):
        if self.lobe_radius is None:
            self.set_lobe_radius()
        self.outer_radius = ((self.ldf + 1.0 + np.sin(self.alpha)) / np.sin(self.alpha)) * self.lobe_radius
        return self
        # the present implementation assumes that ldf and eq_radius are known!

        # if self.ldf is None and self.lobe_radius is None and self.outer_radius is None:
        #     raise ValueError("At least two of three must be supplied.")
        # else:
        #     if self.ldf is None:
        #         self.ldf = ((self.outer_radius / self.lobe_radius) - 1.0) * np.sin(self.alpha) - 1.0
        #     else:
        #         k = (self.ldf + 1.0 + np.sin(self.alpha)) / np.sin(self.alpha)
        #         if self.outer_radius is None:
        #             self.outer_radius = self.lobe_radius * k
        #         elif self.lobe_radius is None:
        #             self.lobe_radius = self.outer_radius / k
        #

    def perimeter(self):
        if self.lobe_radius is None:
            self.set_lobe_radius()
        return 2.0 * self.num_lobes * self.lobe_radius * (self.alpha + (2.0 * self.theta))

    def area(self):
        if self.lobe_radius is None:
            self.set_lobe_radius()
        return self.num_lobes * self.lobe_radius * self.lobe_radius * (
                self.alpha + (2.0 * (1.0 + self.ldf) * np.sin(self.alpha + self.theta) / np.sin(self.alpha))
        )

    def shape_factor(self, p=None, a=None):
        if a is None and p is None:
            p, a = self.perimeter(), self.area()
        return p / np.sqrt(4.0 * np.pi * a)


class BoundingBox:
    def __init__(self, *bbox: float):
        assert len(bbox) in (4, 6), "Length of the bounding box must be either 4 (for 2D) or 6 (for 3D)"
        self.bbox = bbox
        if len(self.bbox) == 4:
            self.dim: int = 2
            self.xlb: float = self.bbox[0]
            self.ylb: float = self.bbox[1]
            self.xub: float = self.bbox[2]
            self.yub: float = self.bbox[3]
            assert self.xub > self.xlb, f"x upper bound ({self.xub}) > ({self.xlb}) x lower bound"
            assert self.yub > self.ylb, f"y upper bound ({self.yub}) > ({self.ylb}) y lower bound"
            self.lx: float = self.xub - self.xlb
            self.ly: float = self.yub - self.ylb
            self.perimeter: float = 2.0 * (self.lx + self.ly)
            self.area: float = self.lx * self.ly
            self.domain: float = self.area

        elif len(self.bbox) == 6:
            self.dim: int = 3
            self.xlb: float = self.bbox[0]
            self.ylb: float = self.bbox[1]
            self.zlb: float = self.bbox[2]
            self.xub: float = self.bbox[3]
            self.yub: float = self.bbox[4]
            self.zub: float = self.bbox[5]
            assert self.xub > self.xlb, f"x upper bound ({self.xub}) > ({self.xlb}) x lower bound"
            assert self.yub > self.ylb, f"y upper bound ({self.yub}) > ({self.ylb}) y lower bound"
            assert self.zub > self.zlb, f"z upper bound ({self.zub}) > ({self.zlb}) z lower bound"
            self.lx: float = self.xub - self.xlb
            self.ly: float = self.yub - self.ylb
            self.lz: float = self.zub - self.zlb
            self.surface_area: float = 2.0 * (self.lx * self.ly + self.ly * self.lz + self.lz * self.lx)
            self.volume: float = self.lx * self.ly * self.lz
            self.domain: float = self.volume
        else:
            raise ValueError(f"The length of the bounding box can be either 4 or 6 but not {len(self.bbox)}")

    # def scale(self, *scaling_factors: float):
    #     num_scaling_factors: int = len(scaling_factors)
    #     assert num_scaling_factors in (1, 4, 6), "number of scaling factors must be 1 or 4 or 6"
    #     if num_scaling_factors == 1:
    #
    #     return
