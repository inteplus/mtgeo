import unittest

import numpy as np

from mt.base.casting import cast
from mt.geo import join_volume, lower_bound, upper_bound
from mt.geond import Hyperellipsoid
from mt.geo3d import Aff3d, Box, Ellipsoid, Moments3d


class TestBox(unittest.TestCase):
    def test_surface_area(self):
        self.assertEqual(Box(0, 0, 0, 1, 2, 3).surface_area, 22.0)

    def test_cast_to_moments(self):
        m = cast(Box(0, 0, 0, 1, 2, 3), Moments3d)
        self.assertIsInstance(m, Moments3d)
        np.testing.assert_allclose(m.m0, 6.0)
        np.testing.assert_allclose(m.mean, [0.5, 1.0, 1.5], rtol=1e-6)
        np.testing.assert_allclose(m.m1, [3.0, 6.0, 9.0], rtol=1e-6)

    def test_intersect_disjoint(self):
        a, b = Box(0, 0, 0, 1, 1, 1), Box(3, 3, 3, 4, 4, 4)
        self.assertIsInstance(a.intersect(b), Box)
        self.assertEqual(a.intersect(b).volume, 0)
        self.assertEqual(join_volume(a, b), (0, 1.0, 1.0, 2.0))
        self.assertEqual(a.iou(b), 0)

    def test_intersect_overlap(self):
        a, b = Box(0, 0, 0, 1, 2, 3), Box(0.5, 1, 1, 2.5, 4, 3)
        self.assertEqual(join_volume(a, b), (1.0, 5.0, 11.0, 17.0))


class TestEllipsoid(unittest.TestCase):
    def setUp(self):
        self.a = Aff3d(offset=np.array([1.0, 2.0, 3.0]), linear=np.diag([2.0, 3.0, 4.0]))

    def test_normalised_default(self):
        e = Ellipsoid(self.a)
        np.testing.assert_allclose(e.aff_tfm.bias, [1, 2, 3])
        self.assertAlmostEqual(e.volume, 4 * np.pi / 3 * 24, places=3)
        w = e.aff_tfm.weight
        np.testing.assert_allclose(w.T @ w, np.diag([4.0, 9.0, 16.0])[::-1, ::-1], atol=1e-4)
        e2 = e.normalised()
        np.testing.assert_allclose(e2.aff_tfm.bias, [1, 2, 3])

    def test_transform(self):
        t = Aff3d(offset=np.array([1.0, 1.0, 1.0]))
        e = t << Ellipsoid(self.a)
        self.assertIsInstance(e, Ellipsoid)
        np.testing.assert_allclose(e.aff_tfm.bias, [2, 3, 4])

    def test_volume_absolute(self):
        a = Aff3d(linear=np.diag([-1.0, 1.0, 1.0]))
        self.assertGreater(Ellipsoid(a, make_normalised=False).volume, 0)
        self.assertGreater(Ellipsoid(a).volume, 0)

    def test_upper_bound(self):
        b = upper_bound(Ellipsoid(self.a), Box)
        self.assertIsInstance(b, Box)
        np.testing.assert_allclose(b.min_coords, [-1, -1, -1], atol=1e-5)
        np.testing.assert_allclose(b.max_coords, [3, 5, 7], atol=1e-5)

    def test_lower_bound(self):
        e = lower_bound(Box(0, 0, 0, 2, 4, 6), Ellipsoid)
        self.assertIsInstance(e, Ellipsoid)
        np.testing.assert_allclose(e.aff_tfm.bias, [1, 2, 3])
        self.assertAlmostEqual(e.volume, 4 * np.pi / 3 * 6, places=3)

    def test_cast_hyperellipsoid(self):
        h = cast(Ellipsoid(self.a), Hyperellipsoid)
        self.assertEqual(h.ndim, 3)
        self.assertIsInstance(cast(h, Ellipsoid), Ellipsoid)


if __name__ == "__main__":
    unittest.main()
