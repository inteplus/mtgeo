import unittest

import numpy as np

from mt import geond as gn
from mt.geo import join_volume


def box(lo, hi):
    return gn.Hyperbox(np.array(lo, dtype=float), np.array(hi, dtype=float))


class TestHyperbox(unittest.TestCase):
    def test_single_arg(self):
        b = gn.Hyperbox(np.array([2.0, 4.0]))
        np.testing.assert_allclose(b.min_coords, [0, 0])
        np.testing.assert_allclose(b.max_coords, [2, 4])

    def test_ndim_is_value(self):
        b = box([0, 0], [2, 4])
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.dim, 2)

    def test_validated(self):
        b = box([2, 4], [0, 0])
        v = b.validated()
        self.assertTrue(v.is_valid())
        np.testing.assert_allclose(v.min_coords, [0, 0])
        np.testing.assert_allclose(v.max_coords, [2, 4])

    def test_size(self):
        np.testing.assert_allclose(box([0, 0], [2, 4]).size, [2, 4])

    def test_intersect_disjoint(self):
        a, b = box([0, 0], [1, 1]), box([3, 3], [4, 4])
        i = a.intersect(b)
        self.assertTrue(i.is_valid())
        self.assertEqual(i.volume, 0)
        self.assertEqual(join_volume(a, b), (0, 1.0, 1.0, 2.0))
        self.assertEqual(a.iou(b), 0)

    def test_intersect_disjoint_one_axis(self):
        a, b = box([0, 0], [2, 2]), box([1, 5], [3, 6])
        self.assertEqual(a.intersect(b).volume, 0)

    def test_intersect_overlap(self):
        a, b = box([0, 0], [2, 4]), box([1, 2], [5, 3])
        self.assertEqual(a.intersect(b).volume, 1.0)
        self.assertEqual(join_volume(a, b), (1.0, 7.0, 3.0, 11.0))


class TestHypercube(unittest.TestCase):
    def test_build_from_offset(self):
        c = gn.Hypercube(np.array([1.0, 2.0]), 3.0)
        self.assertEqual(c.dim, 2)
        self.assertEqual(c.ndim, 2)
        np.testing.assert_allclose(c.center_pt, [1, 2])
        np.testing.assert_allclose(c.size, [6, 6])
        np.testing.assert_allclose(c.min_coords, [-2, -1])
        np.testing.assert_allclose(c.max_coords, [4, 5])

    def test_build_from_dltra(self):
        c = gn.Hypercube(gn.Dltra(offset=np.array([0.0, 0.0]), scale=1.0))
        np.testing.assert_allclose(c.size, [2, 2])

    def test_cast_and_approx(self):
        from mt.base.casting import cast
        from mt.geo import approx

        c = gn.Hypercube(np.array([1.0, 2.0]), 3.0)
        b = cast(c, gn.Hyperbox)
        np.testing.assert_allclose(b.min_coords, [-2, -1])
        np.testing.assert_allclose(b.max_coords, [4, 5])
        c2 = approx(box([0, 0], [2, 4]), gn.Hypercube)
        np.testing.assert_allclose(c2.center_pt, [1, 2])
        self.assertAlmostEqual(c2.dltra_tfm.scale, 1.5)


class TestHyperellipsoid(unittest.TestCase):
    def setUp(self):
        self.e = gn.Hyperellipsoid(gn.Aff(weight=np.diag([4.0, 5.0]), bias=np.array([2.0, 3.0])))

    def test_ndim(self):
        self.assertEqual(self.e.ndim, 2)

    def test_aff_transform(self):
        t = gn.Aff(weight=np.eye(2), bias=np.array([1.0, 1.0]))
        out = t << self.e
        self.assertIsInstance(out, gn.Hyperellipsoid)
        np.testing.assert_allclose(out.aff_tfm.bias, [3, 4])

    def test_upper_bound(self):
        from mt.geo import upper_bound

        b = upper_bound(self.e, gn.Hyperbox)
        np.testing.assert_allclose(b.min_coords, [-2, -2])
        np.testing.assert_allclose(b.max_coords, [6, 8])


if __name__ == "__main__":
    unittest.main()
