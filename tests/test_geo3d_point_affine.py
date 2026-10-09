import unittest

import numpy as np

import mt.geo3d
from mt.base.casting import cast
from mt.geond import Aff, PointList
from mt.geo3d import Aff3d, Moments3d, Point3d, PointList3d

assert mt.geo3d.__file__.startswith("/home/minhtri/gitcentre/mtgeo/")


class TestPoints(unittest.TestCase):
    def test_pointlist3d_check_arg(self):
        pl = PointList3d([[1, 2, 3]], check=False)
        self.assertEqual(pl.points.shape, (1, 3))
        with self.assertRaises(ValueError):
            PointList3d([[1, 2]])
        self.assertEqual(PointList3d([]).points.shape, (0, 3))

    def test_pointlist3d_casts(self):
        pl = cast(np.ones((2, 3)), PointList3d)
        self.assertEqual(pl.points.shape, (2, 3))
        pl2 = cast(PointList(np.ones((2, 3)), check=False), PointList3d)
        self.assertEqual(pl2.points.shape, (2, 3))
        self.assertEqual(cast(pl2, PointList).points.shape, (2, 3))

    def test_point3d_check_arg(self):
        self.assertEqual(Point3d([1, 2, 3], check=True).ndim, 3)
        with self.assertRaises(ValueError):
            Point3d([1, 2], check=True)
        with self.assertRaises(ValueError):
            Point3d([1, 2])
        with self.assertRaises(TypeError):
            Point3d([1, 2], check=False)

    def test_aff3d_on_pointlist(self):
        a = Aff3d(offset=np.array([1.0, 2.0, 3.0]), linear=np.diag([2.0, 3.0, 4.0]))
        out = a << PointList3d([[1, 1, 1]])
        self.assertIsInstance(out, PointList3d)
        np.testing.assert_allclose(out.points, [[3, 5, 7]])


class TestAff3d(unittest.TestCase):
    def setUp(self):
        self.a = Aff3d(
            offset=np.array([1.0, 2.0, 3.0]),
            linear=np.array([[1.0, 2, 0], [0, 1, 0], [0, 0, 3]]),
        )

    def test_casts(self):
        n = cast(self.a, Aff)
        self.assertIsInstance(n, Aff)
        np.testing.assert_allclose(n.weight, self.a.weight)
        np.testing.assert_allclose(n.bias, self.a.bias)
        b = cast(n, Aff3d)
        self.assertIsInstance(b, Aff3d)
        np.testing.assert_allclose(b.matrix, self.a.matrix)

    def test_moments_transform(self):
        pts = np.array([[0.0, 0, 0], [2, 0, 0], [0, 4, 0], [0, 0, 6], [1, 1, 1]])
        m = cast(PointList3d(pts), Moments3d)
        out = self.a << m
        self.assertIsInstance(out, Moments3d)
        A, b = self.a.weight.astype(float), self.a.bias.astype(float)
        np.testing.assert_allclose(out.mean, A @ m.mean + b, rtol=1e-5)
        np.testing.assert_allclose(out.cov, A @ m.cov @ A.T, rtol=1e-4, atol=1e-5)
        self.assertAlmostEqual(out.m0, m.m0 * abs(self.a.det), places=4)


if __name__ == "__main__":
    unittest.main()
