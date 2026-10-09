import os
import unittest

import numpy as np

import mt.geond
from mt.base.casting import cast
from mt.geond import Moments, PointList, moments_from_pointlist

assert os.path.abspath(mt.geond.__file__).startswith("/home/minhtri/gitcentre/mtgeo/")


class TestMoments(unittest.TestCase):
    def test_base_moments_can_be_built(self):
        m = Moments(4, np.array([2.0, 4.0]), np.eye(2) * 8)
        self.assertEqual(m.m0, 4.0)
        np.testing.assert_allclose(m.mean, [0.5, 1.0])
        self.assertEqual(m.ndim, 2)
        np.testing.assert_allclose((m * 2).m1, [4.0, 8.0])

    def test_moments_from_pointlist_nd(self):
        pl = PointList(np.arange(12.0).reshape(3, 4), check=False)
        m = moments_from_pointlist(pl)
        self.assertEqual(type(m), Moments)
        self.assertEqual(m.m0, 3)
        np.testing.assert_allclose(m.m1, pl.points.sum(axis=0))

    def test_moments_from_pointlist_2d(self):
        from mt.geo2d import Moments2d

        pl = PointList(np.array([[0.0, 0.0], [2.0, 4.0]]), check=False)
        m = moments_from_pointlist(pl)
        self.assertIsInstance(m, Moments2d)
        np.testing.assert_allclose(m.mean, [1.0, 2.0])

    def test_moments_from_pointlist_3d(self):
        from mt.geo3d import Moments3d, PointList3d

        pts = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
        m = moments_from_pointlist(PointList(pts, check=False))
        self.assertIsInstance(m, Moments3d)
        np.testing.assert_allclose(m.mean, [1.0, 2.0, 3.0])
        m = cast(PointList3d(pts), Moments3d)
        self.assertIsInstance(m, Moments3d)
        np.testing.assert_allclose(m.m1, [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
