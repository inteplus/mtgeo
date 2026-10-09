"""Regression tests for bugs fixed in mt.geo2d."""

import math
import unittest

import numpy as np

import mt.geo2d as g2
from mt import glm
from mt.base.casting import cast
from mt.geo import transform, join_volume

assert g2.__file__.startswith("/home/minhtri/gitcentre/mtgeo/"), g2.__file__

from mt.geo2d import (
    Point2d, PointList2d, Polygon, Rect, RRect, Moments2d, Aff2d, iou, translate2d, scale2d,
)
from mt.geo2d.rect import join_volume_Rect_and_Rect


class TestCheckParam(unittest.TestCase):
    def test_constructors_accept_check(self):
        self.assertEqual(Point2d([1, 2], check=False).point.tolist(), [1, 2])
        self.assertEqual(PointList2d([[1, 2]], check=False).points.shape, (1, 2))
        self.assertEqual(Polygon([[0, 0], [1, 0], [0, 1]], check=False).points.shape, (3, 2))

    def test_check_true_rejects_bad_shape(self):
        with self.assertRaises(Exception):
            PointList2d([[1, 2, 3]], check=True)

    def test_transform_pointlist_and_polygon(self):
        t = translate2d(1, 2)
        pl = transform(t, PointList2d([[0, 0], [1, 1]]))
        self.assertEqual(pl.points.tolist(), [[1, 2], [2, 3]])
        poly = transform(t, Polygon([[0, 0], [1, 0], [0, 1]]))
        self.assertIsInstance(poly, Polygon)
        self.assertEqual(poly.points.tolist(), [[1, 2], [2, 2], [1, 3]])

    def test_casts(self):
        a = np.array([[0.0, 1.0], [2.0, 3.0]])
        self.assertIsInstance(cast(a, PointList2d), PointList2d)
        self.assertIsInstance(cast(a, Polygon), Polygon)
        self.assertIsInstance(cast(np.array([1.0, 2.0]), Point2d), Point2d)
        pl = PointList2d(a)
        self.assertIsInstance(cast(pl, Polygon), Polygon)


class TestMomentsTransform(unittest.TestCase):
    def test_transform_moments(self):
        m = Moments2d(10, [20, 30], [[50, 60], [60, 100]])  # mean (2,3)
        t = translate2d(1, 2) * scale2d(2, 3)
        r = transform(t, m)
        self.assertAlmostEqual(r.m0, 60, places=3)
        np.testing.assert_allclose(r.mean, [5, 11], rtol=1e-5)
        A = np.diag([2.0, 3.0])
        np.testing.assert_allclose(r.cov, A @ m.cov @ A.T, rtol=1e-4)
        # raw moments consistent with mean/cov
        np.testing.assert_allclose(r.m1, r.m0 * r.mean, rtol=1e-4)

    def test_transform_moments_nonsymmetric_linear(self):
        m = Moments2d(4, [4, 8], [[8, 10], [10, 30]])
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        r = transform(Aff2d(offset=np.array([1.0, 1.0]), linear=A), m)
        np.testing.assert_allclose(r.cov, A @ m.cov @ A.T, rtol=1e-4)

    def test_cast_ellipse_to_moments(self):
        from mt.geo2d import Ellipse
        e = Ellipse(g2.Aff2d(offset=np.array([1.0, 2.0]), linear=np.array([[2.0, 0], [0, 1.0]])))
        mom = cast(e, Moments2d)
        np.testing.assert_allclose(mom.mean, [1, 2], rtol=1e-5)


class TestRRect(unittest.TestCase):
    def test_json_roundtrip(self):
        r = RRect(4, 2, cx=10, cy=11, angle=0.5)
        j = r.to_json()
        self.assertEqual(j, [4, 2, 10, 11, 0.5])
        r2 = RRect.from_json(j)
        np.testing.assert_allclose(r2.corners, r.corners)

    def test_moment1(self):
        r = RRect(4, 2, cx=10, cy=11, angle=0.3)
        np.testing.assert_allclose(r.moment1, 8 * np.array([10, 11]))
        rf = RRect(-4, 2, cx=10, cy=11, angle=0.3)
        np.testing.assert_allclose(rf.moment1, -8 * np.array([10, 11]))

    def test_cast_to_moments_matches_rect(self):
        r = RRect(4, 2, cx=10, cy=11, angle=0.0)
        rect = Rect(8, 10, 12, 12)
        m1 = cast(r, Moments2d)
        m2 = cast(rect, Moments2d)
        self.assertAlmostEqual(m1.m0, m2.m0)
        np.testing.assert_allclose(m1.m1, m2.m1, rtol=1e-5)
        np.testing.assert_allclose(m1.mean, [10, 11], rtol=1e-5)
        np.testing.assert_allclose(m1.cov, m2.cov, rtol=1e-4)


class TestRectDisjoint(unittest.TestCase):
    def test_intersect_disjoint_is_empty(self):
        r = Rect(0, 0, 10, 10).intersect(Rect(20, 20, 30, 30))
        self.assertEqual(r.area, 0)
        r = Rect(0, 0, 10, 10).intersect(Rect(20, 0, 30, 10))
        self.assertEqual(r.area, 0)

    def test_intersect_overlap_unchanged(self):
        r = Rect(0, 0, 10, 10).intersect(Rect(5, 5, 20, 20))
        self.assertEqual(r.to_json(), [5, 5, 10, 10])

    def test_iou_disjoint(self):
        self.assertAlmostEqual(Rect(0, 0, 10, 10).iou(Rect(20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(Rect(0, 0, 10, 10).iou(Rect(20, 0, 30, 10)), 0.0)
        self.assertAlmostEqual(Rect(0, 0, 10, 10).iou(Rect(5, 0, 15, 10)), 1 / 3, places=5)

    def test_join_volume_disjoint(self):
        res = join_volume_Rect_and_Rect(Rect(0, 0, 10, 10), Rect(20, 20, 30, 30))
        self.assertEqual(res, (0, 100, 100, 200))
        res = join_volume(Rect(0, 0, 10, 10), Rect(20, 20, 30, 30))
        self.assertEqual(res, (0, 100, 100, 200))

    def test_touching(self):
        self.assertEqual(Rect(0, 0, 10, 10).intersect(Rect(10, 0, 20, 10)).area, 0)


class TestIouLists(unittest.TestCase):
    def test_single(self):
        a = Rect(0, 0, 1, 1)
        self.assertAlmostEqual(iou([a], [a]), 1.0, places=3)

    def test_lists(self):
        a, b = Rect(0, 0, 1, 1), Rect(2, 0, 3, 1)
        self.assertAlmostEqual(iou([a, b], [a]), 0.5, places=3)
        self.assertAlmostEqual(iou([a], [a, b]), 0.5, places=3)
        self.assertAlmostEqual(iou([a, b], [a, b]), 1.0, places=3)
        self.assertAlmostEqual(iou([a, b], [Rect(5, 5, 6, 6)]), 0.0, places=3)
        self.assertAlmostEqual(iou([], [a]), 0.0, places=3)

    def test_matrix_property(self):
        np.testing.assert_allclose(
            translate2d(1, 2).matrix, [[1, 0, 1], [0, 1, 2], [0, 0, 1]]
        )
        m = (scale2d(2, 3)).matrix
        np.testing.assert_allclose(m, [[2, 0, 0], [0, 3, 0], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
