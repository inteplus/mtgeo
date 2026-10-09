import unittest

import numpy as np

from mt import geond as gn
from mt.base.casting import cast, castable
from mt.geo import transform


class TestDlt(unittest.TestCase):
    def test_cast_aff_to_dlt(self):
        a = gn.Aff(weight=np.diag([2.0, 3.0]), bias=np.array([1.0, 1.0]))
        self.assertTrue(castable(a, gn.Dlt))
        d = cast(a, gn.Dlt)
        self.assertIsInstance(d, gn.Dlt)
        np.testing.assert_allclose(d.scale, [2.0, 3.0])
        np.testing.assert_allclose(d.offset, [1.0, 1.0])

    def test_not_castable_non_diagonal(self):
        a = gn.Aff(weight=np.array([[1.0, 2.0], [0.0, 1.0]]), bias=np.zeros(2))
        self.assertFalse(castable(a, gn.Dlt))

    def test_transform_on_pointlist(self):
        d = gn.Dlt(offset=np.array([1.0, 2.0]), scale=np.array([3.0, 4.0]))
        pl = gn.PointList(np.array([[1.0, 1.0], [0.0, 0.0]]), check=False)
        out = transform(d, pl)
        np.testing.assert_allclose(out.points, [[4.0, 6.0], [1.0, 2.0]])
        out = d << pl
        np.testing.assert_allclose(out.points, [[4.0, 6.0], [1.0, 2.0]])


class TestDliso(unittest.TestCase):
    def test_approx_aff_to_dliso(self):
        from mt.geo import approx

        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        a = gn.Aff(weight=3.0 * R, bias=np.array([1.0, 2.0]))
        d = approx(a, gn.Dliso)
        self.assertIsInstance(d, gn.Dliso)
        self.assertAlmostEqual(d.scale, 3.0)
        np.testing.assert_allclose(d.unitary, R, atol=1e-12)
        np.testing.assert_allclose(d.offset, [1.0, 2.0])

    def test_approx_aff_to_dliso_3d(self):
        from mt.geo import approx

        a = gn.Aff(weight=2.0 * np.eye(3), bias=np.zeros(3))
        d = approx(a, gn.Dliso)
        self.assertAlmostEqual(d.scale, 2.0)
        np.testing.assert_allclose(d.unitary, np.eye(3), atol=1e-12)


    def test_approx_singular_raises_value_error(self):
        from mt.geo import approx

        a = gn.Aff(weight=np.zeros((2, 2)), bias=np.zeros(2))
        with self.assertRaises(ValueError):
            approx(a, gn.Dliso)


class TestDltra(unittest.TestCase):
    def test_exported(self):
        self.assertTrue(hasattr(gn, "Dltra"))
        self.assertIn("Dltra", gn.dilated_translation.__all__)

    def test_expm_logm(self):
        t = gn.Dltra(offset=np.array([1.0, 2.0]), scale=2.0)
        u, v = t.logm()
        e = gn.Dltra.expm(u, v)
        self.assertIsInstance(e, gn.Dltra)
        np.testing.assert_allclose(e.offset, [1.0, 2.0])
        self.assertAlmostEqual(e.scale, 2.0)

    def test_invert(self):
        t = gn.Dltra(offset=np.array([1.0, 2.0, 3.0]), scale=2.0)
        inv = ~t
        np.testing.assert_allclose(inv << (t << np.ones(3)), np.ones(3))
        np.testing.assert_allclose((t * inv).matrix, np.eye(4), atol=1e-12)

    def test_cast_to_dlt(self):
        t = gn.Dltra(offset=np.array([1.0, 2.0, 3.0]), scale=2.0)
        d = cast(t, gn.Dlt)
        self.assertIsInstance(d, gn.Dlt)
        np.testing.assert_allclose(d.scale, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(d.offset, [1.0, 2.0, 3.0])

    def test_cast_to_dliso_3d(self):
        t = gn.Dltra(offset=np.array([1.0, 2.0, 3.0]), scale=2.0)
        d = cast(t, gn.Dliso)
        np.testing.assert_allclose(d.matrix, t.matrix)


class TestIso(unittest.TestCase):
    def test_multiply_non_iso(self):
        i = gn.Iso(offset=np.array([1.0, 2.0]))
        d = gn.Dliso(offset=np.array([0.0, 1.0]), scale=2.0)
        p = i * d
        np.testing.assert_allclose(p.matrix, i.matrix @ d.matrix)
        a = gn.Aff(weight=np.array([[1.0, 1.0], [0.0, 1.0]]), bias=np.zeros(2))
        np.testing.assert_allclose((i * a).matrix, i.matrix @ a.matrix)

    def test_multiply_iso(self):
        i = gn.Iso(offset=np.array([1.0, 2.0]))
        self.assertIsInstance(i * i, gn.Iso)


if __name__ == "__main__":
    unittest.main()
