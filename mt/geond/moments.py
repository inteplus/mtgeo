"""Raw moments up to 2nd order of ND points.

The module provides :class:`Moments`, which summarises a (possibly weighted or continuous) set of
points by its 0th-, 1st- and 2nd-order raw moments, from which the mean and covariance matrix are
derived, and :func:`moments_from_pointlist` to compute the moments of a point list. `EPSILON` is the
threshold below which `m0` is considered to be zero.
"""

from mt import np
import sys as _sys

import mt.base.casting as _bc

from ..geo import GeometricObject
from .point_list import PointList


__all__ = ["EPSILON", "Moments", "moments_from_pointlist"]


EPSILON = np.sqrt(_sys.float_info.epsilon)


class Moments(GeometricObject):
    """Raw moments up to 2nd order of points living in the same Euclidean space.

    For a set of points `x_i` (or a continuous region), the raw moments are `m0 = sum 1` (the
    number of points, or the volume), `m1 = sum x_i` and `m2 = sum x_i x_i^T`. From them the
    properties `mean = m1/m0` and `cov = m2/m0 - mean mean^T` are derived. If `abs(m0)` is smaller
    than `EPSILON`, the mean is the zero vector and the covariance matrix is the identity matrix.

    Overloaded operators are negation, multiplication with a scalar and true division with a scalar.
    Each returns a new instance of the same type with all three moments negated, multiplied or
    divided.

    Parameters
    ----------
    m0 : scalar
        0th-order raw moment
    m1 : numpy 1d array of length D, where D is the `ndim` of the class
        1st-order raw moment
    m2 : numpy DxD matrix
        2nd-order raw moment

    See Also
    --------
    moments_from_pointlist : computes the moments of a point list
    mt.geo3d.Moments3d : the 3D version, implemented in glm

    """

    def __init__(self, m0, m1, m2):
        self._m0 = float(m0)
        self._m1 = np.array(m1)
        self._m2 = np.array(m2)
        self._mean = None
        self._cov = None

    def __repr__(self):
        return f"Moments(ndim={self.ndim}, m0={self.m0}, mean={self.mean})"

    @property
    def m0(self):
        """0th-order raw moment: the number of points, or the volume of the region (scalar)."""
        return self._m0

    @property
    def m1(self):
        """1st-order raw moment: the sum of the points, a vector of length `ndim`."""
        return self._m1

    @property
    def m2(self):
        """2nd-order raw moment: the sum of the outer products of the points, a `ndim` x `ndim`
        matrix."""
        return self._m2

    @property
    def mean(self):
        """Returns the mean vector `m1/m0`, or the zero vector if `abs(m0) < EPSILON`.

        The result is computed once and cached.
        """
        if self._mean is None:
            self._mean = (
                np.zeros(self.ndim) if abs(self.m0) < EPSILON else self.m1 / self.m0
            )
        return self._mean

    @property
    def cov(self):
        """Returns the covariance matrix `m2/m0 - outer(mean, mean)`.

        If `abs(m0) < EPSILON`, the identity matrix is returned instead. The result is computed
        once and cached.
        """
        if self._cov is None:
            self._cov = (
                np.eye(self.ndim)
                if abs(self.m0) < EPSILON
                else (self.m2 / self.m0) - np.outer(self.mean, self.mean)
            )
        return self._cov

    # ----- operators -----

    @property
    def ndim(self):
        return len(self.m1)

    def __neg__(self):
        """Returns a new instance of the same type where all the moments are negated."""
        return type(self)(-self.m0, -self.m1, -self.m2)

    def __mul__(self, scalar):
        """Returns a new instance of the same type where all the moments are multiplied by a
        scalar."""
        return type(self)(self.m0 * scalar, self.m1 * scalar, self.m2 * scalar)

    def __truediv__(self, scalar):
        """Returns a new instance of the same type where all the moments are divided by a
        scalar."""
        return type(self)(self.m0 / scalar, self.m1 / scalar, self.m2 / scalar)


def moments_from_pointlist(pl):
    """Constructs a Moments object from a list of points.

    The moments are `m0 = N`, `m1 = pl.points.sum(axis=0)` and `m2 = pl.points.T @ pl.points`. It
    is registered as the cast function from :class:`PointList` to :class:`Moments`, and from
    :class:`mt.geo3d.PointList3d` to :class:`mt.geo3d.Moments3d`.

    Parameters
    ----------
    pl : PointList
        list of points from which the moments are computed

    Returns
    -------
    Moments, Moments2d or Moments3d
        raw moments of the point list, depending on the dimension of the points (the number of
        columns of `pl.points`): 2 gives `Moments2d`, 3 gives `Moments3d`, otherwise `Moments`

    """
    arr = pl.points
    m0 = len(arr)
    m1 = arr.sum(axis=0)
    m2 = np.dot(arr.T, arr)
    if arr.shape[1] == 2:
        from ..geo2d.moments import Moments2d  # local import to avoid a circular import

        return Moments2d(m0, m1, m2)
    if arr.shape[1] == 3:
        from ..geo3d.moments import Moments3d  # local import to avoid a circular import

        return Moments3d(m0, m1, m2)
    return Moments(m0, m1, m2)


_bc.register_cast(PointList, Moments, moments_from_pointlist)
