'''Raw moments up to 2nd order of 2D points.'''

import numpy as _np
import math as _m
import sys as _sys

import mt.base.cast as _bc

from .object import GeometricObject, TwoD, ThreeD
from .point_list import PointList, PointList2d, PointList3d
from .polygon import Polygon


__all__ = ['EPSILON', 'Moments', 'Moments2d', 'Moments3d']


EPSILON = _m.sqrt(_sys.float_info.epsilon)


class Moments(GeometricObject):
    '''Raw moments up to 2nd order of points living in the same Euclidean space.

    Parameters
    ----------
    m0 : scalar
        0th-order raw moment
    m1 : numpy 1d array of length D, where D is the `ndim` of the class
        1st-order raw moment
    m2 : numpy DxD matrix
        2nd-order raw moment

    Examples
    --------
    >>> import mt.geo.polygon as gp
    >>> poly = gp.Polygon([[0,0],[0,1],[1,2],[1,0]])
    >>> import mt.geo.moments as gm
    >>> from mt.base.cast import cast
    >>> m = cast(poly, gm.Moments2d)
    >>> m.m0
    4.0
    >>> m.m1
    array([2, 3])
    >>> m.m2
    array([[2, 2],
           [2, 5]])
    >>> m.mean
    array([0.5 , 0.75])
    >>> m.cov
    array([[0.25  , 0.125 ],
           [0.125 , 0.6875]])
    '''

    def __init__(self, m0, m1, m2):
        self._m0 = _np.float(m0)
        self._m1 = _np.array(m1)
        self._m2 = _np.array(m2)
        self._mean = None
        self._cov = None

    @property
    def m0(self):
        '''0th-order moment'''
        return self._m0

    @property
    def m1(self):
        '''1st-order moment'''
        return self._m1

    @property
    def m2(self):
        '''2nd-order moment'''
        return self._m2

    @property
    def mean(self):
        '''Returns the mean vector.'''
        if self._mean is None:
            self._mean = _np.zeros(self.ndim) if abs(self.m0) < EPSILON else self.m1/self.m0
        return self._mean

    @property
    def cov(self):
        '''Returns the covariance matrix.'''
        if self._cov is None:
            self._cov = _np.eye(self.ndim) if abs(self.m0) < EPSILON else (self.m2/self.m0) - _np.outer(self.mean, self.mean)
        return self._cov

    def negate(self):
        '''Returns a new instance where all the moments are negated.'''
        return type(self)(-self.m0, -self.m1, -self.m2)


class Moments2d(TwoD, Moments):
    '''Raw moments up to 2nd order of points living in 2D. See Moments for more details.'''
    pass


class Moments3d(ThreeD, Moments):
    '''Raw moments up to 2nd order of points living in 3D. See Moments for more details.'''
    pass


def moments_from_pointlist(pl):
    '''Constructs a Moments object from a list of points.

    Parameters
    ----------
    pl : PointList
        list of points from which the moments are computed

    Returns
    -------
    Moments, Moments2d or Moments3d
        raw moments of the point list, depending on the value of `ndim` provided
    '''
    arr = pl.points
    m0 = len(arr)
    m1 = arr.sum(axis=0)
    m2 = _np.dot(arr.T, arr)
    if arr.shape[1] == 2:
        return Moments2d(m0, m1, m2)
    if arr.shape[1] == 3:
        return Moments3d(m0, m1, m2)
    return Moments(m0, m1, m2)

_bc.register_cast(PointList, Moments, moments_from_pointlist)
_bc.register_cast(PointList2d, Moments2d, moments_from_pointlist)
_bc.register_cast(PointList3d, Moments3d, moments_from_pointlist)