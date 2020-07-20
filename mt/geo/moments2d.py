'''Raw moments up to 2nd order of 2D points.'''

import numpy as _np
import math as _m
import sys as _sys

from mt.base import logger
logger.warn_module_move('mt.geo.moments2d', 'mt.geo.moments')


__all__ = ['EPSILON', 'moments2d']


EPSILON = _m.sqrt(_sys.float_info.epsilon)

class moments2d(object):
    '''Raw moments up to 2nd order of 2D points.'''

    def __init__(self, m0, m1, m2):
        '''Initialises the object.

        Parameters:
            m0 : scalar
                0th-order raw moment
            m1 : numpy 1d array of length 2
                1st-order raw moment
            m2 : numpy 2x2 matrix
                2nd-order raw moment
        '''
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
            self._mean = _np.zeros(2) if abs(self.m0) < EPSILON else self.m1/self.m0
        return self._mean

    @property
    def cov(self):
        '''Returns the covariance matrix.'''
        if self._cov is None:
            self._cov = _np.eye(2) if abs(self.m0) < EPSILON else (self.m2/self.m0) - _np.outer(self.mean, self.mean)
        return self._cov

    def negate(self):
        '''Returns a new instance where all the moments are negated.'''
        return moments2d(-self.m0, -self.m1, -self.m2)

    @staticmethod
    def from_pointset(arr):
        '''Constructs a moments2d object from a set of points.

        :Parameters:
            arr : numpy Nx2 matrix

        :Returns:
            ret_val : moments2d
                raw moments of the point set
        '''
        if len(arr.shape) != 2 or arr.shape[1] != 2:
            raise ValueError("Input array has an invalid shape, expecting (_,2), but receiving {}".format(arr.shape))
        return moments2d(len(arr), arr.sum(axis=1), _np.dot(arr.T, arr))
