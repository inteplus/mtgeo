'''Raw moments up to 2nd order of 2D points.'''

import numpy as _np
import math as _m

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
        self.m0 = _np.float(m0)
        self.m1 = _np.array(m1)
        self.m2 = _np.array(m2)

    @property
    def mean(self):
        '''Returns the mean vector.'''
        return self.m1 / self.m0

    @property
    def cov(self):
        '''Returns the covariance matrix.'''
        return (self.m2 / self.m0) - _np.dot(self.m1, self.m1.T)

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
