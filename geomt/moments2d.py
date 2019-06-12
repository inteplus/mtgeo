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
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2

    @property
    def mean(self):
        '''Returns the mean vector.'''
        return self.m1 / self.m0

    @property
    def cov(self):
        '''Returns the covariance matrix.'''
        return (self.m2 / self.m0) - _np.dot(self.m1, self.m1.T)
