'''A 2D rectangle.'''

import math as _m
import numpy as _np

from mt.base import logger
from mt.base.casting import *

from .box import box
from .moments import EPSILON, Moments2d
from .approx import *
from .object import TwoD

__all__ = ['rect']

class rect(TwoD, box):
    '''A 2D rectangle,

    Note we do not care if the rectangle is open or partially closed or closed.'''

    
    # ----- derived properties -----

    
    @property
    def min_x(self):
        '''lowest x-coordinate.'''
        return self.min_coords[0]

    @property
    def min_y(self):
        '''lowest x-coodinate.'''
        return self.min_coords[1]

    @property
    def max_x(self):
        '''highest x-coordinate.'''
        return self.max_coords[0]

    @property
    def max_y(self):
        '''highest y-coordinate.'''
        return self.max_coords[1]

    @property
    def x(self):
        '''left, same as min_x.'''
        return self.min_x

    @property
    def y(self):
        '''top, same as min_y.'''
        return self.min_y

    @property
    def w(self):
        '''width'''
        return self.max_x - self.min_x

    @property
    def h(self):
        '''height'''
        return self.max_y - self.min_y

    @property
    def cx(self):
        '''Center x-coordinate.'''
        return (self.min_x + self.max_x)/2

    @property
    def cy(self):
        '''Center y-coordinate.'''
        return (self.min_y + self.max_y)/2

    @property
    def area(self):
        '''Absolute area.'''
        return abs(self.w*self.h)

    
    # ----- moments -----


    @property
    def signed_area(self):
        '''Returns the signed area of the rectangle.'''
        return self.w*self.h

    @property
    def moment_x(self):
        '''Returns the integral of x over the rectangle's interior.'''
        return self.signed_area*self.cx

    @property
    def moment_y(self):
        '''Returns the integral of y over the rectangle's interior.'''
        return self.signed_area*self.cy

    @property
    def moment_xy(self):
        '''Returns the integral of x*y over the rectangle's interior.'''
        return self.moment_x*self.cy

    @property
    def moment_xx(self):
        '''Returns the integral of x*x over the rectangle's interior.'''
        return self.signed_area*(self.min_x*self.min_x+self.min_x*self.max_x+self.max_x*self.max_x)/3

    @property
    def moment_yy(self):
        '''Returns the integral of y*y over the rectangle's interior.'''
        return self.signed_area*(self.min_y*self.min_y+self.min_y*self.max_y+self.max_y*self.max_y)/3

    @property
    def to_moments2d(self):
        '''Computes all moments, up to 2nd-order of the rectangle's interior.

        This function is being deprecated. Please use mt.base.casting.cast() to cast an object of type rect to type Moments2d instead.'''
        logger.warn_func_move('mt.geo.rect.to_moments2d', 'mt.base.casting.cast(r:rect, mt.geo.moments.Moments2d)')
        from .moments2d import moments2d
        m0 = self.signed_area
        m1 = [self.moment_x, self.moment_y]
        mxy = self.moment_xy
        m2 = [[self.moment_xx, mxy], [mxy, self.moment_yy]]
        return moments2d(m0, m1, m2)

    @staticmethod
    def from_moments2d(obj):
        '''Returns a rectangle that best approximates the moments2d instance.

        The function returns a rectangle such that its mean is the same as the mean of the instance, and its x-variance and y-variance are the same as those of the instance. The correlation is ignored.

        This function is being deprecated. Please use mt.geo.approx.approx() to approximate an object of type Moments2d to type rect instead.

        Parameters
        ----------
        obj : moments2d
            an instance containing moments of 2D points up to 2nd order

        Returns
        -------
        rect
            the output rectangle
        '''
        logger.warn_func_move('mt.geo.rect.from_moments2d', 'mt.geo.approx.approx(obj:mt.geo.moments.Moments2d, mt.geo.rect.rect)')
        cx, cy = obj.mean
        cov = obj.cov

        # w = half width, h = half height
        size = abs(obj.m0)
        hw3 = cov[0][0]*size*0.75 # should be >= 0
        wh3 = cov[1][1]*size*0.75 # should be >= 0
        wh = _m.sqrt(_m.sqrt(wh3*hw3))
        h = _m.sqrt(wh3/wh)
        w = _m.sqrt(hw3/wh)
        return rect(cx-w, cy-h, cx+w, cy+h)


    # ----- methods -----

    
    def __init__(self, min_x, min_y, max_x, max_y, force_valid=False):
        super(rect, self).__init__(_np.array([min_x, min_y]), _np.array([max_x, max_y]), force_valid = force_valid)

    def __repr__(self):
        return "rect(x={}, y={}, w={}, h={})".format(self.x, self.y, self.w, self.h)

    def intersect(self, other):
        res = super(rect, self).intersect(other)
        return rect(res.min_coords[0], res.min_coords[1], res.max_coords[0], res.max_coords[1])

    def union(self, other):
        res = super(rect, self).union(other)
        return rect(res.min_coords[0], res.min_coords[1], res.max_coords[0], res.max_coords[1])

    def move(self, offset):
        '''Moves the rect by a given offset vector.'''
        return rect(self.min_x + offset[0], self.min_y + offset[1], self.max_x + offset[0], self.max_y + offset[1])


def cast_rect_to_moments(obj):
    m0 = obj.signed_area
    m1 = [obj.moment_x, obj.moment_y]
    mxy = obj.moment_xy
    m2 = [[obj.moment_xx, mxy], [mxy, obj.moment_yy]]
    return Moments2d(m0, m1, m2)
register_cast(rect, Moments2d, cast_rect_to_moments)


def approx_moments_to_rect(obj):
    '''Approximates a Moments2d instance with a rect such that the mean aligns with the rect's center, and the covariance matrix of the instance is closest to the moment convariance matrix of the rect.'''
    cx, cy = obj.mean
    cov = obj.cov

    # w = half width, h = half height
    size = abs(obj.m0)
    hw3 = cov[0][0]*size*0.75 # should be >= 0
    wh3 = cov[1][1]*size*0.75 # should be >= 0
    wh = _m.sqrt(_m.sqrt(wh3*hw3))
    h = _m.sqrt(wh3/wh)
    w = _m.sqrt(hw3/wh)
    return rect(cx-w, cy-h, cx+w, cy+h)
register_approx(Moments2d, rect, approx_moments_to_rect)
