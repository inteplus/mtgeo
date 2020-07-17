'''A 2D rectangle.'''

import math as _m
import numpy as _np

from .box import box
from .moments2d import EPSILON, moments2d

__all__ = ['rect']

class rect(box):
    '''A 2D rectangle,

    Note we do not care if the rectangle is open or partially closed or closed.'''

    
    # ----- derived properties -----

    
    @property
    def min_x(self):
        return self.min_coords[0]

    @property
    def min_y(self):
        return self.min_coords[1]

    @property
    def max_x(self):
        return self.max_coords[0]

    @property
    def max_y(self):
        return self.max_coords[1]

    @property
    def x(self):
        return self.min_x

    @property
    def y(self):
        return self.min_y

    @property
    def w(self):
        return self.max_x - self.min_x

    @property
    def h(self):
        return self.max_y - self.min_y

    @property
    def cx(self):
        return (self.min_x + self.max_x)/2

    @property
    def cy(self):
        return (self.min_y + self.max_y)/2

    @property
    def area(self):
        return self.w*self.h

    
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
        '''Computes all moments, up to 2nd-order of the rectangle's interior.'''
        m0 = self.signed_area
        m1 = [self.moment_x, self.moment_y]
        mxy = self.moment_xy
        m2 = [[self.moment_xx, mxy], [mxy, self.moment_yy]]
        return moments2d(m0, m1, m2)

    @staticmethod
    def from_moments2d(obj):
        '''Returns a rectangle that best approximates the moments2d instance.

        Parameters
        ----------
        obj : moments2d
            an instance containing moments of 2D points up to 2nd order

        Returns
        -------
        rect
            the output rectangle
        '''
        cx, cy = obj.mean
        cov = obj.cov

        # w = half width, h = half height
        hw3 = cov[0][0]*obj.m0*0.75
        wh3 = cov[1][1]*obj.m0*0.75
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
