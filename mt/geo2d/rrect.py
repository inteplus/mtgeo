'''A 2D rotated rectangle.'''

import math

from mt import np
from mt.base.casting import *

from ..geo import GeometricObject, TwoD, register_approx
from .moments import Moments2d
from .linear import Lin2d


__all__ = ['RRect', 'cast_RRect_to_Moments2d', 'approx_Moments2d_to_RRect']


class RRect(TwoD, GeometricObject):
    '''A 2D rotated rectangle.

    An RRect is defined as the 2D affine transform of the unit square '[0,0,1,1]' where the
    transformation in the sshr representation has no shearing.  Note we do not care if the
    rectangle is open or partially closed or closed. Scaling along x-axis and y-axis allow
    flipping vertically and horizontally. Scaling with 0 coefficient is undefined.


    Parameters
    ----------
    lin2d : Lin2d
        the linear part of the affine transformation
    ofs2d : numpy.ndarray
        the transalation part of the affine transformation, or the position of the transform of
        point (0,0)
    force_valid : bool
        whether or not to reset the shearing to zero

    Attributes
    ----------
    tl : point
        the transform of point (0,0)
    br : point
        the transform of point (1,1)
    tr : point
        the transform of point (0,1)
    bl : point
        the transform of point (1,0)
    w : float
        the width, always non-negative
    h : float
        the height, always non-negative
    center_pt : point
        center point
    area : float
        the area
    circumference : float
        the circumference
    '''

    
    # ----- internal representations -----


    @property
    def shapely(self):
        '''Shapely representation for fast intersection operations.'''
        raise NotImplementedError("MT: to do one day")
        #if not hasattr(self, '_shapely'):
            #import shapely.geometry as _sg
            #self._shapely = _sg.box(self.min_x, self.min_y, self.max_x, self.max_y)
            #self._shapely = self._shapely.buffer(0.0001) # to clean up any (multi and/or non-simple) polygon into a simple polygon
        #return self._shapely


    # ----- derived properties -----\

    
    @property
    def tl(self):
        '''The transform of point (0,0).'''
        return self.ofs2d

    @property
    def br(self):
        '''The transform of point (1,1).'''
        return self.transform(np.ones(2))

    @property
    def tr(self):
        '''The transform of point (1,0).'''
        return self.transform(np.array([1,0]))

    @property
    def bl(self):
        '''The transform of point (0,1).'''
        return self.transform(np.array([0,1]))

    @property
    def w(self):
        '''width'''
        return abs(self.lin2d.sx)

    @property
    def h(self):
        '''height'''
        return abs(self.lin2d.sh)

    @property
    def center_pt(self):
        '''The transform of point (0.5, 0.5).'''
        return self.transform(np.array([0.5,0.5]))

    @property
    def area(self):
        '''Absolute area.'''
        return abs(self.w*self.h*math.sin(self.ofs2d.angle))

    @property
    def circumference(self):
        '''Circumference.'''
        return (abs(self.w)+abs(self.h))*2

    
    # ----- moments -----


    @property
    def signed_area(self):
        '''Returns the signed area of the rectangle.'''
        return self.lin2d.sx*self.lin2d.sy

    @property
    def moment_x(self):
        '''Returns the integral of x over the rectangle's interior.'''
        raise NotImplementedError("MT: to do one day")
        #return self.signed_area*self.cx

    @property
    def moment_y(self):
        '''Returns the integral of y over the rectangle's interior.'''
        raise NotImplementedError("MT: to do one day")
        #return self.signed_area*self.cy

    @property
    def moment_xy(self):
        '''Returns the integral of x*y over the rectangle's interior.'''
        raise NotImplementedError("MT: to do one day")
        #return self.moment_x*self.cy

    @property
    def moment_xx(self):
        '''Returns the integral of x*x over the rectangle's interior.'''
        raise NotImplementedError("MT: to do one day")
        #return self.signed_area*(self.min_x*self.min_x+self.min_x*self.max_x+self.max_x*self.max_x)/3

    @property
    def moment_yy(self):
        '''Returns the integral of y*y over the rectangle's interior.'''
        raise NotImplementedError("MT: to do one day")
        #return self.signed_area*(self.min_y*self.min_y+self.min_y*self.max_y+self.max_y*self.max_y)/3


    # ----- serialization -----


    def to_json(self):
        '''Returns a list [scale_x, scale_y, angle, ofs_x, ofs_y].'''
        return [self.lin2d.sx, self.lin2d.sy, self.lin2d.angle, self.ofs2d[0], self.ofs2d[1]]


    @staticmethod
    def from_json(json_obj):
        '''Creates a RRect from a JSON-like object.

        Parameters
        ----------
        json_obj : list
            list [scale_x, scale_y, angle, ofs_x, ofs_y]

        Returns
        -------
        RRect
            output rotated rectangle
        '''
        return RRect(Lin2d(scale=np.array([json_obj[0], json_obj[1]]), angle=json_obj[2]), ofs2d=np.array([json_obj[3], json_obj[4]]))


    def to_tensor(self):
        '''Returns a tensor [scale_x, scale_y, angle, ofs_x, ofs_y] representing the RRect .'''
        from mt import tf
        return tf.convert_to_tensor(self.to_json())

    
    # ----- methods -----

    
    def __init__(self, lin2d: Lin2d = Lin2d(), ofs2d: np.ndarray = np.zeros(2), force_valid: bool = False):
        if force_valid:
            lin2d = Lin2d(scale=lin2d.scale, angle=lin2d.angle)
        self.lin2d = lin2d
        self.ofs2d = ofs2d

    def __repr__(self):
        return "RRect(lin2d={}, ofs2d={})".format(self.lin2d, self.ofs2d)

    def transform(self, pt):
        return np.dot(pt, self.lin2d.matrix) + self.ofs2d


# ----- casting -----
        

def cast_RRect_to_Moments2d(obj):
    m0 = obj.signed_area
    m1 = [obj.moment_x, obj.moment_y]
    mxy = obj.moment_xy
    m2 = [[obj.moment_xx, mxy], [mxy, obj.moment_yy]]
    return Moments2d(m0, m1, m2)
register_cast(RRect, Moments2d, cast_RRect_to_Moments2d)


# ----- approximation -----


def approx_Moments2d_to_RRect(obj):
    '''Approximates a Moments2d instance with a RRect such that the mean aligns with the RRect's center, and the covariance matrix of the instance is closest to the moment convariance matrix of the RRect.'''
    raise NotImplementedError("MT-TODO: to implement one day")
    #cx, cy = obj.mean
    #cov = obj.cov

    ## w = half width, h = half height
    #size = abs(obj.m0)
    #hw3 = cov[0][0]*size*0.75 # should be >= 0
    #wh3 = cov[1][1]*size*0.75 # should be >= 0
    #wh = np.sqrt(np.sqrt(wh3*hw3))
    #h = np.sqrt(wh3/wh)
    #w = np.sqrt(hw3/wh)
    #return Rect(cx-w, cy-h, cx+w, cy+h)
register_approx(Moments2d, RRect, approx_Moments2d_to_RRect)
