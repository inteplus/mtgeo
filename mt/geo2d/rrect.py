'''A 2D rotated rectangle.'''

import math

from mt import np
from mt.base.casting import *

from ..geo import GeometricObject, TwoD, register_approx
from .moments import Moments2d
from .linear import Lin2d
from .rect import Rect


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

    Notes
    -----

    For more details, see `this page <https://structx.com/Shape_Formulas_033.html>`_. But note that
    we primarily use CV convention for images, top left is (0,0).
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
        if not hasattr(self, '_br'):
            self._br = self.transform(np.ones(2))
        return self._br

    @property
    def tr(self):
        '''The transform of point (1,0).'''
        if not hasattr(self, '_tr'):
            self._tr = self.transform(np.array([1,0]))
        return self._tr

    @property
    def bl(self):
        '''The transform of point (0,1).'''
        if not hasattr(self, '_bl'):
            self._bl = self.transform(np.array([0,1]))
        return self._bl

    @property
    def sign(self):
        return np.sign(self.lin2d.det)

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
        if not hasattr(self, '_center_pt'):
            self._center_pt = self.transform(np.array([0.5,0.5]))
        return self._center_pt

    @property
    def area(self):
        '''Absolute area.'''
        return abs(self.lin2d.det)

    @property
    def circumference(self):
        '''Circumference.'''
        return (self.w+self.h)*2

    
    # ----- moments -----


    @property
    def signed_area(self):
        '''Returns the signed area of the rectangle.'''
        return self.lin2d.sx*self.lin2d.sy

    @property
    def moment1(self):
        '''First-order moment.'''
        if not hasattr(self, '_moment1'):
            self._moment1 = self.sign*self.center_pt
        return self._moment1

    @property
    def moment_x(self):
        '''Returns the integral of x over the rectangle's interior.'''
        return self.moment1[0]

    @property
    def moment_y(self):
        '''Returns the integral of y over the rectangle's interior.'''
        return self.moment1[1]

    @property
    def moment2(self):
        '''Second-order moment.'''
        if not hasattr(self, '_moment2'):
            # 2nd-order central moments if the rectangle were not rotated
            rx = self.width/2
            ry = self.height/2
            r = Rect(-rx, -ry, rx, ry)
            Muu = r.moment_xx
            Muv = r.moment_xy
            Mvv = r.moment_xy

            # Rotate the central moments (a.k.a. moments of inertia):
            #   Original axes: x, y
            #   Rotated axes:  u, v
            #   Dst-to-src change-of-coordinates formulae:
            #     x =  c*u+s*v
            #     y = -s*u+c*v
            #   where c = cos(theta), s = sin(theta). Therefore,
            #     Mxx =  c*c*Muu +     2*c*s*Muv + s*s*Mvv
            #     Mxy = -c*s*Muu + (c*c-s*s)*Muv + c*s*Mvv
            #     Myy =  s*s*Muu +    -2*c*s*Muv + c*c*Mvv
            # Reference: `link <https://calcresource.com/moment-of-inertia-rotation.html>`_
            c = self.lin2d.cos_angle
            s = self.lin2d.sin_angle
            cc = c*c
            cs = c*s
            ss = s*s
            Mxx =  cc*Muu +    2*cs*Muv + ss*Mvv
            Mxy = -cs*Muu + (cc-ss)*Muv + cs*Mvv
            Myy =  ss*Muu -    2*cs*Muv + cc*Mvv

            # Shift the origin to where it should be:
            #   Axes from the rectangle center point: x, y
            #   Image axes: p, q
            #   Dst-to-src chang-of-coordinates formulae:
            #     p = x+cx
            #     q = y+cy
            #   Formulae:
            #     Mpp = Mxx + 2*cx*Mx       + cx*cx*M1
            #     Mpq = Mxy + cx*My + cy*Mx + cx*cy*M1
            #     Mqq = Myy + 2*cy*My       + cy*cy*M1
            #   where M1 is the signed area. But because the rectangle is symmetric, Mx = My = 0.
            cx = self.center_pt[0]
            cy = self.center_pt[1]
            M1 = self.signed_area
            Mpp = Mxx + cx*cx*M1
            Mpq = Mxy + cx*cy*M1
            Mqq = Myy + cy*cy*M1

            self._moment2 = np.array([[Mpp, Mpq], [Mpq, Mqq]])
        return self._moment2

    @property
    def moment_xy(self):
        '''Returns the integral of x*y over the rectangle's interior.'''
        return self.moment2[0][1]

    @property
    def moment_xx(self):
        '''Returns the integral of x*x over the rectangle's interior.'''
        return self.moment2[0][0]

    @property
    def moment_yy(self):
        '''Returns the integral of y*y over the rectangle's interior.'''
        return self.moment2[1][1]


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
