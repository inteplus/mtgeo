# distutils: language = c++
# distutils: language_level = 3

import numpy as _np

from libcpp cimport bool
from libc.math cimport fabs, hypot, atan2, sin, cos, M_PI, M_PI_2


__all__ = ['feq', 'radian_range', 'ssr2mat', 'mat2ssr', 'rss2mat', 'mat2rss', 'lin2']


cpdef bool feq(double a, double b, double eps=1e-06):
    '''Checks if two scalars are nearly equal.'''
    return fabs(a-b) < eps


cpdef double radian_range(double rad):
    '''Makes sure a radian value is in range [-pi,+pi). Only works for a value not too far from 0.'''
    while rad >= M_PI:
        rad -= M_PI
    while rad < -M_PI:
        rad += M_PI
    return rad


# Redundant vector (sx,sy,h,r,cr,sr).
# Matrix 2x2 parametrised as [a0,a1,a2,a3].


cpdef void ssr2mat(double[:] i, double[:] o):
    '''Converts redundant vector (sx,sy,h,r,cr,sr) representing scale2d(sx,sy) shear2d(h) rotate2d(r) into a 2x2 transformation matrix.
    
    Formula::

        a0 = sx*(h*sr + cr)
        a1 = sx*(h*cr - sr)
        a2 = sy*sr
        a3 = sy*cr

    Parameters
    ----------
    i : input 6-vector
        redundant vector (sx,sy,h,r,cr,sr)
    o : output 4-vector
        flattened 2x2 matrix
    '''
    cdef double sx, sy, h, r, cr, sr
    sx = i[0]
    sy = i[1]
    h = i[2]
    r = i[3]
    cr = i[4]
    sr = i[5]
    o[0] = sx*(h*sr + cr)
    o[1] = sx*(h*cr - sr)
    o[2] = sy*sr
    o[3] = sy*cr


cpdef void mat2ssr(double[:] i, double[:] o):
    '''Converts a 2x2 transformation matrix into an redundant vector (sx,sy,h,r,cr,sr) representing scale2d(sx,sy) shear2d(h) rotate2d(r).
    
    Formula::

        sy = hypot(a2,a3)
        r = atan2(a2,a3)
        sx = det(A)/sy
        h = (a0*sr + a1*cr)/sx

    Parameters
    ----------
    i : input 4-vector
        flattened 2x2 matrix
    o : output 6-vector
        redundant vector (sx,sy,h,r,cr,sr)
    '''
    cdef double sx, sy, h, r, cr, sr
    sy = hypot(i[2],i[3])
    r = atan2(i[2],i[3])
    sr = i[2]/sy
    cr = i[3]/sy
    sx = (i[0]*i[3] - i[1]*i[2])/sy
    h = (i[0]*sr + i[1]*cr)/sx
    o[0] = sx
    o[1] = sy
    o[2] = h
    o[3] = r
    o[4] = cr
    o[5] = sr


cpdef void rss2mat(double[:] i, double[:] o):
    '''Converts redundant vector (sx,sy,h,r,cr,sr) representing rotate2d(r) shear2d(h) scale2d(sx,sy) into a 2x2 transformation matrix.
    
    Formula::

        a0 = sx*sr
        a1 = sy*(h*cr - sr)
        a2 = sx*cr
        a3 = sy*(h*sr + cr)

    Parameters
    ----------
    i : input 6-vector
        redundant vector (sx,sy,h,r,cr,sr)
    o : output 4-vector
        flattened 2x2 matrix
    '''
    cdef double sx, sy, h, r, cr, sr
    sx = i[0]
    sy = i[1]
    h = i[2]
    r = i[3]
    cr = i[4]
    sr = i[5]
    o[0] = sx*sr
    o[1] = sy*(h*cr - sr)
    o[2] = sx*cr
    o[3] = sy*(h*sr + cr)


cpdef void mat2rss(double[:] i, double[:] o):
    '''Converts a 2x2 transformation matrix into an redundant vector (sx,sy,h,r,cr,sr) representing rotate2d(r) shear2d(h) scale2d(sx,sy).
    
    Formula when `det(A) > 0`::

        sx = hypot(a2,a0)
        r = atan2(a2,a0)
        sy = det(A)/sx
        h = (a3*sr + a1*cr)/sy

    Formula when `det(A) < 0`::

        sx = -hypot(a2,a0)
        r = atan2(-a2,-a0)
        sy = det(A)/sx
        h = (a3*sr + a1*cr)/sy

    Parameters
    ----------
    i : input 4-vector
        flattened 2x2 matrix
    o : output 6-vector
        redundant vector (sx,sy,h,r,cr,sr)
    '''
    cdef double sx, sy, h, r, cr, sr, det
    det = i[0]*i[3] - i[1]*i[2]
    if det > 0:
        sx = hypot(i[2],i[0])
        r = atan2(i[2],i[0])
    else:
        sx = -hypot(i[2],i[0])
        r = atan2(-i[2],-i[0])
    sr = i[2]/sx
    cr = i[0]/sx
    sy = det/sx
    h = (i[3]*sr + i[1]*cr)/sy
    o[0] = sx
    o[1] = sy
    o[2] = h
    o[3] = r
    o[4] = cr
    o[5] = sr


cdef class lin2(object):
    '''Linear transformation in 2D.

    The 2D linear transformation in this class is parametrised as the following::

        lin2(sx, sy, h, r) = scale2d(sx,sy) shear2d(h) rotate2d(r)

    where `sx != 0` and `sy > 0`, `scale2d(sx,sy) = [[sx, 0], [0, sy]]`, `shear2d(h) = [[1, h], [0, 1]]` and `rotate2d(r) = [[cr, -sr], [sr, cr]]` with `cr = cos(r)` and `sr = sin(r)`. We call this parametrisation as the `ssr` parametrisation.

    There is a reverse parametrisation::

        lin2_reverse(sx, sy, h, r) = rotate2d(r) shear2d(h) scale2d(sx,sy)

    with the same conditions, which we call the `rss` parametrisation. If `lin2(sx,sy,h,r) = lin2_reverse(sx',sy',h'r')` then inv(lin2(sx,sy,h,r)) = lin2(1/sx',1/sy',-h',-r')`.

    Examples
    --------

    >>> import numpy as np
    >>> import math as m
    >>> import mt.geo.linear2d as gl
    >>> a = gl.lin2()
    >>> a
    lin2(scale=[1. 1.], shear=0.0, angle=0.0)
    >>> ~a
    lin2(scale=[1. 1.], shear=0.0, angle=0.0)
    >>> a = gl.lin2(shear=1)
    >>> a
    lin2(scale=[1. 1.], shear=1.0, angle=0.0)
    >>> a*a
    lin2(scale=[1. 1.], shear=2.0, angle=0.0)
    >>> a/a
    lin2(scale=[1. 1.], shear=0.0, angle=0.0)
    >>> a%a
    lin2(scale=[1. 1.], shear=0.0, angle=0.0)
    >>> ~a
    lin2(scale=[1. 1.], shear=-1.0, angle=-0.0)
    >>> a = gl.lin2(scale=[-3,1])
    >>> a
    lin2(scale=[-3.  1.], shear=0.0, angle=0.0)
    >>> ~a
    lin2(scale=[-0.33333333  1.        ], shear=0.0, angle=0.0)
    >>> a*a
    lin2(scale=[9. 1.], shear=0.0, angle=0.0)
    >>> ~a/a
    lin2(scale=[0.11111111 1.        ], shear=0.0, angle=0.0)
    >>> a = gl.lin2(angle=m.pi/6)
    >>> a
    lin2(scale=[1. 1.], shear=0.0, angle=0.5235987755982988)
    >>> a.matrix
    array([[ 0.8660254, -0.5      ],
           [ 0.5      ,  0.8660254]])
    >>> a*a
    lin2(scale=[1. 1.], shear=0.0, angle=1.0471975511965976)
    >>> (a*a).matrix
    array([[ 0.5      , -0.8660254],
           [ 0.8660254,  0.5      ]])
    >>> a = gl.lin2(scale=[-2,3], shear=4, angle=1)
    >>> a
    lin2(scale=[-2.  3.], shear=4.0, angle=1.0)
    >>> a.matrix
    array([[-7.81237249, -2.63947648],
           [ 2.52441295,  1.62090692]])
    >>> a/a*a
    lin2(scale=[-2.  3.], shear=3.9999999999999982, angle=0.9999999999999998)


    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    # ----- C/C++ vars -----

    cdef double[6] m_buf

    # ----- static methods -----

    @staticmethod
    def from_matrix(mat):
        '''Obtains a lin2 instance from a non-singular transformation matrix.

        Parameters
        ----------
        mat : a 2x2 array
            non-singular transformation matrix

        Returns
        -------
        lin2
            An instance representing the transformation

        Notes
        -----
        For speed reasons, no checking is involved.
        '''
        cdef double[4] i = [mat[0,0], mat[0,1], mat[1,0], mat[1,1]]
        cdef double[6] o
        mat2ssr(i,o)
        return lin2(scale=[o[0], o[1]], shear=o[2], angle=[o[3], o[4], o[5]])

    # ----- data encapsulation -----

    @property
    def sx(self):
        return self.m_buf[0]

    @property
    def sy(self):
        return self.m_buf[1]

    @property
    def scale(self):
        return _np.array([self.m_buf[0], self.m_buf[1]])

    @scale.setter
    def scale(self, scale):
        if len(scale) != 2:
            raise ValueError("Scale is not a pair: {}".format(scale))
        if not (scale[0] != 0):
            raise ValueError(
                "The first scale component ({}) is not non-zero.".format(scale[0]))
        if not (scale[1] > 0):
            raise ValueError(
                "The second scale component ({}) is not positive.".format(scale[1]))
        self.m_buf[0] = scale[0]
        self.m_buf[1] = scale[1]

    @property
    def shear(self):
        return self.m_buf[2]

    @shear.setter
    def shear(self, shear):
        self.m_buf[2] = shear

    @property
    def angle(self):
        return self.m_buf[3]

    @property
    def cos_angle(self):
        return self.m_buf[4]

    @property
    def sin_angle(self):
        return self.m_buf[5]

    @angle.setter
    def angle(self, angle):
        if isinstance(angle, (float, int)):
            self.m_buf[3] = angle
            self.m_buf[4] = cos(angle)
            self.m_buf[5] = sin(angle)
        else:
            self.m_buf[3] = angle[0]
            self.m_buf[4] = angle[1]
            self.m_buf[5] = angle[2]

    # ----- derived properties -----

    @property
    def matrix(self):
        '''Returns the linear transformation matrix.'''
        cdef double[4] m
        ssr2mat(self.m_buf, m)
        return _np.array([[m[0], m[1]], [m[2], m[3]]])

    @property
    def det(self):
        '''Returns the determinant of the transformation matrix.'''
        return self.m_buf[0]*self.m_buf[1]

    # ----- methods -----

    def __init__(self, scale=_np.ones(2), shear=0, angle=0):
        self.scale = scale
        self.shear = shear
        self.angle = angle

    def __repr__(self):
        return "lin2(scale={}, shear={}, angle={})".format(self.scale, self.shear, self.angle)

    def __lshift__(self, x):
        '''left shift = Lie action'''
        if x.shape != (2,):
            raise ValueError("Input shape {} is not (2,).".format(x.shape))
        return _np.dot(self.matrix, x)

    def __rshift__(self, x):
        '''right shift = inverted Lie action'''
        return (~self) << x

    def __mul__(self, other):
        '''a*b = Lie operator'''
        return lin2.from_matrix(_np.dot(self.matrix, other.matrix))

    def __invert__(self):
        '''Lie inverse'''
        cdef double[4] m
        cdef double[6] rss
        ssr2mat(self.m_buf, m)
        mat2rss(m, rss)
        return lin2(scale=[1/rss[0], 1/rss[1]], shear=-rss[2], angle=[-rss[3], rss[4], -rss[5]])

    def __truediv__(self, other):
        '''a/b = a*(~b)'''
        return self*(~other)

    def __mod__(self, other):
        '''a%b = (~a)*b'''
        return (~self)*other

    def conjugate(self, other):
        '''Conjugate: `self.conjugate(other) = self*other*self^{-1}`'''
        return self*(other/self)
