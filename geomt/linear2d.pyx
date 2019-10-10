# distutils: language = c++

import numpy as _np

from libcpp cimport bool
from libc.math cimport fabs, hypot, atan2, sin, cos, M_PI, M_PI_2


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


cdef class lin2(object):
    '''Linear transformation in 2D.

    #skimage.transform.AffineTransform>`_. However, we rip the translation part from it.
    We follow skimage's parametrization of 2D affine transformations `skimage.transform.AffineTransformation <https://scikit-image.org/docs/dev/api/skimage.transform.html

    The 2D linear transformation has the following form::

        X = a0*x + a1*y
        X = sx*cos(r)*x - sy*sin(r + h)*y
        Y = a2*x + a3*y
        Y = sx*sin(r)*x + sy*cos(r + h)*y
        mat = [[a0, a1], [a2, a3]]

    where `s=(sx,sy)` are the scaling parameters, `r` is the rotation angle (in radian), `h` is the shearing angle (in radian). To make the parametrization unique, we assume both `sx`, `sy` are positive, both `r` and `h` are in range '[-pi, +pi)' and `h` is not 'pi/2' or '-pi/2' (at which point the transformation matrix is singular). We then obtain the inverse form::

        sx = hypot(a0, a2)
        sy = hypot(a1, a3)
        r = atan2(a2, a0)
        h = atan2(a1, a3) - r

    Multiplication of two 2D linear transformations is a bit involved. Suppose the two 2D linear transformations are (sx_a, sy_a, r_a, h_a) and (sx_b, sy_b, r_b, h_b). Let the product of two corresponding matrices be [[c0, c1], [c2, c3]]. First, we rewrite each row of mat(sx_a, sy_a, r_a, h_a) as::

        (lx, qx) = trig2(sx_a*cos(r_a), sy_a*sin(r_a + h_a))
        (ly, qy) = trig2(sy_a*cos(r_a + h_a), sx_a*sin(r_a))
        [a0, a1] = [lx*cos(qx), -lx*sin(qx)]
        [a2, a3] = [ly*sin(qy),  ly*cos(qy)]

    Key equation:

        cos(qx-qy) = 0 <=> cos(h_a) = 0

    Multiplying it with the second matrix, we obtain::

        c0 =  lx*sx_b*cos(qx + r_b)
        c1 = -lx*sy_b*sin(qx + r_b + h_b)
        c2 =  ly*sx_b*sin(qy + r_b)
        c3 =  ly*sy_b*cos(qy + r_b + h_b)

    From the above equations, if we assume (sx_b, sy_b, r_b, h_b) is the inverse of (sx_a, sy_a, r_a, h_a), then we obtain a closed form::

        r_b  = -qy or pi - qy, whichever that makes cos(qx + r_b) positive
        h_b  = qy - qx or pi + qy - qx, whichever that makes cos(qy + r_b + h_b) positive
        sx_b = 1/|cos(qx - qy)|/lx
        sy_b = 1/|cos(qx - qy)|/ly

    Examples
    --------

    >>> import numpy as _np
    >>> import math as _m
    >>> import geomt.linear2d as _gl
    >>> a = gl.lin2()
    >>> a
    lin2(scale=[1. 1.], angle=0.0, shear=0.0)
    >>> ~a
    lin2(scale=[1. 1.], angle=0.0, shear=0.0)
    >>> (~a).matrix
    array([[ 1., -0.],
           [-0.,  1.]])
    >>> a = gl.lin2(scale=(0.4, 5))
    >>> a
    lin2(scale=[0.4 5. ], angle=0.0, shear=0.0)
    >>> ~a
    lin2(scale=[2.5 0.2], angle=-0.0, shear=0.0)
    >>> a = gl.lin2(angle=10)
    >>> a
    lin2(scale=[1. 1.], angle=0.5752220392306207, shear=0.0)
    >>> ~a
    lin2(scale=[1. 1.], angle=-0.5752220392306207, shear=0.0)
    >>> a = gl.lin2(shear=1)
    >>> a
    lin2(scale=[1. 1.], angle=0.0, shear=1.0)
    >>> ~a
    lin2(scale=[1.         2.41889182], angle=-0.0, shear=2.4420710092412734)


    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    # ----- C/C++ vars -----

    cdef double m_sx, m_sy, m_r, m_h
    cdef bool m_dirty
    cdef double m_lx, m_ly, m_qx, m_qy

    cdef cleanse(self):
        cdef double c0, c1, s0, s1

        if self.m_dirty:
            c0 = self.m_sx*cos(self.m_r)
            s0 = self.m_sy*sin(self.m_r + self.m_h)
            c1 = self.m_sy*cos(self.m_r + self.m_h)
            s1 = self.m_sx*sin(self.m_r)
            self.m_lx = hypot(c0, s0)
            self.m_qx = atan2(s0, c0)
            self.m_ly = hypot(c1, s1)
            self.m_qy = atan2(s1, c1)
            self.m_dirty = False

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
        cdef double a0, a1, a2, a3
        cdef double sx, sy, r, h

        a0 = mat[0, 0]
        a1 = mat[0, 1]
        a2 = mat[1, 0]
        a3 = mat[1, 1]

        sx = hypot(a0, a2)
        sy = hypot(a1, a3)
        r = radian_range(atan2(a2, a0))
        h = radian_range(atan2(a1, a3) - r)

        return lin2(scale=_np.array([sx, sy]), angle=r, shear=h)

    # ----- data encapsulation -----

    @property
    def sx(self):
        return self.m_sx

    @property
    def sy(self):
        return self.m_sy

    @property
    def scale(self):
        return _np.array([self.m_sx, self.m_sy])

    @scale.setter
    def scale(self, scale):
        if len(scale) != 2:
            raise ValueError("Scale is not a pair: {}".format(scale))
        if not (scale[0] > 0):
            raise ValueError(
                "The first scale component ({}) is not positive.".format(scale[0]))
        if not (scale[1] > 0):
            raise ValueError(
                "The second scale component ({}) is not positive.".format(scale[1]))
        self.m_sx = scale[0]
        self.m_sy = scale[1]
        self.m_dirty = True

    @property
    def angle(self):
        return self.m_r

    @angle.setter
    def angle(self, angle):
        self.m_r = radian_range(angle)
        self.m_dirty = True

    @property
    def shear(self):
        return self.m_h

    @shear.setter
    def shear(self, shear):
        cdef double h = radian_range(shear)
        if feq(h, M_PI_2):
            raise ValueError(
                "Singular case detected. Shearing of right angle is not accepted.")
        if feq(h, -M_PI_2):
            raise ValueError(
                "Singular case detected. Shearing of negative right angle is not accepted.")
        self.m_h = h
        self.m_dirty = True

    # ----- derived properties -----

    @property
    def matrix(self):
        '''Returns the linear transformation matrix.'''
        return _np.array([
            [self.m_sx*cos(self.m_r), -self.m_sy*sin(self.m_r + self.m_h)],
            [self.m_sx*sin(self.m_r),  self.m_sy*cos(self.m_r + self.m_h)]])

    # ----- methods -----

    def __init__(self, scale=_np.ones(2), angle=0, shear=0):
        self.scale = scale
        self.angle = angle
        self.shear = shear

    def __repr__(self):
        return "lin2(scale={}, angle={}, shear={})".format(self.scale, self.angle, self.shear)

    def __lshift__(self, x):
        '''left shift = Lie action'''
        if x.shape != (2,):
            raise ValueError("Input shape {} is not (2,).".format(x.shape))
        return _np.dot(self.to_matrix(), x)

    def __rshift__(self, x):
        '''right shift = inverted Lie action'''
        return (~self) << x

    cpdef __cmul__(self, lin2 other):  # hack
        cdef double b_sx, b_sy, b_r, b_h

        b_sx = other.m_sx
        b_sy = other.m_sy
        b_r = other.m_r
        b_h = other.m_h

        self.cleanse()

        # MT-NOTE: this is a bit slow.
        return lin2.from_matrix(_np.array([
            [self.m_lx*b_sx*cos(self.m_qx + b_r), -self.m_lx *
             b_sy*sin(self.m_qx + b_r + b_h)],
            [self.m_ly*b_sx*sin(self.m_qy + b_r),  self.m_ly*b_sy*cos(self.m_qy + b_r + b_h)]]))

    def __mul__(self, other):
        '''a*b = Lie operator'''
        return self.__cmul__(other)

    def __invert__(self):
        '''Lie inverse'''
        cdef double b_r, b_h, g

        self.cleanse()

        # Non-singularity implies: `cos(qx-qy) |? 0 <=> cos(h_a) |? 0` where `|?` can be '=', '<' and '>'

        # rotation and shearing angle
        # cos(h_a) and cos(qx-qy) > 0
        if self.m_h > -M_PI_2 and self.m_h < M_PI_2:
            # so that sin(qy + r_b) = 0 and cos(qx + r_b) > 0
            b_r = radian_range(-self.m_qy)
            # so that sin(qx + r_b + h_b) = 0 and cos(qy + r_b + h_b) > 0
            b_h = radian_range(M_PI + self.m_qy - self.m_qx)
            g = 1/cos(self.m_qx - self.m_qy)
        else:  # cos(h_a) and cos(qx-qy) < 0
            # so that sin(qy + r_b) = 0 and cos(qx + r_b) > 0
            b_r = radian_range(M_PI - self.m_qy)
            # so that sin(qx + r_b + h_b) = 0 and cos(qy + r_b + h_b) > 0
            b_h = radian_range(self.m_qy - self.m_qx)
            g = -1/cos(self.m_qx - self.m_qy)

        return lin2(scale=_np.array([g/self.m_lx, g/self.m_ly]), angle=b_r, shear=b_h)

    def __truediv__(self, other):
        '''a/b = a*(~b)'''
        return self*(~other)

    def __mod__(self, other):
        '''a%b = (~a)*b'''
        return (~self)*other

    def conjugate(self, other):
        '''Conjugate: `self.conjugate(other) = self*other*self^{-1}`'''
        return self*(other/self)
