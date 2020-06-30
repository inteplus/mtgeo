import numpy as _np
import numpy.linalg as _nl
import math as _m

from .dilated_isometry import dliso


__all__ = ['iso', 'rotate2d', 'translate2d']


class iso(dliso):
    '''Isometry = orthogonal transformation followed by translation.

    An isometry is a (Euclidean-)metric-preserving transformation. In other words, it is an affine transformation but the linear part is a unitary matrix.

    References:
        [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky MT is trying to advertise his paper!)
    '''

    # ----- base adaptation -----

    @property
    def scale(self):
        return 1
    scale.__doc__ = dliso.scale.__doc__

    @scale.setter
    def scale(self, scale):
        raise TypeError("Scale value is read-only.")

    @property
    def linear(self):
        '''Returns the linear part of the affine transformation matrix of the isometry.'''
        return self.unitary

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), unitary=_np.identity(2)):
        self.offset = offset
        self.unitary = unitary

    def __repr__(self):
        return "iso(offset={}, unitary_diagonal={})".format(self.offset, self.unitary.diagonal())

    def __mul__(self, other):
        if not isinstance(other, iso):
            return super(iso, self).__mul__(other)
        return iso(offset=self << other.offset, unitary=_np.dot(self.unitary, other.unitary))
    __mul__.__doc__ = dliso.__mul__.__doc__

    def __invert__(self):
        invUnitary = _nl.inv(self.unitary) # slow, and assuming the unitary matrix is invertible
        return iso(offset=_np.dot(invUnitary, -self.offset), unitary=invUnitary)
    __invert__.__doc__ = dliso.__invert__.__doc__


# ----- useful 2D transformations -----

def rotate2d(theta):
    '''Returns the rotation. Theta is in radian.'''
    if not rotate2d.warned:
        print("mt.geo.isometry.rotate2d() is deprecated. Use mt.geo.affine2d.rotate2d() instead.")
        rotate2d.warned = True
    return iso(
        offset=_np.zeros(2),
        unitary=_np.array([
            [_np.cos(theta), -_np.sin(theta)],
            [_np.sin(theta), _np.cos(theta)]]))
rotate2d.warned = False

def translate2d(x,y):
    '''Returns the translation.'''
    if not translate2d.warned:
        print("mt.geo.isometry.translate2d() is deprecated. Use mt.geo.affine2d.translate2d() instead.")
        translate2d.warned = True
    return iso(offset=_np.array((x,y)), unitary=_np.identity(2))
translate2d.warned = False
