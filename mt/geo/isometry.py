import numpy as _np
import numpy.linalg as _nl
import math as _m

import mt.base.casting as _bc
from mt.base.deprecated import deprecated_func

from .dilated_isometry import Dliso, Aff


__all__ = ['Iso', 'iso', 'rotate2d', 'translate2d']


class Iso(Dliso):
    '''Isometry = orthogonal transformation followed by translation.

    An isometry is a (Euclidean-)metric-preserving transformation. In other words, it is an affine transformation but the linear part is a unitary matrix.

    References:
        [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky MT is trying to advertise his paper!)
    '''

    # ----- base adaptation -----

    @property
    def scale(self):
        return 1
    scale.__doc__ = Dliso.scale.__doc__

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
        return "Iso(offset={}, unitary_diagonal={})".format(self.offset, self.unitary.diagonal())

    # ----- base adaptation -----

    def multiply(self, other):
        if not isinstance(other, Iso):
            return super(Iso, self).__mul__(other)
        return Iso(offset=self << other.offset, unitary=_np.dot(self.unitary, other.unitary))
    multiply.__doc__ = Dliso.multiply.__doc__

    def invert(self):
        invUnitary = _nl.inv(self.unitary) # slow, and assuming the unitary matrix is invertible
        return Iso(offset=_np.dot(invUnitary, -self.offset), unitary=invUnitary)
    invert.__doc__ = Dliso.invert.__doc__


class iso(Iso):

    __doc__ = Iso.__doc__

    @deprecated_func("0.4.3", suggested_func='mt.geo.isometry.Iso.__init__', removed_version="0.6.0")
    def __init__(self, *args, **kwargs):
        super(iso, self).__init__(*args, **kwargs)


# ----- casting -----


_bc.register_cast(Iso, Dliso, lambda x: Dliso(offset=x.offset, unitary=x.unitary))
_bc.register_cast(Iso, Aff, lambda x: Aff(bias=x.offset, weight=x.weight))


# ----- useful 2D transformations -----


@deprecated_func("0.3.1", suggested_func="mt.geo.affine2d.rotate2d", removed_version="0.5.0", docstring_prefix="    ")
def rotate2d(theta):
    '''Returns the rotation. Theta is in radian.'''
    return Iso(
        offset=_np.zeros(2),
        unitary=_np.array([
            [_np.cos(theta), -_np.sin(theta)],
            [_np.sin(theta), _np.cos(theta)]]))


@deprecated_func("0.3.1", suggested_func="mt.geo.affine2d.translate2d", removed_version="0.5.0", docstring_prefix="    ")
def translate2d(x,y):
    '''Returns the translation.'''
    return Iso(offset=_np.array((x,y)), unitary=_np.identity(2))
