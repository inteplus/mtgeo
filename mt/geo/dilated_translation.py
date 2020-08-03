import numpy as _np
import numpy.linalg as _nl
import math as _m

import mt.base.casting as _bc
from .approximation import register_approx
from .transformation import register_transform, register_transformable
from .dilated_isometry import Dliso


__all__ = ['dliso', 'Dliso', 'approx_Dliso_to_Dltra']


class Dltra(Dliso):
    '''Dilated translation = Translation following a uniform scaling.

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky MT is trying to advertise his paper!)
    '''

    # ----- base adaptation -----

    offset.__doc__ = '''The offset/bias part of the dilated translation.'''

    @property
    def unitary(self):
        '''The unitary matrix of the dilated translation.'''
        return _np.identity(self.ndim) # returns identity

    @unitary.setter
    def unitary(self, unitary):
        raise TypeError("Unitary matrix is the identity matrix and is read-only.")

    # ----- derived properties -----

    @property
    def linear(self):
        '''Returns the linear part of the affine transformation matrix of the dilated translation.'''
        return _np.identity(self.ndim)*self.scale

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), scale=1):
        self.offset = offset
        self.scale = scale

    def __repr__(self):
        return "Dltra(offset={}, scale={})".format(self.offset, self.scale)

    # ----- base adaptation -----

    def multiply(self, other):
        if not isinstance(other, Dltra):
            return super(Dltra, self).multiply(other)
        return Dltra(self << other.offset,
            self.scale*other.scale)
    multiply.__doc__ = Dliso.multiply.__doc__

    def invert(self):
        invScale = 1/self.scale
        return Dliso(self.offset*(-invScale), invScale)
    invert.__doc__ = Dliso.invert.__doc__


# ----- casting -----


def approx_Dliso_to_Dltra(obj):
    '''Approximates an Dliso instance wiht a Dltra by ignoring the unitary part.'''
    return Dltra(offset=obj.offset, scale=obj.scale)


_bc.register_cast(Dltra, Dliso, lambda x: Dliso(offset=x.offset, scale=x.scale))
register_approx(Dliso, Dltra, approx_Dliso_to_Dltra)


# ----- transform functions ------


def transform_Dltra_on_ndarray(dlt_tfm, point_array):
    '''Transform an array of points using a dilated transalation.

    Parameters
    ----------
    dlt_tfm : Dltra
        a dilated translation
    point_array : numpy.ndarray with last dimension having the same length as the dimensionality of the transformation
        a point array

    Returns
    -------
    numpy.ndarray
        dilated-translated point array
    '''
    return point_array*dlt_tfm.scale + dlt_tfm.bias
register_transform(Dltra, _np.ndarray, transform_Dltra_on_ndarray)
register_transformable(Dltra, _np.ndarray, lambda x, y: x.ndim == y.shape[-1])


