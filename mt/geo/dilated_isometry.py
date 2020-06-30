import numpy as _np
import numpy.linalg as _nl
import math as _m

from .affine_transformation import aff


__all__ = ['dliso']


class dliso(aff):
    '''Dilated isometry = Isometry followed by a uniform scaling.

    An isometry is a (Euclidean-)metric-preserving transformation. In other words, it is an affine transformation but the linear part is a unitary matrix.

    References:
        [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky MT is trying to advertise his paper!)
    '''

    # ----- base adaptation -----

    @property
    def bias(self):
        return self.__offset
    bias.__doc__ = aff.bias.__doc__

    @bias.setter
    def bias(self, bias):
        if len(bias.shape) != 1:
            raise ValueError("Bias is not a vector, shape {}.".format(bias.shape))
        self.__offset = bias

    @property
    def bias_dim(self):
        return self.__offset.shape[0]
    bias_dim.__doc__ = aff.bias_dim.__doc__

    @property
    def weight(self):
        return self.linear
    weight.__doc__ = aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only.")

    @property
    def weight_shape(self):
        return self.__unitary.shape
    weight_shape.__doc__ = aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):
        '''The offset/bias part of the dilated isometry.'''
        return self.__offset

    @offset.setter
    def offset(self, offset):
        if len(offset.shape) != 1:
            raise ValueError("Offset is not a vector, shape {}.".format(offset.shape))
        self.__offset = offset

    @property
    def scale(self):
        '''The scale component/scalar of the dilated isometry.'''
        return self.__scale

    @scale.setter
    def scale(self, scale):
        if not scale > 0:
            raise ValueError("Scale is not positive {}.".format(scale))
        self.__scale = scale

    @property
    def unitary(self):
        '''The unitary matrix of the dilated isometry.'''
        return self.__unitary

    @unitary.setter
    def unitary(self, unitary):
        if len(unitary.shape) != 2:
            raise ValueError("Unitary is not a matrix of, shape {} given.".format(unitary.shape))
        self.__unitary = unitary

    # ----- derived properties -----

    @property
    def linear(self):
        '''Returns the linear part of the affine transformation matrix of the dilated isometry.'''
        return self.unitary*self.scale

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), scale=1, unitary=_np.identity(2)):
        self.offset = offset
        self.scale = scale
        self.unitary = unitary

    def __repr__(self):
        return "dliso(offset={}, scale={}, unitary_diagonal={})".format(self.offset, self.scale, self.unitary.diagonal())

    def __mul__(self, other):
        if not isinstance(other, dliso):
            return super(dliso, self).__mul__(other)
        return dliso(self << other.offset,
            self.scale*other.scale,
            _np.dot(self.unitary, other.unitary)
            )
    __mul__.__doc__ = aff.__mul__.__doc__

    def __invert__(self):
        invScale = 1/self.scale
        invUnitary = _nl.inv(self.unitary) # slow, and assuming the unitary matrix is invertible
        return dliso(_np.dot(invUnitary, -self.offset*invScale), invScale, invUnitary)
    __invert__.__doc__ = aff.__invert__.__doc__
