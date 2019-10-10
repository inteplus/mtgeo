import numpy as _np
import math as _m

from .affine_transformation import aff
from .linear2d import lin2


class aff2(aff):
    '''Affine transformation in 2D.

    The 2D affine transformation defined here consists of a linear/weight part and an offset/bias part.

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    # ----- static methods -----

    @staticmethod
    def from_matrix(mat):
        '''Obtains an aff2 instance from a non-singular affine transformation matrix.

        Parameters
        ----------
        mat : a 3x3 array
            a non-singular affine transformation matrix

        Returns
        -------
        aff2
            An instance representing the transformation

        Notes
        -----
        For speed reasons, no checking is involved.
        '''
        return aff2(offset=mat[:2,2], linear=lin2.from_matrix(mat[:2,:2]))

    # ----- base adaptation -----

    @property
    def bias(self):
        return self.__offset
    bias.__doc__ = aff.bias.__doc__

    @bias.setter
    def bias(self, bias):
        raise TypeError("Bias vector is read-only. Use self.offset vector instead.")

    @property
    def bias_dim(self):
        return 2
    bias_dim.__doc__ = aff.bias_dim.__doc__

    @property
    def weight(self):
        return self.linear.matrix
    weight.__doc__ = aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only. Use self.linear instead.")

    @property
    def weight_shape(self):
        return (2,2)
    weight_shape.__doc__ = aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):  
        return self.__offset

    @offset.setter
    def offset(self, offset):  
        if len(offset.shape) != 1 or offset.shape[0] != 2:
            raise ValueError(
                "Offset is not a 2D vector, shape {}.".format(offset.shape))
        self.__offset = offset

    @property
    def linear(self):
        return self.__linear

    @linear.setter
    def linear(self, linear):
        if not isinstance(linear, lin2):
            raise ValueError(
                "Expected a lin2 instance. Received a '{}' instance.".format(linear.__class__))
        self.__linear = linear

    # ----- derived properties -----

    @property
    def matrix(self):  
        a = _np.empty((3,3))
        a[:2,:2] = self.linear.matrix
        a[:2,2] = self.offset
        a[2,:2] = 0
        a[2,2] = 1
        return a
    matrix.__doc__ = aff.matrix.__doc__

    @property
    def det(self):
        return self.linear.det
    det.__doc__ = aff.det.__doc__

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), linear=lin2()):
        self.offset = offset
        self.linear = linear

    def __repr__(self):
        return "aff2(offset={}, linear={})".format(self.offset, self.linear)

    def __mul__(self, other):
        if not isinstance(other, aff2):
            return super(aff2, self).__mul__(other)
        return aff2(
            offset=(self.linear << other.offset) + self.offset,
            linear=self.linear*other.linear)
    __mul__.__doc__ = aff.__mul__.__doc__

    def __invert__(self):
        invLinear = ~self.linear
        invOffset = invLinear << (-self.offset)
        return aff2(offset=invOffset, linear=invLinear)
    __invert__.__doc__ = aff.__invert__.__doc__

# ----- useful 2D transformations -----

def fliplr2d(width):
    '''Returns a fliplr for a given width.'''
    return aff2.from_matrix(_np.array([
        [-1, 0, width],
        [ 0, 1, 0]]))

def flipud2d(height):
    '''Returns a flipud for a given height.'''
    return aff2.from_matrix(_np.array([
        [1,  0, 0],
        [0, -1, height]]))

def shear2d(theta):
    '''Returns the shearing. Theta is a scalar.'''
    return aff2(shear=theta)

def originate2d(tfm, x, y):
    '''Tweaks a 2D affine transformation so that it acts as if it originates at (x,y) instead of (0,0).'''
    return aff2(offset=_np.array((x, y))).conjugate(tfm)
