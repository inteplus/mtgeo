import numpy as _np
import math as _m

from .object import TwoD
from .affine_transformation import Aff
from .linear2d import lin2


__all__ = ['Aff2d', 'aff2', 'swapAxes2d', 'flipLR2d', 'flipUD2d', 'shearX2d', 'shearY2d', 'originate2d', 'rotate2d', 'translate2d', 'scale2d', 'crop2d']


class Aff2d(TwoD, Aff):
    '''Affine transformation in 2D.

    The 2D affine transformation defined here consists of a linear/weight part and an offset/bias part.

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    # ----- static methods -----

    @staticmethod
    def from_matrix(mat):
        '''Obtains an Aff2d instance from a non-singular affine transformation matrix.

        Parameters
        ----------
        mat : a 3x3 array
            a non-singular affine transformation matrix

        Returns
        -------
        Aff2d
            An instance representing the transformation

        Notes
        -----
        For speed reasons, no checking is involved.
        '''
        return Aff2d(offset=mat[:2, 2], linear=lin2.from_matrix(mat[:2, :2]))

    # ----- base adaptation -----

    @property
    def ndim(self):
        return 2

    def multiply(self, other):
        if not isinstance(other, Aff2d):
            return super(Aff2d, self).__mul__(other)
        return Aff2d(
            offset=(self.linear << other.offset) + self.offset,
            linear=self.linear*other.linear)
    multiply.__doc__ = Aff.multiply.__doc__

    def invert(self):
        invLinear = ~self.linear
        invOffset = invLinear << (-self.offset)
        return Aff2d(offset=invOffset, linear=invLinear)
    invert.__doc__ = Aff.invert.__doc__

    @property
    def bias(self):
        return self.__offset
    bias.__doc__ = Aff.bias.__doc__

    @bias.setter
    def bias(self, bias):
        raise TypeError(
            "Bias vector is read-only. Use self.offset vector instead.")

    @property
    def bias_dim(self):
        return 2
    bias_dim.__doc__ = Aff.bias_dim.__doc__

    @property
    def weight(self):
        return self.linear.matrix
    weight.__doc__ = Aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only. Use self.linear instead.")

    @property
    def weight_shape(self):
        return (2, 2)
    weight_shape.__doc__ = Aff.weight_shape.__doc__

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
        a = _np.empty((3, 3))
        a[:2, :2] = self.linear.matrix
        a[:2, 2] = self.offset
        a[2, :2] = 0
        a[2, 2] = 1
        return a
    matrix.__doc__ = Aff.matrix.__doc__

    @property
    def det(self):
        return self.linear.det
    det.__doc__ = Aff.det.__doc__

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), linear=lin2()):
        self.offset = offset
        self.linear = linear

    def __repr__(self):
        return "Aff2d(offset={}, linear={})".format(self.offset, self.linear)


aff2 = Aff2d # for backward compatibility


# MT-TODO: write a cast function to convert aff to aff2


# ----- useful 2D transformations -----


def swapAxes2d():
    '''Returns the affine transformation that swaps the x-axis with the y-axis.'''
    return Aff2d(linear=lin2.from_matrix(_np.array([[0, 1], [1, 0]])))


def flipLR2d(width):
    '''Returns a left-right flip for a given width.'''
    return Aff2d.from_matrix(_np.array([
        [-1, 0, width],
        [0, 1, 0]]))


def flipUD2d(height):
    '''Returns a up-down flip for a given height.'''
    return Aff2d.from_matrix(_np.array([
        [1,  0, 0],
        [0, -1, height]]))


def shearX2d(h):
    '''Returns the shearing along the x-axis.'''
    return Aff2d(linear=lin2(shear=h))


def shearY2d(h):
    '''Returns the shearing along the y-axis.'''
    return Aff2d(linear=lin2.from_matrix(_np.array([[1, 0], [h, 1]])))


def originate2d(tfm, x, y):
    '''Tweaks a 2D affine transformation so that it acts as if it originates at (x,y) instead of (0,0).'''
    return Aff2d(offset=_np.array((x, y))).conjugate(tfm)


def rotate2d(theta, x, y):
    '''Returns the rotation about a reference point (x,y). Theta is in radian.'''
    return originate2d(Aff2d(angle=theta), x, y)


def translate2d(x, y):
    '''Returns the translation.'''
    return Aff2d(offset=_np.array([x, y]))


def scale2d(scale_x=1, scale_y=None):
    '''Returns the scaling.'''
    if scale_y is None:
        scale_y = scale_x
    return Aff2d(linear=lin2(scale=[scale_x, scale_y]))


def crop2d(tl, br=None):
    '''Transforms an axis-aligned rectangle into [(0,0),(1,1)].

    Parameters
    ----------
    tl : 2d point (x,y)
        coordinates to be mapped to (0,0) if `br` is specified. If `br` is not specified, then the transformation is 2d scaling. In other words, (0,0) is mapped to (0,0) and tl is mapped to (1,1).
    br : 2d point (x,y), optional
        If specified, coordinates to be mapped to (1,1).

    Returns
    -------
    Aff2d
        A transformation that maps points in [(0,0),(1,1)] to the crop given by `tl` and `br`.
    '''
    if br is None:
        return scale2d(1.0/tl[0], 1.0/tl[1])
    return Aff2d(offset=_np.array([-tl[0]/(br[0]-tl[0]), -tl[1]/(br[1]-tl[1])]), linear=lin2(scale=[1.0/(br[0]-tl[0]), 1.0/(br[1]-tl[1])]))
