import math as _m
from mt import np
import mt.base.casting as _bc
from ..geo.approximation import register_approx
from .affine import Aff


__all__ = ['Dliso', 'approx_Aff_to_Dliso']


class Dliso(Aff):
    '''Dilated isometry = Isometry following a uniform scaling.

    An isometry is a (Euclidean-)metric-preserving transformation. In other words, it is an affine
    transformation but the linear part is a unitary matrix. A dilated isometry maps a point `x` to
    `scale * (unitary @ x) + offset`, so its weight (linear) matrix is `scale * unitary`. The
    inherited operators `*`, `/`, `%`, `~`, `<<` and `>>` work as in :class:`Aff`; the product of
    two dilated isometries is a dilated isometry.

    Parameters
    ----------
    offset : numpy.ndarray
        the 1D translation vector. Defaults to the zero vector of length 2.
    scale : float
        the positive uniform scale. A `ValueError` is raised if it is not positive. Defaults to 1.
    unitary : numpy.ndarray
        the 2D unitary (orthogonal) matrix, assumed but not checked to be unitary. Defaults to the
        2x2 identity matrix.

    Attributes
    ----------
    offset : numpy.ndarray
        the offset vector, same as `bias`
    scale : float
        the uniform scale
    unitary : numpy.ndarray
        the unitary matrix
    linear : numpy.ndarray
        the weight matrix `unitary * scale`, same as `weight`

    See Also
    --------
    Aff : general affine transformation
    Iso : isometry, the special case of unit scale
    Dltra : dilated translation, the special case of identity unitary matrix

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geond.dilated_isometry import Dliso
    >>> R = np.array([[0., -1.], [1., 0.]])  # rotation by 90 degrees
    >>> s = Dliso(offset=np.array([1., 2.]), scale=2., unitary=R)
    >>> s
    Dliso(offset=[1. 2.], scale=2.0, unitary_diagonal=[0. 0.])
    >>> s.linear
    array([[ 0., -2.],
           [ 2.,  0.]])
    >>> s << np.array([1., 0.])
    array([1., 4.])
    >>> s.matrix
    array([[ 0., -2.,  1.],
           [ 2.,  0.,  2.],
           [ 0.,  0.,  1.]])
    >>> s * ~s
    Dliso(offset=[0. 0.], scale=1.0, unitary_diagonal=[1. 1.])

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky MT is trying to advertise his paper!)
    '''

    # ----- base adaptation -----

    @property
    def bias(self):
        return self.__offset
    bias.__doc__ = Aff.bias.__doc__

    @bias.setter
    def bias(self, bias):
        if len(bias.shape) != 1:
            raise ValueError(f"Bias is not a vector, shape {bias.shape}.")
        self.__offset = bias

    @property
    def bias_dim(self):
        return self.__offset.shape[0]
    bias_dim.__doc__ = Aff.bias_dim.__doc__

    @property
    def weight(self):
        return self.linear
    weight.__doc__ = Aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only.")

    @property
    def weight_shape(self):
        return self.__unitary.shape
    weight_shape.__doc__ = Aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):
        '''The offset/bias vector of the dilated isometry (1D numpy array).

        The setter raises `ValueError` if the value is not a 1D array.
        '''
        return self.__offset

    @offset.setter
    def offset(self, offset):
        if len(offset.shape) != 1:
            raise ValueError(f"Offset is not a vector, shape {offset.shape}.")
        self.__offset = offset

    @property
    def scale(self):
        '''The uniform positive scale of the dilated isometry.

        The setter raises `ValueError` if the value is not positive.
        '''
        return self.__scale

    @scale.setter
    def scale(self, scale):
        if not scale > 0:
            raise ValueError(f"Scale is not positive {scale}.")
        self.__scale = scale

    @property
    def unitary(self):
        '''The unitary matrix of the dilated isometry (2D numpy array).

        The setter raises `ValueError` if the value is not a 2D array, but does not check that it
        is unitary.
        '''
        return self.__unitary

    @unitary.setter
    def unitary(self, unitary):
        if len(unitary.shape) != 2:
            raise ValueError(f"Unitary is not a matrix of, shape {unitary.shape} given.")
        self.__unitary = unitary

    # ----- derived properties -----

    @property
    def linear(self):
        '''Returns the linear part of the affine transformation matrix of the dilated isometry,
        which is `unitary * scale`.'''
        return self.unitary*self.scale

    # ----- methods -----

    def __init__(self, offset=np.zeros(2), scale=1, unitary=np.identity(2)):
        self.offset = offset
        self.scale = scale
        self.unitary = unitary

    def __repr__(self):
        return (
            f"Dliso(offset={self.offset}, scale={self.scale}, "
            f"unitary_diagonal={self.unitary.diagonal()})"
        )

    # ----- base adaptation -----

    def multiply(self, other):
        if not isinstance(other, Dliso):
            return super(Dliso, self).multiply(other)
        return Dliso(self << other.offset,
            self.scale*other.scale,
            np.dot(self.unitary, other.unitary)
            )
    multiply.__doc__ = Aff.multiply.__doc__

    def invert(self):
        invScale = 1/self.scale
        # slow, and assuming the unitary matrix is invertible
        invUnitary = np.linalg.inv(self.unitary)
        return Dliso(np.dot(invUnitary, -self.offset*invScale), invScale, invUnitary)
    invert.__doc__ = Aff.invert.__doc__


# ----- casting -----


def approx_Aff_to_Dliso(obj):
    '''Approximates an Aff instance with a Dliso.

    The approximation keeps the offset, takes the geometric mean of the singular values of the
    weight matrix as the scale and the product `U @ VT` of the singular-vector matrices as the
    unitary matrix. It is registered as the approximation from :class:`Aff` to :class:`Dliso`.

    Parameters
    ----------
    obj : Aff
        the affine transformation to approximate

    Returns
    -------
    Dliso
        the approximating dilated isometry

    Raises
    ------
    ValueError
        if the weight matrix is singular, as :class:`Dliso` requires a positive scale
    '''
    ndim = obj.ndim
    U, S, VT = np.linalg.svd(obj.weight, full_matrices=True)
    if len(S) < ndim or S[ndim-1] == 0:
        return Dliso(offset=obj.bias, scale=0, unitary=np.identity(ndim))
    
    scale = _m.pow(S.prod(), 1/ndim)
    return Dliso(offset=obj.bias, scale=scale, unitary=U@VT)


_bc.register_cast(Dliso, Aff, lambda x: Aff(weight=x.weight, bias=x.offset, check_shapes=False))
register_approx(Aff, Dliso, approx_Aff_to_Dliso)
