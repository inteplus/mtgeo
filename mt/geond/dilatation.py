from mt import np
import mt.base.casting as _bc

from ..geo import register_transform, register_transformable
from .point_list import PointList
from .moments import Moments
from .affine import Aff


__all__ = ['Dlt']


class Dlt(Aff):
    '''Dilatation (scaling and translation) in n-dim space. Here, scaling is per dimension, not
    uniform scaling.

    A dilatation maps a point `x` to `scale * x + offset` (element-wise product), so it is an affine
    transformation whose weight matrix is `diag(scale)` and whose bias is `offset`. It is cheaper
    than :class:`Aff` and is the representation behind :class:`Hyperbox`. A negative scale flips
    the corresponding axis. The inherited operators `*`, `/`, `%`, `~`, `<<` and `>>` work as in
    :class:`Aff`; multiplying two dilatations gives a dilatation, while mixing with another affine
    transformation gives an :class:`Aff`.

    Parameters
    ----------
    offset : numpy.ndarray
        the 1D translation vector. Defaults to the zero vector of length 3.
    scale : numpy.ndarray
        the 1D vector of per-dimension scales. Defaults to the ones vector of length 3.
    check_shapes : bool
        whether to check that `offset` and `scale` have the same length, raising `ValueError`
        otherwise

    Attributes
    ----------
    offset : numpy.ndarray
        the offset vector, same as `bias`
    scale : numpy.ndarray
        the scale vector. The weight matrix is `weight = diag(scale)`.

    See Also
    --------
    Aff : general affine transformation
    Hyperbox : axis-aligned box defined via a dilatation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geond.dilatation import Dlt
    >>> a = Dlt(offset=np.array([1,2]), scale=np.array([3,4]))
    >>> ~a
    Dlt(offset=[-0.33333333 -0.5       ], scale=[0.33333333 0.25      ])
    >>> a*~a
    Dlt(offset=[0. 0.], scale=[1. 1.])
    >>> a/a
    Dlt(offset=[0. 0.], scale=[1. 1.])
    >>> a%a
    Dlt(offset=[0. 0.], scale=[1. 1.])
    >>> b = Dlt(offset=np.array([1,0.5]), scale=np.array([1/3,0.25]))
    >>> a*b
    Dlt(offset=[4. 4.], scale=[1. 1.])
    >>> b*a
    Dlt(offset=[1.33333333 1.        ], scale=[1. 1.])
    >>> a/b
    Dlt(offset=[-8. -6.], scale=[ 9. 16.])
    >>> b/a
    Dlt(offset=[0.88888889 0.375     ], scale=[0.11111111 0.0625    ])
    >>> a << np.ones(2)
    array([4., 6.])
    >>> a.weight
    array([[3, 0],
           [0, 4]])
    >>> a.matrix
    array([[3., 0., 1.],
           [0., 4., 2.],
           [0., 0., 1.]])
    '''

    # ----- base adaptation -----

    @property
    def dim(self):
        return self.bias_dim

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
        return np.diag(self.scale)
    weight.__doc__ = Aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only.")

    @property
    def weight_shape(self):
        dim = self.scale_dim
        return (dim,dim)
    weight_shape.__doc__ = Aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):
        '''The offset/bias vector of the dilatation (1D numpy array).

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
        '''The per-dimension scale vector of the dilatation (1D numpy array).

        The setter raises `ValueError` if the value is not a 1D array.
        '''
        return self.__scale

    @scale.setter
    def scale(self, scale):
        if len(scale.shape) != 1:
            raise ValueError(f"Scale is not a vector, shape {scale.shape}.")
        self.__scale = scale

    @property
    def scale_dim(self):
        return self.__scale.shape[0]

    # ----- methods -----

    def __init__(self, offset=np.zeros(3), scale=np.ones(3), check_shapes=True):
        self.offset = offset
        self.scale = scale
        if check_shapes:
            _ = self.dim # just to check the shapes

    def __repr__(self):
        return f"Dlt(offset={self.offset}, scale={self.scale})"

    def multiply(self, other):
        if not isinstance(other, Dlt):
            return super(Dlt, self).multiply(other)
        return Dlt(self << other.offset, self.scale*other.scale)
    multiply.__doc__ = Aff.multiply.__doc__

    def invert(self):
        return Dlt(-self.offset/self.scale, 1/self.scale)
    invert.__doc__ = Aff.invert.__doc__


# ----- casting -----


_bc.register_cast(Dlt, Aff, lambda x: Aff(weight=x.weight, bias=x.bias, check_shapes=False))
_bc.register_cast(Aff, Dlt, lambda x: Dlt(offset=x.bias, scale=np.diagonal(x.weight)))
_bc.register_castable(
    Aff, Dlt, lambda x: np.count_nonzero(x.weight - np.diag(np.diagonal(x.weight))) == 0
)


# ----- transform functions -----


def transform_Dlt_on_Moments(dlt_tfm, moments):
    '''Transform the Moments using a dilatation.

    The mean becomes `scale * mean + offset`, the covariance matrix becomes
    `diag(scale) @ cov @ diag(scale)` and the 0th moment is multiplied by `abs(prod(scale))`. It is
    the function registered to apply a :class:`Dlt` to a :class:`Moments`.

    Parameters
    ----------
    dlt_tfm : Dlt
        general dilatation
    moments : Moments
        general moments

    Returns
    -------
    Moments
        dilated moments
    '''
    a = dlt_tfm.scale
    A = np.diag(a)
    old_m0 = moments.m0
    old_mean = moments.mean
    old_cov = moments.cov
    new_mean = a*old_mean + dlt_tfm.offset
    new_cov = A @ old_cov @ A.T
    new_m0 = old_m0*abs(np.prod(a))
    new_m1 = new_m0*new_mean
    new_m2 = new_m0*(np.outer(new_mean, new_mean) + new_cov)
    return Moments(new_m0, new_m1, new_m2)
register_transform(Dlt, Moments, transform_Dlt_on_Moments)


def transform_Dlt_on_ndarray(dlt_tfm, point_array):
    '''Transform an array of points using a dilatation.

    It computes `point_array * scale + offset` with broadcasting over the last axis. It is the
    function registered to apply a :class:`Dlt` to a `numpy.ndarray`.

    Parameters
    ----------
    dlt_tfm : Dlt
        general dilatation
    point_array : numpy.ndarray
        a point array whose last dimension has the same length as the dimensionality of the
        transformation

    Returns
    -------
    numpy.ndarray
        dilated point array, of the same shape as the input
    '''
    return point_array * dlt_tfm.scale + dlt_tfm.offset
register_transform(Dlt, np.ndarray, transform_Dlt_on_ndarray)
register_transformable(Dlt, np.ndarray, lambda x, y: x.ndim == y.shape[-1])


def transform_Dlt_on_PointList(dlt_tfm, point_list):
    '''Transform a point list using a dilatation.

    It is the function registered to apply a :class:`Dlt` to a :class:`PointList`.

    Parameters
    ----------
    dlt_tfm : Dlt
        general dilatation
    point_list : PointList
        a point list

    Returns
    -------
    PointList
        dilated point list
    '''
    return PointList(point_list.points * dlt_tfm.scale + dlt_tfm.offset, check=False)
register_transform(Dlt, PointList, transform_Dlt_on_PointList)
