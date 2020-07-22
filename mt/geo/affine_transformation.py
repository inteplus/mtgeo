import numpy as _np
import numpy.linalg as _nl

from mt.base import logger
from mt.base.deprecated import deprecated_func

from .object import GeometricObject
from .transform import LieTransformer, transform, register_transform


__all__ = ['aff', 'Aff']


class Aff(LieTransformer, GeometricObject):
    '''A transformer to perform affine transformations using the same transformation matrix in n-dim space.

    Examples
    --------
    >>> import numpy as _np
    >>> import mt.geo as _mg
    >>> a = _mg.Aff(weight=_np.array([[1,-1],[-2,3]]), bias=_np.array([3,4]))
    >>> a.bias
    array([3, 4])
    >>> a.weight
    array([[ 1, -1],
           [-2,  3]])
    >>> a.matrix
    array([[ 1., -1.,  3.],
           [-2.,  3.,  4.],
           [ 0.,  0.,  1.]])
    >>> ~a
    Aff(weight_diagonal=[3. 1.], bias=[-13. -10.])
    >>> a*~a
    Aff(weight_diagonal=[1. 1.], bias=[0. 0.])
    >>> ~a*a
    Aff(weight_diagonal=[1. 1.], bias=[0. 0.])
    >>> a/a
    Aff(weight_diagonal=[1. 1.], bias=[0. 0.])
    >>> a%a
    Aff(weight_diagonal=[1. 1.], bias=[0. 0.])
    >>> a*a
    Aff(weight_diagonal=[ 3 11], bias=[ 2 10])
    >>> a*a*a
    Aff(weight_diagonal=[11 41], bias=[-5 30])
    >>> (a*a)*a
    Aff(weight_diagonal=[11 41], bias=[-5 30])
    >>> a*(a*a)
    Aff(weight_diagonal=[11 41], bias=[-5 30])
    >>> b = _mg.Aff(weight=_np.array([[1,-1],[-3,2]]), bias=_np.array([2,1]))
    >>> a*b
    Aff(weight_diagonal=[4 8], bias=[4 3])
    >>> b*a
    Aff(weight_diagonal=[3 9], bias=[1 0])
    >>> a.conjugate(b)
    Aff(weight_diagonal=[ 6. -3.], bias=[-18.  66.])
    '''

    # ----- base adaptation -----

    @property
    def ndim(self):
        return self.dim

    def invert(self):
        '''Lie inverse'''
        invWeight = _nl.inv(
            self.weight)  # slow, and assuming weight matrix is invertible
        return Aff(invWeight, _np.dot(invWeight, -self.bias))

    def multiply(self, other):
        '''a*b = Lie operator'''
        if not isinstance(other, aff):
            raise ValueError(
                "Expecting 'other' to be an affine transformation, but {} received.".format(other.__class__))
        return Aff(_np.dot(self.weight, other.weight), self << other.bias)

    # ----- data encapsulation -----

    @property
    def bias(self):
        '''The bias component of the affine transformation matrix.'''
        return self.__bias

    @bias.setter
    def bias(self, bias):
        if len(bias.shape) != 1:
            raise ValueError(
                "Bias is not a vector, shape {}.".format(bias.shape))
        self.__bias = bias

    @property
    def weight(self):
        '''The weight/linear component of the affine transformation matrix.'''
        return self.__weight

    @weight.setter
    def weight(self, weight):
        if len(weight.shape) != 2:
            raise ValueError(
                "bias has non-matrix shape {}.".format(weight.shape))
        self.__weight = weight

    # ----- derived properties -----

    @property
    def bias_dim(self):
        '''Returns the dimension of the bias vector.'''
        return self.bias.shape[0]

    @property
    def weight_shape(self):
        '''Returns the shape of the weight matrix.'''
        return self.weight.shape

    @property
    def dim(self):
        '''Returns the dimension of the transformation.'''
        val = self.bias_dim
        if self.weight_shape != (val, val):
            raise ValueError(
                "Weight does not have a square matrix shape {}.".format(self.weight.shape))
        return val

    @property
    def matrix(self):
        '''Returns the transformation matrix.'''
        dim = self.dim
        a = _np.empty((dim+1, dim+1))
        a[:dim, :dim] = self.weight
        a[:dim, dim] = self.bias
        a[dim, :dim] = 0
        a[dim, dim] = 1
        return a

    @property
    def det(self):
        '''Returns the determinant of the transformation matrix.'''
        return _nl.det(self.weight)  # slow method

    # ----- methods -----

    def __init__(self, weight=_np.identity(3), bias=_np.zeros(3), check_shapes=True):
        self.weight = weight
        self.bias = bias
        if check_shapes:
            _ = self.dim  # just to check shapes

    def __repr__(self):
        return "Aff(weight_diagonal={}, bias={})".format(self.weight.diagonal(), self.bias)

    def __lshift__(self, x):
        '''left shift = Lie action'''
        if x.shape != (self.dim,):
            raise ValueError(
                "Input shape {} is not ({},).".format(x.shape, self.dim))
        return _np.dot(self.weight, x)+self.bias

    def transform_points(self, X):
        '''Transforms a list of points.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (N,D)
            transforms N points. D is the dimensionality of the transformation

        Returns
        -------
        X2 : numpy.ndarray of shape (N,D)
            transformed N points, where `X2 = np.dot(X, self.weight^T) + self.bias`
        '''
        if len(X.shape) != 2 or X.shape[1] != self.dim:
            raise ValueErro("Input shape {} is not (N,{}).".format(X.shape, self.dim))
        return _np.dot(X, self.weight.T) + self.bias

    def weight_sign(self, eps=1e-06):
        '''Returns whether weight determinant is positive (+1), close to zero (0), or negative (-1).'''
        det = self.det
        return 1 if det > eps else -1 if det < -eps else 0

aff = Aff # for backward compatibility


# MT-TODO: register_transform the left shift operator and transform_points() function of Aff


# ----- obsolete useful 2D transformations -----

@deprecated_func("0.3.4", suggested_func="mt.geo.affine2d.shear2d", removed_version="0.5.0", docstring_prefix="    ") 
def shear2d(theta):
    '''Returns the shearing. Theta is in radian.'''
    return Aff(weight=_np.array([
        [1, -_np.sin(theta)],
        [0, _np.cos(theta)]]),
        bias=_np.zeros(2))

@deprecated_func("0.3.4", suggested_func="mt.geo.affine2d.originate2d", removed_version="0.5.0", docstring_prefix="    ") 
def originate2d(tfm, x, y):
    '''Tweaks an affine transformation so that it acts as if it originates at (x,y) instead of (0,0).'''
    return Aff(weight=_np.identity(2), bias=_np.array((x, y))).conjugate(tfm)
