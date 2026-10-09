"""Affine transformation in ND.

The module provides :class:`Aff`, an affine map `x -> weight @ x + bias` in an n-dimensional
space, together with the functions that apply it to numpy point arrays, point lists and moments.
"""

from mt import np
from mt.base import logger
from mt.base.deprecated import deprecated_func

from ..geo import (
    GeometricObject,
    LieTransformer,
    transform,
    register_transform,
    register_transformable,
)
from .point_list import PointList
from .moments import Moments


__all__ = ["Aff", "transform_Aff_on_Moments", "transform_Aff_on_PointList"]


class Aff(LieTransformer, GeometricObject):
    """A transformer to perform affine transformations using the same transformation matrix in n-dim
    space.

    The transformation maps a point `x` to `weight @ x + bias`, where `weight` is a square matrix
    and `bias` a vector of the same size, whose length is the dimension `dim` (also `ndim`) of the
    space. Transformations are composed with `a*b` (apply `b` first, then `a`), inverted with `~a`,
    and applied to objects with `a << obj` (or undone with `a >> obj`). It can transform numpy
    arrays whose last axis has length `dim`, :class:`PointList` and :class:`Moments`. As a
    :class:`mt.geo.LieTransformer` it also supports `a/b`, `a%b` and `a.conjugate(b)`.

    Parameters
    ----------
    weight : numpy.ndarray
        the 2D weight/linear matrix. Defaults to the 3x3 identity matrix.
    bias : numpy.ndarray
        the 1D bias/translation vector. Defaults to the zero vector of length 3.
    check_shapes : bool
        whether to check that `weight` is a square matrix whose size matches the length of `bias`.
        If the check fails, a `ValueError` is raised.

    See Also
    --------
    mt.geo3d.Aff3d : the 3D version, implemented in glm
    Dlt : dilatation, a special case with a diagonal weight matrix
    Dliso : dilated isometry, a special case with a uniformly scaled unitary weight matrix

    Examples
    --------
    >>> from mt import np, geond as gn
    >>> a = gn.Aff(weight=np.array([[1,-1],[-2,3]]), bias=np.array([3,4]))
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
    >>> b = gn.Aff(weight=np.array([[1,-1],[-3,2]]), bias=np.array([2,1]))
    >>> a*b
    Aff(weight_diagonal=[4 8], bias=[4 3])
    >>> b*a
    Aff(weight_diagonal=[3 9], bias=[1 0])
    >>> a.conjugate(b)
    Aff(weight_diagonal=[ 6. -3.], bias=[-18.  66.])
    >>> a << np.array([1, 1])
    array([3, 5])
    >>> a >> (a << np.array([1., 1.]))
    array([1., 1.])
    >>> a.det
    1.0
    """

    # ----- base adaptation -----

    @property
    def ndim(self):
        return self.dim

    def invert(self):
        """Returns the inverse transformation `x -> inv(weight) @ (x - bias)`.

        The weight matrix is assumed to be invertible. A numpy `LinAlgError` is raised if it is
        singular.

        Returns
        -------
        Aff
            the inverse transformation
        """
        invWeight = np.linalg.inv(
            self.weight
        )  # slow, and assuming weight matrix is invertible
        return Aff(invWeight, np.dot(invWeight, -self.bias))

    def multiply(self, other):
        """Composes two affine transformations: `a*b` applies `b` first, then `a`.

        The result has weight `a.weight @ b.weight` and bias `a << b.bias`.

        Parameters
        ----------
        other : Aff
            the transformation to apply first

        Returns
        -------
        Aff
            the composed transformation

        Raises
        ------
        ValueError
            if `other` is not an instance of :class:`Aff`
        """
        if not isinstance(other, Aff):
            raise ValueError(
                f"Expecting 'other' to be an affine transformation, but {other.__class__} received."
            )
        return Aff(np.dot(self.weight, other.weight), self << other.bias)

    # ----- data encapsulation -----

    @property
    def bias(self):
        """The bias/translation vector of the affine transformation (1D numpy array).

        The setter raises `ValueError` if the value is not a 1D array.
        """
        return self.__bias

    @bias.setter
    def bias(self, bias):
        if len(bias.shape) != 1:
            raise ValueError(f"Bias is not a vector, shape {bias.shape}.")
        self.__bias = bias

    @property
    def weight(self):
        """The weight/linear matrix of the affine transformation (2D numpy array).

        The setter raises `ValueError` if the value is not a 2D array.
        """
        return self.__weight

    @weight.setter
    def weight(self, weight):
        if len(weight.shape) != 2:
            raise ValueError(f"bias has non-matrix shape {weight.shape}.")
        self.__weight = weight

    # ----- derived properties -----

    @property
    def bias_dim(self):
        """Returns the dimension (length) of the bias vector."""
        return self.bias.shape[0]

    @property
    def weight_shape(self):
        """Returns the shape of the weight matrix, as a tuple."""
        return self.weight.shape

    @property
    def dim(self):
        """Returns the dimension of the transformation, which is the length of the bias vector.

        Raises
        ------
        ValueError
            if the weight matrix is not square with the same size as the bias vector
        """
        val = self.bias_dim
        if self.weight_shape != (val, val):
            raise ValueError(
                f"Weight {self.weight.shape} does not have a square matrix shape {(val, val)}."
            )
        return val

    @property
    def matrix(self):
        """Returns the homogeneous transformation matrix as a `(dim+1, dim+1)` float array.

        The top-left block is the weight matrix, the last column holds the bias and the last row is
        `[0, ..., 0, 1]`.
        """
        dim = self.dim
        a = np.empty((dim + 1, dim + 1))
        a[:dim, :dim] = self.weight
        a[:dim, dim] = self.bias
        a[dim, :dim] = 0
        a[dim, dim] = 1
        return a

    @property
    def det(self):
        """Returns the determinant of the weight matrix. Its absolute value is the factor by which
        volumes are scaled by the transformation."""
        return np.linalg.det(self.weight)  # slow method

    # ----- methods -----

    def __init__(self, weight=np.identity(3), bias=np.zeros(3), check_shapes=True):
        self.weight = weight
        self.bias = bias
        if check_shapes:
            _ = self.dim  # just to check shapes

    def __repr__(self):
        return f"Aff(weight_diagonal={self.weight.diagonal()}, bias={self.bias})"


# ----- transform functions -----


def transform_Aff_on_Moments(aff_tfm, moments):
    """Transform the Moments using an affine transformation.

    The mean becomes `A @ mean + bias`, the covariance matrix becomes `A @ cov @ A.T` and the 0th
    moment is multiplied by `abs(det A)`, where `A` is the weight matrix. It is the function
    registered to apply an :class:`Aff` to a :class:`Moments`.

    Parameters
    ----------
    aff_tfm : Aff
        general affine transformation
    moments : Moments
        general moments

    Returns
    -------
    Moments
        affined-transformed moments
    """
    A = aff_tfm.weight
    old_m0 = moments.m0
    old_mean = moments.mean
    old_cov = moments.cov
    new_mean = A @ old_mean + aff_tfm.bias
    new_cov = A @ old_cov @ A.T
    new_m0 = old_m0 * abs(aff_tfm.det)
    new_m1 = new_m0 * new_mean
    new_m2 = new_m0 * (np.outer(new_mean, new_mean) + new_cov)
    return Moments(new_m0, new_m1, new_m2)


register_transform(Aff, Moments, transform_Aff_on_Moments)


def transform_Aff_on_ndarray(aff_tfm, point_array):
    """Transform an array of points using an affine transformation.

    It computes `point_array @ weight.T + bias`, so each row (or last-axis vector) of the array is
    a point. It is the function registered to apply an :class:`Aff` to a `numpy.ndarray`.

    Parameters
    ----------
    aff_tfm : Aff
        general affine transformation
    point_array : numpy.ndarray
        a point array whose last dimension has the same length as the dimensionality of the
        transformation

    Returns
    -------
    numpy.ndarray
        affine-transformed point array, of the same shape as the input
    """
    return point_array @ aff_tfm.weight.T + aff_tfm.bias


register_transform(Aff, np.ndarray, transform_Aff_on_ndarray)
register_transformable(Aff, np.ndarray, lambda x, y: x.ndim == y.shape[-1])


def transform_Aff_on_PointList(aff_tfm, point_list):
    """Transform a point list using an affine transformation.

    It is the function registered to apply an :class:`Aff` to a :class:`PointList`.

    Parameters
    ----------
    aff_tfm : Aff
        general affine transformation
    point_list : PointList
        a point list

    Returns
    -------
    PointList
        affine-transformed point list
    """
    return PointList(point_list.points @ aff_tfm.weight.T + aff_tfm.bias, check=False)


register_transform(Aff, PointList, transform_Aff_on_PointList)
