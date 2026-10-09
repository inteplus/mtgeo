"""Affine transformation in 3D.

The module provides :class:`Aff3d`, a 3D affine map implemented in glm, the functions that apply it
to numpy arrays, point lists and moments, and helpers :func:`rot3d_x`, :func:`rot3d_y` and
:func:`rot3d_z` that build rotations about the coordinate axes.
"""

from mt import tp, np, glm
import mt.base.casting as _bc

from ..geo import ThreeD, register_transform, register_transformable
from ..geond import Aff
from .moments import Moments3d
from .point_list import PointList3d


__all__ = [
    "Aff3d",
    "transform_Aff3d_on_Moments3d",
    "transform_Aff3d_on_PointList3d",
    "rot3d_x",
    "rot3d_y",
    "rot3d_z",
]


class Aff3d(ThreeD, Aff):
    """Affine transformation in 3D.

    The 3D affine transformation defined here consists of a linear/weight part and an offset/bias
    part. It maps a point `x` to `linear @ x + offset`. The data are stored in glm in single
    precision (float32). As a :class:`mt.geo.LieTransformer` it supports composition `a*b` (apply
    `b` first), inversion `~a`, `a/b`, `a%b`, and transforms objects with `a << obj` (or undoes it
    with `a >> obj`). It can transform numpy arrays whose last axis has length 3.

    Parameters
    ----------
    offset : numpy.ndarray or glm.vec3
        the bias vector, of 3 elements. Defaults to the zero vector.
    linear : numpy.ndarray or glm.mat3
        the weight matrix. A numpy array is read as a row-major 3x3 matrix, while a glm matrix is
        column-major (it is the usual glm convention). Defaults to the identity matrix.

    Attributes
    ----------
    offset : glm.vec3
        the bias vector
    bias : numpy.ndarray
        the numpy view of `offset`
    linear : glm.mat3
        the weight matrix
    weight : numpy.ndarray
        the numpy view of `linear`, a row-major 3x3 matrix. It is read-only.
    matrix : numpy.ndarray
        the 4x4 homogeneous transformation matrix, `[[linear, offset], [0, 0, 0, 1]]`
    affine : glm.mat4
        the same matrix in glm
    det : float
        the determinant of `linear`

    See Also
    --------
    mt.geond.Aff : the dimension-agnostic version, implemented in numpy
    rot3d_x, rot3d_y, rot3d_z : rotations about the coordinate axes

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo3d import Aff3d
    >>> a = Aff3d(offset=np.array([1., 2., 3.]), linear=np.diag([2., 3., 4.]))
    >>> a
    Aff3d(offset=vec3( 1, 2, 3 ), linear=mat3x3(( 2, 0, 0 ), ( 0, 3, 0 ), ( 0, 0, 4 )))
    >>> a.matrix
    array([[2., 0., 0., 1.],
           [0., 3., 0., 2.],
           [0., 0., 4., 3.],
           [0., 0., 0., 1.]], dtype=float32)
    >>> a << np.array([1., 1., 1.])
    array([3., 5., 7.])
    >>> a << np.ones((2, 3))
    array([[3., 5., 7.],
           [3., 5., 7.]])
    >>> np.round(a >> np.array([3., 5., 7.]), 5)
    array([1., 1., 1.])
    >>> a * a
    Aff3d(offset=vec3( 3, 8, 15 ), linear=mat3x3(( 4, 0, 0 ), ( 0, 9, 0 ), ( 0, 0, 16 )))
    >>> a.det
    24.0
    >>> a * ~a
    Aff3d(offset=vec3( 0, 0, 0 ), linear=mat3x3(( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 )))

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    # ----- static methods -----

    @staticmethod
    def from_matrix(mat: np.ndarray):
        """Obtains an Aff3d instance from a non-singular affine transformation matrix.

        The offset is taken from the first 3 rows of the last column and the linear part from the
        top-left 3x3 block, so `Aff3d.from_matrix(a.matrix)` recovers `a`.

        Parameters
        ----------
        mat : numpy.ndarray
            a non-singular 4x4 homogeneous affine transformation matrix (3x4 is enough since the
            last row is not read)

        Returns
        -------
        Aff3d
            An instance representing the transformation

        Notes
        -----
        For speed reasons, no checking is involved.

        Examples
        --------
        >>> import numpy as np
        >>> from mt.geo3d import Aff3d
        >>> m = np.array([[1., 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7], [0, 0, 0, 1]])
        >>> Aff3d.from_matrix(m)
        Aff3d(offset=vec3( 5, 6, 7 ), linear=mat3x3(( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 )))
        """
        return Aff3d(offset=mat[:3, 3], linear=mat[:3, :3])

    # ----- base adaptation -----

    @property
    def ndim(self):  # reimplementation to enforce constantness
        return 3

    def multiply(self, other):
        if not isinstance(other, Aff3d):
            return super(Aff3d, self).multiply(other)
        return Aff3d(
            offset=self.linear * other.offset + self.offset,
            linear=self.linear * other.linear,
        )

    multiply.__doc__ = Aff.multiply.__doc__

    def invert(self):
        invLinear = glm.inverse(self.linear)
        invOffset = invLinear * (-self.offset)
        return Aff3d(offset=invOffset, linear=invLinear)

    invert.__doc__ = Aff.invert.__doc__

    @property
    def bias(self):
        return np.frombuffer(self.__offset.to_bytes(), dtype=np.float32)

    bias.__doc__ = Aff.bias.__doc__

    @bias.setter
    def bias(self, bias):
        raise TypeError("Bias vector is read-only. Use self.offset vector instead.")

    @property
    def bias_dim(self):
        return 3

    bias_dim.__doc__ = Aff.bias_dim.__doc__

    @property
    def weight(self):
        return np.frombuffer(self.__linear.to_bytes(), dtype=np.float32).reshape(3, 3).T

    weight.__doc__ = Aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only. Use self.linear instead.")

    @property
    def weight_shape(self):
        return (3, 3)

    weight_shape.__doc__ = Aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, value: tp.Union[np.ndarray, glm.vec3]):
        self.__offset = glm.vec3(value)

    @property
    def linear(self):
        return self.__linear

    @linear.setter
    def linear(self, value: tp.Union[np.ndarray, glm.mat3]):
        if isinstance(value, np.ndarray):
            self.__linear = glm.mat3(
                glm.vec3(value[:, 0]), glm.vec3(value[:, 1]), glm.vec3(value[:, 2])
            )
        else:
            self.__linear = glm.mat3(value)

    # ----- derived properties -----

    @property
    def affine(self):
        return glm.mat4(
            glm.vec4(self.linear[0], 0),
            glm.vec4(self.linear[1], 0),
            glm.vec4(self.linear[2], 0),
            glm.vec4(self.offset, 1),
        )

    @property
    def matrix(self):
        return np.array(self.affine)

    matrix.__doc__ = Aff.matrix.__doc__

    @property
    def det(self):
        return glm.determinant(self.linear)

    det.__doc__ = Aff.det.__doc__

    # ----- methods -----

    def __init__(
        self,
        offset=np.zeros(3),
        linear: tp.Union[np.ndarray, glm.mat3] = glm.mat3(),
    ):
        self.offset = offset
        self.linear = linear

    def __repr__(self):
        return f"Aff3d(offset={repr(self.offset)}, linear={repr(self.linear)})"


# ----- casting -----


_bc.register_cast(
    Aff3d, Aff, lambda x: Aff(weight=x.weight.copy(), bias=x.bias.copy(), check_shapes=False)
)
_bc.register_cast(Aff, Aff3d, lambda x: Aff3d(offset=x.bias, linear=x.weight))
_bc.register_castable(Aff, Aff3d, lambda x: x.ndim == 3)


# ----- transform functions -----


def transform_Aff3d_on_Moments3d(aff_tfm, moments):
    """Transform a Moments3d using a 3D affine transformation.

    The mean becomes `A*mean + offset`, the covariance matrix becomes `A*cov*A^T` and the 0th
    moment is multiplied by `abs(det A)`, where `A` is the linear part. It is the function
    registered to apply an :class:`Aff3d` to a :class:`Moments3d`.

    Parameters
    ----------
    aff_tfm : Aff3d
        3D affine transformation
    moments : Moments3d
        3D moments

    Returns
    -------
    Moments3d
        affined-transformed 3D moments
    """
    A = aff_tfm.linear
    old_m0 = moments.m0
    old_mean = moments.mean_glm
    old_cov = moments.cov_glm
    new_mean = A * old_mean + aff_tfm.offset
    new_cov = A * old_cov * glm.transpose(A)
    new_m0 = old_m0 * abs(aff_tfm.det)
    new_m1 = new_m0 * new_mean
    new_m2 = new_m0 * (glm.outerProduct(new_mean, new_mean) + new_cov)
    return Moments3d(new_m0, new_m1, new_m2)


register_transform(Aff3d, Moments3d, transform_Aff3d_on_Moments3d)


def transform_Aff3d_on_ndarray(aff_tfm, point_array):
    """Transform an array of 3D points using a 3D affine transformation.

    It computes `point_array @ weight.T + bias`, so each row (or last-axis vector) is a point. It is
    the function registered to apply an :class:`Aff3d` to a `numpy.ndarray`, which is allowed
    only if the last axis of the array has length 3.

    Parameters
    ----------
    aff_tfm : Aff3d or Aff
        a 3D affine transformation
    point_array : numpy.ndarray
        an array of 3D points, whose last dimension has length 3

    Returns
    -------
    numpy.ndarray
        affine-transformed point array, of the same shape as the input
    """
    return point_array @ aff_tfm.weight.T + aff_tfm.bias


register_transform(Aff3d, np.ndarray, transform_Aff3d_on_ndarray)
register_transformable(Aff3d, np.ndarray, lambda x, y: y.shape[-1] == 3)


def transform_Aff3d_on_PointList3d(aff_tfm, point_list):
    """Transform a 3D point list using a 3D affine transformation.

    It is the function registered to apply an :class:`Aff3d` to a :class:`PointList3d`.

    Parameters
    ----------
    aff_tfm : Aff3d
        a 3D affine transformation
    point_list : PointList3d
        a 3D point list

    Returns
    -------
    PointList3d
        affine-transformed point list
    """
    return PointList3d(point_list.points @ aff_tfm.weight.T + aff_tfm.bias, check=False)


register_transform(Aff3d, PointList3d, transform_Aff3d_on_PointList3d)


# ---- utilities -----


def rot3d_x(angle: float):
    """Gets 3D rotation about the x-axis (roll rotation).

    The rotation is right-handed: seen from the positive x-axis towards the origin, it is
    counter-clockwise for a positive angle, so it sends the y-axis towards the z-axis.

    Parameters
    ----------
    angle : float
        the angle to rotate, in radians

    Returns
    -------
    tfm : Aff3d
        The output rotation, with a zero offset

    See Also
    --------
    rot3d_y, rot3d_z : rotations about the other axes

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo3d import rot3d_x
    >>> np.round(rot3d_x(np.pi / 2) << np.array([0., 1., 0.]), 6)
    array([0., 0., 1.])
    """

    sa = glm.sin(angle)
    ca = glm.cos(angle)
    weight = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    return Aff3d(linear=weight)


def rot3d_y(angle: float):
    """Gets 3D rotation about the y-axis (pitch rotation).

    The rotation is right-handed: seen from the positive y-axis towards the origin, it is
    counter-clockwise for a positive angle, so it sends the z-axis towards the x-axis.

    Parameters
    ----------
    angle : float
        the angle to rotate, in radians

    Returns
    -------
    tfm : Aff3d
        The output rotation, with a zero offset

    See Also
    --------
    rot3d_x, rot3d_z : rotations about the other axes

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo3d import rot3d_y
    >>> np.round(rot3d_y(np.pi / 2) << np.array([0., 0., 1.]), 6)
    array([1., 0., 0.])
    """

    sa = glm.sin(angle)
    ca = glm.cos(angle)
    weight = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    return Aff3d(linear=weight)


def rot3d_z(angle: float):
    """Gets 3D rotation about the z-axis (yaw rotation).

    The rotation is right-handed: seen from the positive z-axis towards the origin, it is
    counter-clockwise for a positive angle, so it sends the x-axis towards the y-axis.

    Parameters
    ----------
    angle : float
        the angle to rotate, in radians

    Returns
    -------
    tfm : Aff3d
        The output rotation, with a zero offset

    See Also
    --------
    rot3d_x, rot3d_y : rotations about the other axes

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo3d import rot3d_z
    >>> np.round(rot3d_z(np.pi / 2) << np.array([1., 0., 0.]), 6)
    array([0., 1., 0.])
    """

    sa = glm.sin(angle)
    ca = glm.cos(angle)
    weight = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Aff3d(linear=weight)
