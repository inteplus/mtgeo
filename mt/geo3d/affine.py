"""Affine transformation in 3D."""

import math as m

from mt import np
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

    The 3D affine transformation defined here consists of a linear/weight part and an offset/bias part.

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    # ----- base adaptation -----

    @property
    def ndim(self):  # reimplementation to enforce constantness
        return 3

    def multiply(self, other):
        if not isinstance(other, [Aff3d, Aff]):
            return super(Aff3d, self).__mul__(other)
        return Aff3d(self.weight @ other.weight, self << other.bias)

    multiply.__doc__ = Aff.multiply.__doc__

    def invert(self):
        invWeight = _nl.inv(
            self.weight
        )  # slow, and assuming weight matrix is invertible
        return Aff3d(invWeight, invWeight @ (-self.bias))

    invert.__doc__ = Aff.invert.__doc__

    @property
    def bias_dim(self):
        """Returns the dimension of the bias vector. Raises ValueError if it is not 3."""
        if self.bias.shape[0] != 3:
            raise ValueError(
                "Expected bias dim to be 3, but seeing {}.".format(self.bias.shape[0])
            )

    # ----- methods -----

    def __repr__(self):
        return "Aff3d(weight_diagonal={}, bias={})".format(
            self.weight.diagonal(), self.bias
        )

    def as_glMatrix(self):
        """Returns a float32 16-vector that can be used as an OpenGL 4x4 columnar matrix."""
        a = np.zeros((4, 4), dtype=np.float32, order="F")
        a[:3, :3] = self.weight
        a[:3, 3] = self.bias
        return a


# ----- casting -----


_bc.register_cast(
    Aff3d, Aff, lambda x: Aff(weights=x.weight, bias=x.offset, check_shapes=False)
)
_bc.register_cast(
    Aff, Aff3d, lambda x: Aff3d(weight=x.weight, bias=x.bias, check_shape=False)
)
_bc.register_castable(Aff, Aff3d, lambda x: x.ndim == 3)


# ----- transform functions -----


def transform_Aff3d_on_Moments3d(aff_tfm, moments):
    """Transform a Moments3d using a 3D affine transformation.

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
    A = aff_tfm.weight
    old_m0 = moments.m0
    old_mean = moments.mean
    old_cov = moments.cov
    new_mean = A @ old_mean + aff_tfm.bias
    new_cov = A @ old_cov @ A.T
    new_m0 = old_m0 * abs(aff_tfm.det)
    new_m1 = new_m0 * new_mean
    new_m2 = new_m0 * (np.outer(new_mean, new_mean) + new_cov)
    return Moments3d(new_m0, new_m1, new_m2)


register_transform(Aff3d, Moments3d, transform_Aff3d_on_Moments3d)


def transform_Aff3d_on_ndarray(aff_tfm, point_array):
    """Transform an array of 3D points using a 3D affine transformation.

    Parameters
    ----------
    aff_tfm : Aff3d or Aff
        a 3D affine transformation
    point_array : numpy.ndarray with last dimension having the same length as the dimensionality of the transformation
        an array of 3D points

    Returns
    -------
    numpy.ndarray
        affine-transformed point array
    """
    return point_array @ aff_tfm.weight.T + aff_tfm.bias


register_transform(Aff3d, np.ndarray, transform_Aff3d_on_ndarray)
register_transformable(Aff3d, np.ndarray, lambda x, y: y.shape[-1] == 3)


def transform_Aff3d_on_PointList3d(aff_tfm, point_list):
    """Transform a 3D point list using a 3D affine transformation.

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

    Parameters
    ----------
    angle : float
        the angle to rotate

    Returns
    -------
    tfm : Sim2d
        The output rotation
    """

    sa = m.sin(angle)
    ca = m.cos(angle)
    weight = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    return Aff3d(weight=weight, bias=np.zeros(3), check_shapes=False)


def rot3d_y(angle: float):
    """Gets 3D rotation about the y-axis (pitch rotation).

    Parameters
    ----------
    angle : float
        the angle to rotate

    Returns
    -------
    tfm : Sim2d
        The output rotation
    """

    sa = m.sin(angle)
    ca = m.cos(angle)
    weight = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    return Aff3d(weight=weight, bias=np.zeros(3), check_shapes=False)


def rot3d_z(angle: float):
    """Gets 3D rotation about the z-axis (yaw rotation).

    Parameters
    ----------
    angle : float
        the angle to rotate

    Returns
    -------
    tfm : Sim2d
        The output rotation
    """

    sa = m.sin(angle)
    ca = m.cos(angle)
    weight = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Aff3d(weight=weight, bias=np.zeros(3), check_shapes=False)
