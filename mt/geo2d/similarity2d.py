import math as m

from mt import np
import mt.base.casting as _bc

from ..geond.dilated_isometry import Dliso
from .affine import Aff2d


__all__ = [
    "Sim2d",
    "cv2cg",
]


class Sim2d(Dliso):
    """Similarity (angle-preserving) transformation in 2D.

    Consider the following family of 2D transformations:
    y = Sim2d(offset, scale, angle, on)(x) = Translate(offset)*UniformScale(scale)*Rotate(angle)*ReflectX(on)(x),
    where 'x' is a vector of coordinates in 2D, 'on' is a boolean, ReflectX is the reflection
    through the X axis (the axis of the first dimension), 'angle' is an angle in radian, Rotate is
    the 2D rotation, 'scale' is a positive scalar, UniformScale is uniform scaling, 'offset' is a
    vector of 2D coordinates and Translate is the translation.

    This family forms a Lie group of 2 disconnected components, those without reflection and those
    with reflection. The Lie group multiplication and inverse operators are derived as below:
      - multiplication: Sim2d(offset_x, scale_x, angle_x, on_x)*Sim2d(offset_y, scale_y, angle_y, on_y) = Sim2d(offset_x + UniformScale(scale_x)*Rotate(angle_x)*ReflectX(on_x)(offset_y), scale_x*scale_y, angle_x + (-1)^{on_x} angle_y, on_x xor on_y)
      - inverse: ~Sim2d(offset, scale, angle, on) = Sim2d(-linear(scale, angle, on) offset, 1/scale, -(-1)^{on} angle, on)

    Parameters
    ----------
    offset : numpy.ndarray
        the translation vector of length 2. Default is the zero vector.
    scale : float
        the positive uniform scaling factor. Default is 1.
    angle : float
        the rotation angle in radian, from the x-axis towards the y-axis. Default is 0.
    on : bool
        whether the reflection through the x-axis (applied first, before the rotation) is active.
        Default is False.

    Attributes
    ----------
    offset : numpy.ndarray
        the translation vector
    scale : float
        the scaling factor
    angle : float
        the rotation angle in radian
    on : bool
        whether the transformation includes the reflection
    linear : numpy.ndarray
        the 2x2 linear part `scale*Rotate(angle)*ReflectX(on)`
    unitary : numpy.ndarray
        the 2x2 linear part without the scale (read-only)

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import Sim2d
    >>> s = Sim2d(offset=np.array([1., 2.]), scale=2, angle=np.pi / 2)
    >>> transform(s, np.array([[1., 0.]])).round(6)  # scale 2, rotate 90 degrees, then translate
    array([[1., 4.]])
    >>> t = ~s * s  # the identity, up to rounding
    >>> t.offset.round(6) + 0.0, round(t.scale, 6), round(t.angle, 6)
    (array([0., 0.]), 1.0, 0.0)

    See Also
    --------
    mt.geo2d.Aff2d : general affine transformation, to which a Sim2d can be cast
    cv2cg : the reflection between OpenCV and OpenGL image coordinates

    References:
        [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky
    MT is trying to advertise his paper!)
    """

    # ----- static methods -----

    @staticmethod
    def get_linear(angle, on, scale=1.0):
        """Forms the linear part of the transformation matrix representing
        scaling*rotation*reflection.

        Parameters
        ----------
        angle : float
            the rotation angle in radian
        on : bool
            whether to include the reflection through the x-axis
        scale : float
            the uniform scaling factor

        Returns
        -------
        numpy.ndarray
            the 2x2 matrix `scale*Rotate(angle)*ReflectX(on)`, which is
            `[[c, -s], [s, c]]*scale` without reflection and `[[-c, -s], [-s, c]]*scale` with it,
            where `c = cos(angle)` and `s = sin(angle)`

        Examples
        --------
        >>> from mt.geo2d import Sim2d
        >>> Sim2d.get_linear(0, True, scale=2) + 0.0
        array([[-2.,  0.],
               [ 0.,  2.]])
        """
        ca = np.cos(angle) * scale
        sa = np.sin(angle) * scale
        return (
            np.array([[-ca, -sa], [-sa, ca]]) if on else np.array([[ca, -sa], [sa, ca]])
        )

    # ----- base adaptation -----

    @property
    def bias_dim(self):
        return 2

    bias_dim.__doc__ = Dliso.bias_dim.__doc__

    @property
    def unitary(self):
        """numpy.ndarray: the 2x2 unitary part (rotation and optional reflection) of the linear
        matrix. It is read-only: setting it raises TypeError."""
        return Sim2d.get_linear(self.angle, self.on)

    @unitary.setter
    def unitary(self, unitary):
        raise TypeError("Unitary matrix is read-only.")

    @property
    def weight_shape(self):
        """tuple: the shape of the weight matrix, always (2, 2)."""
        return (2, 2)

    # ----- data encapsulation -----

    @property
    def angle(self):
        """float: the rotation angle in radian."""
        return self.__angle

    @angle.setter
    def angle(self, angle):
        self.__angle = angle

    @property
    def on(self):
        """bool: whether the transformation includes the reflection through the x-axis."""
        return self.__on

    @on.setter
    def on(self, on):
        self.__on = on

    # ----- derived properties -----

    @property
    def linear(self):
        return Sim2d.get_linear(self.angle, self.on, self.scale)

    linear.__doc__ = Dliso.linear.__doc__

    # ----- methods -----

    def __init__(self, offset=np.zeros(2), scale=1, angle=0, on=False):
        """Initialises a 2D similarity transformation. See the class docstring."""
        self.offset = offset
        self.scale = scale
        self.angle = angle
        self.on = on

    def __repr__(self):
        return f"Sim2d(offset={self.offset}, scale={self.scale}, angle={self.angle}, on={self.on})"

    # ----- base adaptation -----

    def multiply(self, other):
        if not isinstance(other, Sim2d):
            return super(Sim2d, self).__mul__(other)
        return Sim2d(
            self << other.offset,
            self.scale * other.scale,
            self.angle - other.angle if self.on else self.angle + other.angle,
            not self.on != (not other.on),  # fastest xor
        )

    multiply.__doc__ = Dliso.multiply.__doc__

    def invert(self):
        invScale = 1 / self.scale
        invAngle = self.angle if self.on else -self.angle
        mat = Sim2d.get_linear(invAngle, self.on, invScale)
        return Sim2d(np.dot(mat, -self.offset), invScale, invAngle, self.on)

    invert.__doc__ = Dliso.invert.__doc__


# ----- casting -----


_bc.register_cast(
    Sim2d, Dliso, lambda x: Dliso(offset=x.offset, scale=x.scale, unitary=x.unitary)
)
_bc.register_cast(Sim2d, Aff2d, lambda x: Aff2d(offset=x.offset, linear=x.linear))


# ---- utilities -----


def cv2cg(height: int):
    """Gets the transformation to convert image coordinates between OpenCV and OpenGL conventions.

    OpenCV has y-axis pointing downward. OpenGL has y-axis pointing upward.

    Since the transformation and its inverse are the same, you can use the same transformation for
    both ways.

    Parameters
    ----------
    height : int
        the number of rows of the image coordinate system

    Returns
    -------
    tfm : Sim2d
        The output transformation

    Notes
    -----
    This function is essentially the same as :func:`mt.geo2d.flipUD2d`, just different return
    values.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import cv2cg
    >>> transform(cv2cg(480), np.array([[10., 0.], [10., 480.]])).round(6) + 0.0
    array([[ 10., 480.],
           [ 10.,   0.]])
    """

    return Sim2d(offset=np.array([0.0, height]), angle=m.pi, on=True)
