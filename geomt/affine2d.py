import numpy as _np
import math as _m

from .affine_transformation import aff


class aff2(aff):
    '''Affine transformation in 2D.

    We follow skimage's parametrization of 2D affine transformations `skimage.transform.AffineTransformation <https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.AffineTransform>`_. It has the following form::

        X = a0*x + a1*y + a2
        X = sx*x*cos(r) - sy*y*sin(r + h) + a2
        Y = b0*x + b1*y + b2
        Y = sx*x*sin(r) + sy*y*cos(r + h) + b2
        mat = [[a0, a1, a2], [b0, b1, b2], [0, 0, 1]]

    where `s=(sx,sy)` are the scaling parameters, `r` is the rotation angle (in radian), `h` is the shearing angle (in radian), `t=(tx,ty)` are the translation parameters. To make the parametrization unique, we assume `sx > 0` and `sy > 0`. We then obtain the inverse form::

        sx = hypot(a0, a3)
        sy = hypot(a1, a4)
        tx = a2
        ty = a5
        r = atan2(a3, a0)
        h = atan2(a1, a4) - atan2(a3, a0)

    We need to figure out from_matrix() and to_matrix() functions.

    And then we need to figure out multiplication and inverse operation.

    MT-TODO: continue from here (2019/10/09).

    Consider the following family of 2D transformations: y = sim2(offset, scale, angle, on)(x) = Translate(offset)*UniformScale(scale)*Rotate(angle)*ReflectX(on)(x), where 'x' is a vector of coordinates in 2D, 'on' is a boolean, ReflectX is the reflection through the X axis (the axis of the first dimension), 'angle' is an angle in radian, Rotate is the 2D rotation, 'scale' is a positive scalar, UniformScale is uniform scaling, 'offset' is a vector of 2D coordinates and Translate is the translation.

    This family forms a Lie group of 2 disconnected components, those without reflection and those with reflection. The Lie group multiplication and inverse operators are derived as below:
      - multiplication: sim2(offset_x, scale_x, angle_x, on_x)*sim2(offset_y, scale_y, angle_y, on_y) = sim2(offset_x + UniformScale(scale_x)*Rotate(angle_x)*ReflectX(on_x)(offset_y), scale_x*scale_y, angle_x + (-1)^{on_x} angle_y, on_x xor on_y)
      - inverse: ~sim2(offset, scale, angle, on) = sim2(-linear(scale, angle, on) offset, 1/scale, -(-1)^{on} angle, on)

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    # ----- static methods -----

    @staticmethod
    def get_linear(angle, on, scale=1.0):
        '''Forms the linear part of the transformation matrix representing scaling*rotation*reflection.'''
        ca = _m.cos(angle)*scale
        sa = _m.sin(angle)*scale
        return _np.array([[-ca, -sa], [-sa, ca]]) if on else _np.array([[ca, -sa], [sa, ca]])

    # ----- base adaptation -----

    @property
    def bias_dim(self):
        return 2
    bias_dim.__doc__ = dliso.bias_dim.__doc__

    @property
    def unitary(self):
        return sim2.get_linear(self.angle, self.on)

    @unitary.setter
    def unitary(self, unitary):
        raise TypeError("Unitary matrix is read-only.")

    @property
    def weight_shape(self):
        return (2, 2)

    # ----- data encapsulation -----

    @property
    def offset(self):  # done
        return self.__offset

    @offset.setter
    def offset(self, offset):  # done
        if len(offset.shape) != 1 or offset.shape[0] != 2:
            raise ValueError(
                "Offset is not a 2D vector, shape {}.".format(offset.shape))
        self.__offset = offset

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):  # done
        if len(scale.shape) != 1 or scale.shape[0] != 2:
            raise ValueError(
                "Scale is not a 2D vector, shape {}.".format(offset.shape))
        self.__scale = scale

    @property
    def angle(self):  # done
        return self.__angle

    @angle.setter
    def angle(self, angle):  # done
        self.__angle = angle

    @property
    def shear(self):  # done
        return self.__shear

    @shear.setter
    def shear(self, shear):  # done
        self.__shear = shear

    # ----- derived properties -----

    @property
    def linear(self):  # done
        '''Returns the linear part of the affine transformation matrix.'''
        return _np.array((
            (self.scale[0]*_np.cos(self.angle), -
             self.scale[1]*_np.sin(self.angle + self.shear)),
            (self.scale[0]*_np.sin(self.angle), self.scale[1]*_np.cos(self.angle + self.shear))))

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), scale=_np.ones(2), angle=0, shear=0):  # done
        self.offset = offset
        self.scale = scale
        self.angle = angle
        self.shear = shear

    def __repr__(self):  # done
        return "aff2(offset={}, scale={}, angle={}, shear={})".format(self.offset, self.scale, self.angle, self.shear)

    def __mul__(self, other):
        if not isinstance(other, sim2):
            return super(sim2, self).__mul__(other)
        return sim2(self << other.offset,
                    self.scale*other.scale,
                    self.angle-other.angle if self.on else self.angle+other.angle,
                    not self.on != (not other.on)  # fastest xor
                    )
    __mul__.__doc__ = dliso.__mul__.__doc__

    def __invert__(self):
        invScale = 1/self.scale
        invAngle = self.angle if self.on else -self.angle
        mat = sim2.get_linear(invAngle, self.on, invScale)
        return sim2(_np.dot(mat, -self.offset), invScale, invAngle, self.on)
    __invert__.__doc__ = dliso.__invert__.__doc__
