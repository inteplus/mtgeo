import numpy as _np
import math as _m

from .dilated_isometry import dliso


__all__ = ['sim2']


class sim2(dliso):
    '''Similarity (angle-preserving) transformation in 2D.

    Consider the following family of 2D transformations: y = sim2(offset, scale, angle, on)(x) = Translate(offset)*UniformScale(scale)*Rotate(angle)*ReflectX(on)(x), where 'x' is a vector of coordinates in 2D, 'on' is a boolean, ReflectX is the reflection through the X axis (the axis of the first dimension), 'angle' is an angle in radian, Rotate is the 2D rotation, 'scale' is a positive scalar, UniformScale is uniform scaling, 'offset' is a vector of 2D coordinates and Translate is the translation.

    This family forms a Lie group of 2 disconnected components, those without reflection and those with reflection. The Lie group multiplication and inverse operators are derived as below:
      - multiplication: sim2(offset_x, scale_x, angle_x, on_x)*sim2(offset_y, scale_y, angle_y, on_y) = sim2(offset_x + UniformScale(scale_x)*Rotate(angle_x)*ReflectX(on_x)(offset_y), scale_x*scale_y, angle_x + (-1)^{on_x} angle_y, on_x xor on_y)
      - inverse: ~sim2(offset, scale, angle, on) = sim2(-linear(scale, angle, on) offset, 1/scale, -(-1)^{on} angle, on)

    References:
        [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
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
        return (2,2)

    # ----- data encapsulation -----

    @property
    def angle(self):
        return self.__angle

    @angle.setter
    def angle(self, angle):
        self.__angle = angle

    @property
    def on(self):
        return self.__on

    @on.setter
    def on(self, on):
        self.__on = on

    # ----- derived properties -----

    @property
    def linear(self):
        return sim2.get_linear(self.angle, self.on, self.scale)
    linear.__doc__ = dliso.linear.__doc__

    # ----- methods -----

    def __init__(self, offset=_np.zeros(2), scale=1, angle=0, on=False):
        self.offset = offset
        self.scale = scale
        self.angle = angle
        self.on = on

    def __repr__(self):
        return "sim2(offset={}, scale={}, angle={}, on={})".format(self.offset, self.scale, self.angle, self.on)

    def __mul__(self, other):
        if not isinstance(other, sim2):
            return super(sim2, self).__mul__(other)
        return sim2(self << other.offset,
            self.scale*other.scale,
            self.angle-other.angle if self.on else self.angle+other.angle,
            not self.on != (not other.on) # fastest xor
            )
    __mul__.__doc__ = dliso.__mul__.__doc__

    def __invert__(self):
        invScale = 1/self.scale
        invAngle = self.angle if self.on else -self.angle
        mat = sim2.get_linear(invAngle, self.on, invScale)
        return sim2(_np.dot(mat, -self.offset), invScale, invAngle, self.on)
    __invert__.__doc__ = dliso.__invert__.__doc__
