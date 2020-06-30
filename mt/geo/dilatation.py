import numpy as _np

from .affine_transformation import aff


__all__ = ['dlt', 'scale2d', 'flipx2d', 'flipy2d']


class dlt(aff):
    '''Dilatation (scaling and translation) in n-dim space. Here, scaling is per dimension, not uniform scaling.

    :Examples:
    >>> import numpy as _np
    >>> import mt.geo as _mg
    >>> a = _mg.dlt(offset=_np.array([1,2]), scale=_np.array([3,4]))
    >>> ~a
    dlt(offset=[-0.33333333 -0.5       ], scale=[0.33333333 0.25      ])
    >>> a*~a
    dlt(offset=[0. 0.], scale=[1. 1.])
    >>> a/a
    dlt(offset=[0. 0.], scale=[1. 1.])
    >>> a%a
    dlt(offset=[0. 0.], scale=[1. 1.])
    >>> b = _mg.dlt(offset=_np.array([1,0.5]), scale=_np.array([1/3,0.25]))
    >>> a*b
    dlt(offset=[4. 4.], scale=[1. 1.])
    >>> b*a
    dlt(offset=[1.33333333 1.        ], scale=[1. 1.])
    >>> a/b
    dlt(offset=[-8. -6.], scale=[ 9. 16.])
    >>> b/a
    dlt(offset=[0.88888889 0.375     ], scale=[0.11111111 0.0625    ])
    >>> a << _np.ones(2)
    array([4., 6.])
    '''

    # ----- base adaptation -----

    @property
    def bias(self):
        return self.__offset
    bias.__doc__ = aff.bias.__doc__

    @bias.setter
    def bias(self, bias):
        if len(bias.shape) != 1:
            raise ValueError("Bias is not a vector, shape {}.".format(bias.shape))
        self.__offset = bias

    @property
    def bias_dim(self):
        return self.__offset.shape[0]
    bias_dim.__doc__ = aff.bias_dim.__doc__

    @property
    def weight(self):
        return _np.diag(self.scale)
    weight.__doc__ = aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only.")

    @property
    def weight_shape(self):
        dim = self.scale_dim
        return (dim,dim)
    weight_shape.__doc__ = aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):
        '''The offset/bias part of the dilated isometry.'''
        return self.__offset

    @offset.setter
    def offset(self, offset):
        if len(offset.shape) != 1:
            raise ValueError("Offset is not a vector, shape {}.".format(offset.shape))
        self.__offset = offset

    @property
    def scale(self):
        '''The scale component/scalar of the dilated isometry.'''
        return self.__scale

    @scale.setter
    def scale(self, scale):
        if len(scale.shape) != 1:
            raise ValueError("Scale is not a vector, shape {}.".format(scale.shape))
        self.__scale = scale

    @property
    def scale_dim(self):
        return self.__scale.shape[0]

    # ----- methods -----

    def __init__(self, offset=_np.zeros(3), scale=_np.ones(3), check_shapes=True):
        self.offset = offset
        self.scale = scale
        if check_shapes:
            _ = self.dim # just to check the shapes

    def __repr__(self):
        return "dlt(offset={}, scale={})".format(self.offset, self.scale)

    def __lshift__(self, x):
        if x.shape != (self.dim,):
            raise ValueError("Shapes of input {} and offset {} do not match.".format(x.shape, self.offset.shape))
        return x*self.scale+self.offset
    __lshift__.__doc__ = aff.__lshift__.__doc__

    def __mul__(self, other):
        if not isinstance(other, dlt):
            return super(dlt, self).__mul__(other)
        return dlt(self << other.offset, self.scale*other.scale)
    __mul__.__doc__ = aff.__mul__.__doc__

    def __invert__(self):
        return dlt(-self.offset/self.scale, 1/self.scale)
    __invert__.__doc__ = aff.__invert__.__doc__


# ----- useful 2D transformations -----


from mt.base import logger # for warning

def scale2d(scale_x=1, scale_y=None):
    '''Returns the scaling.'''
    logger.warn_func_move('mt.geo.dilatation.scale2d', 'mt.geo.affine2d.scale2d')
    if scale_y is None:
        scale_y = scale_x
    return dlt(offset=_np.zeros(2), scale=_np.array([scale_x,scale_y]))

def flipx2d(width):
    '''Returns the transformation representing the horizontal flip of an image.'''
    logger.warn_func_move('mt.geo.dilatation.flipx2d', 'mt.geo.affine2d.flipLR2d')
    return dlt(offset=_np.array([width,0]), scale=_np.array([-1,1]))

def flipy2d(height):
    '''Returns the transformation representing the vertical flip of an image.'''
    logger.warn_func_move('mt.geo.dilatation.flipy2d', 'mt.geo.affine2d.flipUD2d')
    return dlt(offset=_np.array([0,height]), scale=_np.array([1,-1]))

