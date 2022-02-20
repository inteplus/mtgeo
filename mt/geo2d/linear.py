'''Module supporting the Lin2d class.'''

from mt import np

from ..geo import register_transform, register_transformable
from .linear_impl import Lin2d
from .moments import Moments2d
from .point_list import PointList2d
from .polygon import Polygon


__all__ = ['Lin2d', 'mat2sshr', 'sshr2mat', 'make_affine', 'transform_Lin2d_on_Moments2d', 'transform_Lin2d_on_PointList2d', 'transform_Lin2d_on_Polygon']


# ----- sshr representation of a 2x2 matrix -----


# Redundant vector (sx,sy,h,r,cr,sr).
# Matrix 2x2 parametrised as [a0,a1,a2,a3].


def sshr2mat(tensor):
    '''Converts an array of sshr tuples into an array of 2x2 transformation matrices.

    An sshr tuple (sx,sy,h,r) represents the scale2d(sx,sy) shear2d(h) rotate2d(r) transformation.

    Formula::

        cr = cos(r)
        sr = sin(r)
        a0 = sx*(h*sr + cr)
        a1 = sx*(h*cr - sr)
        a2 = sy*sr
        a3 = sy*cr
        m  = [[a0, a1], [a2, a3]]

    Parameters
    ----------
    tensor : tensorflow.Tensor
        a tensor of shape (batch, 4) containing sshr tuples

    Returns
    -------
    tensorflow.Tensor
        a tensor of shape (batch, 2, 2) containing 2d row-major transformation matrices

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    from mt import tf

    sx = tensor[:,0]
    sy = tensor[:,1]
    h = tensor[:,2]
    r = tensor[:,3]

    cr = tf.math.cos(r)
    sr = tf.math.sin(r)
    a0 = sx*(h*sr + cr)
    a1 = sx*(h*cr - sr)
    a2 = sy*sr
    a3 = sy*cr

    m0 = tf.stack([a0, a1], axis=1)
    m1 = tf.stack([a2, a3], axis=1)
    m = tf.stack([m0, m1], axis=1)

    return m


def mat2sshr(tensor):
    '''Converts an array of 2x2 transformation matrices into an array of sshr tuples.

    An sshr tuple (sx,sy,h,r) represents the scale2d(sx,sy) shear2d(h) rotate2d(r) transformation.

    Formula::

        [[a0, a1], [a2, a3]] = m
        sy = hypot(a2,a3)
        r = atan2(a2,a3)
        sx = det(A)/sy
        h = (a0*a2 + a1*a3)/(sx*r)


    Parameters
    ----------
    tensor : tensorflow.Tensor
        a tensor of shape (batch, 2, 2) containing 2d row-major transformation matrices

    Returns
    -------
    tensorflow.Tensor
        a tensor of shape (batch, 4) containing sshr tuples

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    '''

    from mt import tf

    a0 = tensor[:,0,0]
    a1 = tensor[:,0,1]
    a2 = tensor[:,1,0]
    a3 = tensor[:,1,1]

    sy = tf.experimental.numpy.hypot(a2, a3)
    r = tf.math.atan2(a2, a3)
    sx = tf.math.divide_no_nan(a0*a3 - a1*a2, sy)
    h = tf.math.divide_no_nan(a0*a2 + a1*a3, sx*r)

    return tf.stack([sx, sy, h, r], axis=1)


def make_affine(mat2d_tensor, ofs2d_tensor):
    '''Makes an array of 2d affine transformation matrices (in 3x3 with `[0,0,1]` as the 3rd row).

    Parameters
    ----------
    mat2d_tensor : tensorflow.Tensor
        a tensor of shape (batch, 2, 2) containing 2d row-major transformation matrices
    ofs2d_tensor : tensorflow.Tensor
        a tensor of shape (batch, 2) containing 2d translations

    Returns
    -------
    tensorflow.Tensor
        a tensor of shape (batch, 3, 3) containing 2d affine transformation matrices with the
        linear and translation parts defined in the input tensors
    '''

    from mt import tf
    tensor = tf.concat([mat2d_tensor, tf.expand_dims(ofs2d_tensor, axis=-1)], axis=2)
    tensor = tf.concat([tensor, tf.broadcast_to(tf.constant([0, 0, 1], dtype=tensor.dtype), [mat2d_tensor.shape[0], 1, 3])], axis=1)
    return tensor


# ----- transform functions -----


def transform_Lin2d_on_Moments2d(lin_tfm, moments):
    '''Transform a Moments2d using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Lin2d
        2D linear transformation
    moments : Moments2d
        2D moments

    Returns
    -------
    Moments2d
        linear-transformed 2D moments
    '''
    A = lin_tfm.matrix
    old_m0 = moments.m0
    old_mean = moments.mean
    old_cov = moments.cov
    new_mean = A @ old_mean
    new_cov = A @ old_cov @ A.T
    new_m0 = old_m0*abs(lin_tfm.det)
    new_m1 = new_m0*new_mean
    new_m2 = new_m0*(np.outer(new_mean, new_mean) + new_cov)
    return Moments2d(new_m0, new_m1, new_m2)
register_transform(Lin2d, Moments2d, transform_Lin2d_on_Moments2d)


def transform_Lin2d_on_ndarray(lin_tfm, point_array):
    '''Transform an array of 2D points using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Aff
        a 2D linear transformation
    point_array : numpy.ndarray with last dimension having the same length as the dimensionality of the transformation
        an array of 2D points

    Returns
    -------
    numpy.ndarray
        linear-transformed point array
    '''
    return point_array @ lin_tfm.matrix.T
register_transform(Lin2d, np.ndarray, transform_Lin2d_on_ndarray)
register_transformable(Lin2d, np.ndarray, lambda x, y: y.shape[-1] == 2)


def transform_Lin2d_on_PointList2d(lin_tfm, point_list):
    '''Transform a 2D point list using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Lin2d
        a 2D linear transformation
    point_list : PointList2d
        a 2D point list

    Returns
    -------
    PointList2d
        linear-transformed point list
    '''
    return PointList2d(point_list.points @ lin_tfm.matrix.T, check=False)
register_transform(Lin2d, PointList2d, transform_Lin2d_on_PointList2d)


def transform_Lin2d_on_Polygon(lin_tfm, poly):
    '''Transform a polygon using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Lin2d
        a 2D linear transformation
    poly : Polygon
        a 2D polygon

    Returns
    -------
    Polygon
        linear-transformed polygon
    '''
    return Polygon(poly.points @ lin_tfm.matrix.T, check=False)
register_transform(Lin2d, Polygon, transform_Lin2d_on_Polygon)


