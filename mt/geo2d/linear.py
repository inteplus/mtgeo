"""Module supporting the 2x2 linear transformation.

A 2x2 matrix can be parametrised by an "sshr" tuple `(sx, sy, h, r)`: scaling along x and y,
shearing and rotation angle (in radian). This module converts between the two representations (for
single :class:`glm.mat2` matrices and for batches of tensorflow tensors) and builds batches of 3x3
affine matrices.
"""

from mt import glm

__all__ = [
    "mat2sshr",
    "mat2sshr_tf",
    "sshr2mat",
    "sshr2mat_tf",
    "make_affine",
]


# ----- sshr representation of a 2x2 matrix -----


# Redundant vector (sx,sy,h,r,cr,sr).
# Matrix 2x2 parametrised as [a0,a1,a2,a3].


def sshr2mat(sshr: glm.vec4) -> glm.mat2:
    """Converts an sshr tuple into a mat2 transformation matrix.

    An sshr tuple (sx,sy,h,r) represents the scale2d(sx,sy) shear2d(h) rotate2d(r) transformation,
    where `r` is the rotation angle in radian.

    Formula::

        cr = cos(r)
        sr = sin(r)
        a0 = sx*(h*sr + cr)
        a1 = sx*(h*cr - sr)
        a2 = sy*sr
        a3 = sy*cr
        m  = [[a0, a1], [a2, a3]]  # row-major

    Parameters
    ----------
    sshr : glm.vec4
        the input sshr tuple `(sx, sy, h, r)`

    Returns
    -------
    glm.mat2
        the output 2d transformation matrix (glm matrices are column-major)

    Examples
    --------
    >>> import numpy as np
    >>> from mt import glm
    >>> from mt.geo2d import sshr2mat
    >>> m = sshr2mat(glm.vec4(2, 3, 0.5, 0))  # scales (2,3), shear 0.5, no rotation
    >>> np.array(m.to_list()).T  # as a row-major array
    array([[2., 1.],
           [0., 3.]])

    See Also
    --------
    mat2sshr : the inverse conversion

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    sx, sy, h, r = sshr

    cr = glm.cos(r)
    sr = glm.sin(r)
    a0 = sx * (h * sr + cr)
    a1 = sx * (h * cr - sr)
    a2 = sy * sr
    a3 = sy * cr

    return glm.mat2(glm.vec2(a0, a2), glm.vec2(a1, a3))


def sshr2mat_tf(tensor):
    """Converts an array of sshr tuples into an array of 2x2 transformation matrices.

    A tensorflow version of :func:`sshr2mat`, working on a batch.

    An sshr tuple (sx,sy,h,r) represents the scale2d(sx,sy) shear2d(h) rotate2d(r) transformation,
    where `r` is the rotation angle in radian.

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
        a tensor of shape (batch, 4) containing sshr tuples `(sx, sy, h, r)`

    Returns
    -------
    tensorflow.Tensor
        a tensor of shape (batch, 2, 2) containing 2d row-major transformation matrices

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    from mt import tf

    sx = tensor[:, 0]
    sy = tensor[:, 1]
    h = tensor[:, 2]
    r = tensor[:, 3]

    cr = tf.math.cos(r)
    sr = tf.math.sin(r)
    a0 = sx * (h * sr + cr)
    a1 = sx * (h * cr - sr)
    a2 = sy * sr
    a3 = sy * cr

    m0 = tf.stack([a0, a1], axis=1)
    m1 = tf.stack([a2, a3], axis=1)
    m = tf.stack([m0, m1], axis=1)

    return m


def mat2sshr(m: glm.mat2) -> glm.vec4:
    """Converts a 2d transformation matrix into an sshr tuple.

    An sshr tuple (sx,sy,h,r) represents the scale2d(sx,sy) shear2d(h) rotate2d(r) transformation,
    where `r` is the rotation angle in radian. The formulas below are the generic case. The code
    handles the degenerate cases `sy == 0` (assumes `r = 0`) and `sx == 0` (assumes `h = 0`)
    separately, and computes `h` from whichever of `a0`, `a1` is better conditioned.

    Formula::

        [[a0, a1], [a2, a3]] = m  # row-major
        sy = hypot(a2,a3)
        r = atan2(a2,a3)
        sx = det(A)/sy
        h = (a0*a2 + a1*a3)/(sx*r)


    Parameters
    ----------
    m : glm.mat2
        the input 2d transformation matrix (glm matrices are column-major)

    Returns
    -------
    sshr : glm.vec4
        the output sshr tuple `(sx, sy, h, r)`

    Examples
    --------
    >>> from mt import glm
    >>> from mt.geo2d import sshr2mat, mat2sshr
    >>> [round(v, 4) for v in mat2sshr(sshr2mat(glm.vec4(2, 3, 0.5, 0.3)))]
    [2.0, 3.0, 0.5, 0.3]

    See Also
    --------
    sshr2mat : the inverse conversion

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    a0 = m[0, 0]
    a1 = m[1, 0]
    a2 = m[0, 1]
    a3 = m[1, 1]

    sy = glm.length(glm.vec2(a2, a3))
    if sy == 0:  # degenerate case, assume r = 0
        r = 0
        sx = a0
        h = 0 if a0 == 0 else a1 / a0
    else:
        r = glm.atan2(a2, a3)
        sx = glm.determinant(m) / sy
        if sx == 0:  # degenerate case, assume h = 0
            h = 0
        elif glm.abs(a2) < glm.abs(a3):  # use a1
            h = (a1 * sy / sx + a2) / a3
        else:  # use a0
            h = (a0 * sy / sx - a3) / a2

    return glm.vec4(sx, sy, h, r)


def mat2sshr_tf(tensor):
    """Converts an array of 2x2 transformation matrices into an array of sshr tuples.

    A tensorflow version of :func:`mat2sshr`, working on a batch. Unlike :func:`mat2sshr`, it only
    implements the generic formulas below, with `tf.math.divide_no_nan` guarding the divisions.

    An sshr tuple (sx,sy,h,r) represents the scale2d(sx,sy) shear2d(h) rotate2d(r) transformation,
    where `r` is the rotation angle in radian.

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
        a tensor of shape (batch, 4) containing sshr tuples `(sx, sy, h, r)`

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    from mt import tf

    a0 = tensor[:, 0, 0]
    a1 = tensor[:, 0, 1]
    a2 = tensor[:, 1, 0]
    a3 = tensor[:, 1, 1]

    sy = tf.experimental.numpy.hypot(a2, a3)
    r = tf.math.atan2(a2, a3)
    sx = tf.math.divide_no_nan(a0 * a3 - a1 * a2, sy)
    h = tf.math.divide_no_nan(a0 * a2 + a1 * a3, sx * r)

    return tf.stack([sx, sy, h, r], axis=1)


def make_affine(mat2d_tensor, ofs2d_tensor):
    """Makes an array of 2d affine transformation matrices (in 3x3 with `[0,0,1]` as the 3rd row).

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
    """

    from mt import tf

    zero_tensor = tf.zeros_like(mat2d_tensor[:, :1, :])
    col12 = tf.concat([mat2d_tensor, zero_tensor], axis=1)
    one_tensor = tf.ones_like(ofs2d_tensor[:, :1])
    col3 = tf.concat([ofs2d_tensor, one_tensor], axis=1)
    tensor = tf.concat([col12, tf.expand_dims(col3, axis=-1)], axis=2)
    return tensor
