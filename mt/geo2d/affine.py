from mt import tp, np, glm
import mt.base.casting as _bc

from ..geo import TwoD, register_transform, register_transformable
from ..geond import Aff
from .moments import Moments2d
from .point_list import PointList2d
from .polygon import Polygon
from .rect import Rect
from .linear import sshr2mat, mat2sshr


__all__ = [
    "Aff2d",
    "transform_Aff2d_on_Moments2d",
    "transform_Aff2d_on_PointList2d",
    "transform_Aff2d_on_ndarray",
    "transform_Aff2d_on_Polygon",
    "swapAxes2d",
    "flipLR2d",
    "flipUD2d",
    "shearX2d",
    "shearY2d",
    "originate2d",
    "rotate2d",
    "translate2d",
    "scale2d",
    "crop2d",
    "crop_rect",
    "uncrop_rect",
    "rect2rect",
    "rect2rect_tf",
]


class Aff2d(TwoD, Aff):
    """Affine transformation in 2D.

    The 2D affine transformation defined here consists of a linear/weight part and an offset/bias
    part. It maps a point `x` (an `(x, y)` pair) to `linear * x + offset`. Multiplying two
    transformations composes them: `(a * b)` applies `b` first and then `a`. The inverse is
    obtained with `~a` (or `a.invert()`). Use :func:`mt.geo.transform` to apply a transformation to
    points, :class:`~mt.geo2d.PointList2d`, :class:`~mt.geo2d.Polygon`, :class:`~mt.geo2d.Ellipse`
    and so on.

    Parameters
    ----------
    offset : numpy.ndarray or glm.vec2
        the translation (bias) vector of length 2. Default is the zero vector.
    linear : numpy.ndarray or glm.mat2 or glm.vec4
        the 2x2 linear (weight) part. A numpy array is interpreted as a normal row-major matrix.
        A `glm.mat2` is column-major. A `glm.vec4` is interpreted as an sshr tuple `(sx, sy, h, r)`
        (scale, shear, rotation angle in radian), see :func:`~mt.geo2d.sshr2mat`. Default is the
        identity.

    Attributes
    ----------
    offset : glm.vec2
        the bias vector
    bias : numpy.ndarray
        the read-only numpy view of `offset`
    linear : glm.mat2
        the weight matrix
    weight : numpy.ndarray
        the read-only numpy view of `linear`, as a row-major 2x2 array
    matrix : numpy.ndarray
        the 3x3 homogeneous matrix `[[weight, bias], [0, 0, 1]]`
    det : float
        the determinant of the linear part

    Examples
    --------
    >>> from mt import np
    >>> from mt.geo import transform
    >>> from mt.geo2d import Aff2d, translate2d, scale2d
    >>> t = translate2d(1, 2) * scale2d(2)  # scale first, then translate
    >>> t
    Aff2d(offset=vec2( 1, 2 ), linear=mat2x2(( 2, 0 ), ( 0, 2 )))
    >>> transform(t, np.array([[1., 1.], [3., 0.]]))
    array([[3., 4.],
           [7., 2.]])
    >>> (~t * t).matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> Aff2d(offset=np.array([5., 6.])).matrix
    array([[1., 0., 5.],
           [0., 1., 6.],
           [0., 0., 1.]])

    See Also
    --------
    mt.geond.Aff
        the base class, for any dimension
    translate2d, scale2d, rotate2d
        factory functions for common transformations

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not really, cheeky MT is trying to advertise his paper!)
    """

    # ----- static methods -----

    @staticmethod
    def from_matrix(mat: np.ndarray):
        """Obtains an Aff2d instance from a non-singular affine transformation matrix.

        Parameters
        ----------
        mat : numpy.ndarray
            a non-singular affine transformation matrix, of shape 3x3 (homogeneous) or 2x3. Only the
            first two rows are used: `mat[:2, :2]` is the linear part and `mat[:2, 2]` the offset.

        Returns
        -------
        Aff2d
            An instance representing the transformation

        Notes
        -----
        For speed reasons, no checking is involved.

        Examples
        --------
        >>> import numpy as np
        >>> from mt.geo2d import Aff2d
        >>> Aff2d.from_matrix(np.array([[2., 0., 1.], [0., 3., 2.], [0., 0., 1.]]))
        Aff2d(offset=vec2( 1, 2 ), linear=mat2x2(( 2, 0 ), ( 0, 3 )))
        """
        return Aff2d(offset=mat[:2, 2], linear=mat[:2, :2])

    # ----- base adaptation -----

    @property
    def ndim(self):  # reimplementation to enforce constantness
        """The dimensionality of the transformation, always 2."""
        return 2

    def multiply(self, other):
        if not isinstance(other, Aff2d):
            return super(Aff2d, self).multiply(other)
        return Aff2d(
            offset=self.linear * other.offset + self.offset,
            linear=self.linear * other.linear,
        )

    multiply.__doc__ = Aff.multiply.__doc__

    def invert(self):
        invLinear = glm.inverse(self.linear)
        invOffset = invLinear * (-self.offset)
        return Aff2d(offset=invOffset, linear=invLinear)

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
        return 2

    bias_dim.__doc__ = Aff.bias_dim.__doc__

    @property
    def weight(self):
        return np.frombuffer(self.__linear.to_bytes(), dtype=np.float32).reshape(2, 2).T

    weight.__doc__ = Aff.weight.__doc__

    @weight.setter
    def weight(self, weight):
        raise TypeError("Weight matrix is read-only. Use self.linear instead.")

    @property
    def weight_shape(self):
        return (2, 2)

    weight_shape.__doc__ = Aff.weight_shape.__doc__

    # ----- data encapsulation -----

    @property
    def offset(self):
        """glm.vec2: the translation (bias) vector. Assigning converts the value to a vec2."""
        return self.__offset

    @offset.setter
    def offset(self, value: tp.Union[np.ndarray, glm.vec2]):
        self.__offset = glm.vec2(value)

    @property
    def linear(self):
        """glm.mat2: the linear (weight) part. Assigning accepts a numpy array (row-major 2x2), a
        `glm.mat2` or a `glm.vec4` sshr tuple `(sx, sy, h, r)`."""
        return self.__linear

    @linear.setter
    def linear(self, value: tp.Union[np.ndarray, glm.mat2, glm.vec4]):
        if isinstance(value, glm.vec4):
            value = sshr2mat(value)
        if isinstance(value, np.ndarray):
            self.__linear = glm.mat2(glm.vec2(value[:, 0]), glm.vec2(value[:, 1]))
        else:
            self.__linear = glm.mat2(value)

    # ----- derived properties -----

    @property
    def affine(self):
        """glm.mat3: the 3x3 homogeneous (column-major) matrix of the transformation."""
        return glm.mat3(
            glm.vec3(self.linear[0], 0),
            glm.vec3(self.linear[1], 0),
            glm.vec3(self.offset, 1),
        )

    @property
    def matrix(self):
        return np.array(self.affine)

    @property
    def matrix(self):
        a = np.empty((3, 3))
        a[:2, :2] = self.weight
        a[:2, 2] = self.bias
        a[2, :2] = 0
        a[2, 2] = 1
        return a

    matrix.__doc__ = Aff.matrix.__doc__

    @property
    def det(self):
        return glm.determinant(self.linear)

    det.__doc__ = Aff.det.__doc__

    # ----- methods -----

    def __init__(
        self,
        offset=np.zeros(2),
        linear: tp.Union[np.ndarray, glm.mat2, glm.vec4] = glm.mat2(),
    ):
        """Initialises a 2D affine transformation from an offset and a linear part. See the class
        docstring for the accepted types."""
        self.offset = offset
        self.linear = linear

    def __repr__(self):
        return f"Aff2d(offset={repr(self.offset)}, linear={repr(self.linear)})"


# ----- casting -----


_bc.register_cast(
    Aff2d, Aff, lambda x: Aff(weights=x.weight, bias=x.offset, check_shapes=False)
)
_bc.register_cast(Aff, Aff2d, lambda x: Aff2d(offset=x.bias, linear=x.weight))
_bc.register_castable(Aff, Aff2d, lambda x: x.ndim == 2)


# ----- transform functions -----


def transform_Aff2d_on_Moments2d(aff_tfm, moments):
    """Transforms a Moments2d using a 2D affine transformation.

    The mean and covariance of the points are transformed accordingly and the 0th-order moment is
    scaled by the absolute determinant of the linear part. This function is registered with
    :func:`mt.geo.transform`.

    Parameters
    ----------
    aff_tfm : Aff2d
        2D affine transformation
    moments : Moments2d
        2D moments

    Returns
    -------
    Moments2d
        affine-transformed 2D moments
    """
    A = aff_tfm.linear
    old_m0 = moments.m0
    old_mean = moments.mean_glm
    old_cov = moments.cov_glm
    new_mean = A * old_mean + aff_tfm.offset
    new_cov = A * old_cov * glm.transpose(A.T)
    new_m0 = old_m0 * abs(aff_tfm.det)
    new_m1 = new_m0 * new_mean
    new_m2 = new_m0 * (np.outer(new_mean, new_mean) + new_cov)
    return Moments2d(new_m0, new_m1, new_m2)


register_transform(Aff2d, Moments2d, transform_Aff2d_on_Moments2d)


def transform_Aff2d_on_ndarray(aff_tfm, point_array):
    """Transforms an array of 2D points using a 2D affine transformation.

    Parameters
    ----------
    aff_tfm : Aff2d
        a 2D affine transformation
    point_array : numpy.ndarray
        an array of points whose last dimension has length 2, e.g. shape `(2,)`, `(N, 2)` or
        `(B, N, 2)`. Each point is an `(x, y)` pair.

    Returns
    -------
    numpy.ndarray
        the affine-transformed point array, with the same shape as the input

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo2d import translate2d, scale2d, transform_Aff2d_on_ndarray
    >>> tfm = translate2d(1, 2) * scale2d(2)
    >>> transform_Aff2d_on_ndarray(tfm, np.array([[0., 0.], [1., 1.]]))
    array([[1., 2.],
           [3., 4.]])
    """
    return point_array @ aff_tfm.weight.T + aff_tfm.bias


register_transform(Aff2d, np.ndarray, transform_Aff2d_on_ndarray)
register_transformable(Aff2d, np.ndarray, lambda x, y: y.shape[-1] == 2)


def transform_Aff2d_on_PointList2d(aff_tfm, point_list):
    """Transforms a 2D point list using a 2D affine transformation.

    Parameters
    ----------
    aff_tfm : Aff2d
        a 2D affine transformation
    point_list : PointList2d
        a 2D point list

    Returns
    -------
    PointList2d
        affine-transformed point list
    """
    return PointList2d(point_list.points @ aff_tfm.weight.T + aff_tfm.bias, check=False)


register_transform(Aff2d, PointList2d, transform_Aff2d_on_PointList2d)


def transform_Aff2d_on_Polygon(aff_tfm, poly):
    """Transforms a polygon using a 2D affine transformation.

    Parameters
    ----------
    aff_tfm : Aff2d
        a 2D affine transformation
    poly : Polygon
        a 2D polygon

    Returns
    -------
    Polygon
        affine-transformed polygon
    """
    return Polygon(poly.points @ aff_tfm.weight.T + aff_tfm.bias, check=False)


register_transform(Aff2d, Polygon, transform_Aff2d_on_Polygon)


# ----- useful 2D transformations -----


def swapAxes2d():
    """Returns the affine transformation that swaps the x-axis with the y-axis.

    It maps `(x, y)` to `(y, x)`.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import swapAxes2d
    >>> transform(swapAxes2d(), np.array([[1., 2.]]))
    array([[2., 1.]])
    """
    return Aff2d(linear=glm.mat2([0, 1], [1, 0]))


def flipLR2d(width):
    """Returns a left-right flip for a given width.

    It maps `(x, y)` to `(width - x, y)`.

    Parameters
    ----------
    width : float
        the width of the coordinate range being flipped

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import flipLR2d
    >>> transform(flipLR2d(10), np.array([[2., 3.]]))
    array([[8., 3.]])
    """
    return Aff2d.from_matrix(np.array([[-1, 0, width], [0, 1, 0]]))


def flipUD2d(height):
    """Returns an up-down flip for a given height.

    It maps `(x, y)` to `(x, height - y)`.

    Parameters
    ----------
    height : float
        the height of the coordinate range being flipped

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import flipUD2d
    >>> transform(flipUD2d(10), np.array([[2., 3.]]))
    array([[2., 7.]])
    """
    return Aff2d.from_matrix(np.array([[1, 0, 0], [0, -1, height]]))


def shearX2d(h):
    """Returns the shearing along the x-axis.

    It maps `(x, y)` to `(x + h*y, y)`.

    Parameters
    ----------
    h : float
        the shearing factor

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import shearX2d
    >>> transform(shearX2d(0.5), np.array([[0., 2.]]))
    array([[1., 2.]])
    """
    return Aff2d(linear=glm.vec4(1, 1, h, 0))


def shearY2d(h):
    """Returns the shearing along the y-axis.

    It maps `(x, y)` to `(x, y + h*x)`.

    Parameters
    ----------
    h : float
        the shearing factor

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import shearY2d
    >>> transform(shearY2d(0.5), np.array([[2., 0.]]))
    array([[2., 1.]])
    """
    return Aff2d(linear=glm.mat2([1, h], [0, 1]))


def originate2d(tfm, x, y):
    """Tweaks a 2D affine transformation so that it acts as if it originates at (x,y) instead of
    (0,0).

    The result is `translate2d(x, y) * tfm * translate2d(-x, -y)`. For example, a scaling or a
    rotation obtained this way leaves the point `(x, y)` fixed.

    Parameters
    ----------
    tfm : Aff2d
        the transformation to be tweaked
    x : float
        x-coordinate of the new origin
    y : float
        y-coordinate of the new origin

    Returns
    -------
    Aff2d
        the tweaked transformation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import originate2d, scale2d
    >>> t = originate2d(scale2d(2), 1, 1)
    >>> transform(t, np.array([[1., 1.], [2., 2.]]))
    array([[1., 1.],
           [3., 3.]])
    """
    return Aff2d(offset=np.array((x, y))).conjugate(tfm)


def rotate2d(theta, x, y):
    """Returns the rotation about a reference point (x,y).

    The rotation is in the standard convention, from the x-axis towards the y-axis. Note that in
    image coordinates (y pointing downward), this looks clockwise on the screen.

    Parameters
    ----------
    theta : float
        the rotation angle, in radian
    x : float
        x-coordinate of the reference (fixed) point
    y : float
        y-coordinate of the reference (fixed) point

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import rotate2d
    >>> transform(rotate2d(np.pi / 2, 1, 1), np.array([[2., 1.]])).round(6) + 0.0
    array([[1., 2.]])
    """
    return originate2d(Aff2d(linear=glm.vec4(1, 1, 0, theta)), x, y)


def translate2d(x, y):
    """Returns the translation by the vector `(x, y)`.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import translate2d
    >>> transform(translate2d(1, 2), np.array([[10., 10.]]))
    array([[11., 12.]])
    """
    return Aff2d(offset=np.array([x, y]))


def scale2d(scale_x=1, scale_y=None):
    """Returns the scaling about the origin.

    Parameters
    ----------
    scale_x : float
        the scaling factor along the x-axis
    scale_y : float, optional
        the scaling factor along the y-axis. If None, it is the same as `scale_x` (uniform scaling).

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import scale2d
    >>> transform(scale2d(2, 3), np.array([[1., 1.]]))
    array([[2., 3.]])
    >>> transform(scale2d(2), np.array([[1., 1.]]))
    array([[2., 2.]])
    """
    if scale_y is None:
        scale_y = scale_x
    return Aff2d(linear=glm.vec4(scale_x, scale_y, 0, 0))


def crop2d(tl, br=None):
    """Transforms an axis-aligned rectangle into [(0,0),(1,1)].

    Parameters
    ----------
    tl : 2d point (x,y)
        coordinates to be mapped to (0,0) if `br` is specified. If `br` is not specified, then the
        transformation is 2d scaling. In other words, (0,0) is mapped to (0,0) and tl is mapped to
        (1,1).
    br : 2d point (x,y), optional
        If specified, coordinates to be mapped to (1,1).

    Returns
    -------
    Aff2d
        A transformation that maps the crop given by `tl` and `br` to the unit square
        [(0,0),(1,1)] (i.e. it normalises coordinates in the crop). Its inverse maps the unit square
        to the crop.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import crop2d
    >>> t = crop2d((10, 20), (30, 60))
    >>> transform(t, np.array([[10., 20.], [30., 60.], [20., 40.]])).round(6)
    array([[0. , 0. ],
           [1. , 1. ],
           [0.5, 0.5]])
    >>> transform(crop2d((2, 4)), np.array([[2., 4.]]))
    array([[1., 1.]])
    """
    if br is None:
        return scale2d(1.0 / tl[0], 1.0 / tl[1])
    return Aff2d(
        offset=np.array([-tl[0] / (br[0] - tl[0]), -tl[1] / (br[1] - tl[1])]),
        linear=glm.vec4(1.0 / (br[0] - tl[0]), 1.0 / (br[1] - tl[1]), 0, 0),
    )


def crop_rect(r: Rect) -> Aff2d:
    """Transforms an axis-aligned Rect into [(0,0),(1,1)].

    This is :func:`crop2d` applied to the corners of the rectangle.

    Parameters
    ----------
    r : Rect
        the input rectangle

    Returns
    -------
    Aff2d
        A transformation that maps the rectangle `r` to the unit square [(0,0),(1,1)]. Its inverse
        maps the unit square to `r`.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import Rect, crop_rect
    >>> transform(crop_rect(Rect(10, 20, 30, 60)), np.array([[30., 60.]])).round(6)
    array([[1., 1.]])

    See Also
    --------
    uncrop_rect : the reverse operation
    """
    return crop2d((r.min_x, r.min_y), (r.max_x, r.max_y))


def uncrop_rect(tfm: Aff2d) -> Rect:
    """Transforms a 2D affine transformation back to a crop, ignoring shearing and rotation.

    Parameters
    ----------
    tfm : Aff2d
        A 2D affine transformation supposedly transforms a crop to [(0,0),(1,1)]

    Returns
    -------
    Rect
        the recovered rectangle (the image of the unit square under the inverse of `tfm`, using
        only its translation and scaling). Coordinates are float32-precise, so expect small
        rounding noise.

    Examples
    --------
    >>> from mt.geo2d import Rect, crop_rect, uncrop_rect
    >>> r = uncrop_rect(crop_rect(Rect(10, 20, 30, 60)))
    >>> [round(v, 3) for v in r.to_json()]
    [10.0, 20.0, 30.0, 60.0]

    See Also
    --------
    crop_rect : the reverse operation
    """
    tfm = ~tfm  # take the inverse
    x, y = tfm.bias
    linear = mat2sshr(tfm.linear)
    w, h = linear.xy
    return Rect(x, y, x + w, y + h)


def rect2rect(src_rect: Rect, dst_rect: Rect, eps=1e-7) -> Aff2d:
    """Returns an Aff2d that transforms pixels in a source Rect to pixels in a destination Rect.

    The transformation ensures that the source corner pixels match with the destination corner
    pixels. It is a scaling plus a translation (no rotation or reflection), mapping the center of
    the source to the center of the destination. If both the source and destination widths
    (respectively heights) are zero, the scale along x (respectively y) is set to 0.

    Parameters
    ----------
    src_rect : Rect
        source rectangle
    dst_rect : Rect
        destination rectangle
    eps : float
        epsilon to detect zero source width or height

    Returns
    -------
    Aff2d
        the output transformation

    Raises
    ------
    ValueError
        if the source width (or height) is zero but the destination width (or height) is not

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import transform
    >>> from mt.geo2d import Rect, rect2rect
    >>> t = rect2rect(Rect(0, 0, 10, 10), Rect(100, 100, 120, 130))
    >>> transform(t, np.array([[0., 0.], [10., 10.], [5., 5.]]))
    array([[100., 100.],
           [120., 130.],
           [110., 115.]])
    """
    if abs(src_rect.w) < eps:
        if abs(dst_rect.w) < eps:
            sx = 0.0
        else:
            raise ValueError(
                f"Source width is zero {src_rect.w} but destination width is not {dst_rect.w}."
            )
    else:
        sx = dst_rect.w / src_rect.w
    if abs(src_rect.h) < eps:
        if abs(dst_rect.h) < eps:
            sy = 0.0
        else:
            raise ValueError(
                f"Source height is zero {src_rect.h} but destination height is not {dst_rect.h}."
            )
    else:
        sy = dst_rect.h / src_rect.h
    return (
        translate2d(dst_rect.cx, dst_rect.cy)
        * scale2d(sx, sy)
        / translate2d(src_rect.cx, src_rect.cy)
    )


def rect2rect_tf(src_rects, dst_rects):
    """A tensorflow version of :func:`rect2rect` where inputs and outputs are replaced by tensors.

    Unlike :func:`rect2rect`, a zero source width or height does not raise an error: the
    corresponding scale becomes 0 (via `tf.math.divide_no_nan`).

    Parameters
    ----------
    src_rects : tf.Tensor
        source rectangles of shape `[batch, 4]` where each batch item is `[min_x, min_y, max_x,
        max_y]`
    dst_rects : tf.Tensor
        destination rectangles of shape `[batch, 4]` where each batch item is
        `[min_x, min_y, max_x, max_y]`

    Returns
    -------
    tfm_array : tf.Tensor
        the output transformations of shape `[batch, 3, 3]` where each item is a 3x3 affine
        transformation matrix mapping source coordinates to destination coordinates
    """
    from mt import tf

    # batch_size = src_rects.shape[0]

    # center points
    src_pts = (src_rects[:, 0:2] + src_rects[:, 2:4]) * 0.5
    dst_pts = (dst_rects[:, 0:2] + dst_rects[:, 2:4]) * 0.5

    # lengths
    src_lens = src_rects[:, 2:4] - src_rects[:, 0:2]
    dst_lens = dst_rects[:, 2:4] - dst_rects[:, 0:2]

    # scales
    scales = tf.math.divide_no_nan(dst_lens, src_lens)

    # translations: src_pts*scales + transalations = dst_pts
    translations = dst_pts - src_pts * scales

    # parts of the returning array of 3x3 matrices
    sx = tf.tensordot(
        scales[:, 0],
        tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        axes=0,
    )
    sy = tf.tensordot(
        scales[:, 1],
        tf.constant(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        axes=0,
    )
    tx = tf.tensordot(
        translations[:, 0],
        tf.constant(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        axes=0,
    )
    ty = tf.tensordot(
        translations[:, 1],
        tf.constant(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        axes=0,
    )
    ones = tf.expand_dims(
        tf.constant(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        axis=0,
    )

    # returning
    return sx + sy + tx + ty + ones
