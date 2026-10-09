"""There are many definitions of an ellipse. In our case, an ellipse is an affine transform of the
unit circle x^2+y^2=1."""

from mt import np, glm
from mt.base.casting import register_cast, register_castable, cast

from ..geo import (
    GeometricObject,
    TwoD,
    transform,
    register_transform,
    register_approx,
    register_upper_bound,
    register_lower_bound,
)
from ..geond import Hyperellipsoid
from .affine import Aff2d, Aff
from .rect import Rect
from .moments import Moments2d


__all__ = [
    "Ellipse",
    "cast_Ellipse_to_Moments2d",
    "approx_Moments2d_to_Ellipse",
    "upper_bound_Ellipse_to_Rect",
    "lower_bound_Rect_to_Ellipse",
    "transform_Aff2d_on_Ellipse",
]


class Ellipse(TwoD, GeometricObject):
    """Ellipse, defined as an affine transform the unit circle x^2+y^2=1.

    If the unit circle is parameterised by `(cos(t), sin(t))` where `t \\in [0,2\\pi)` then the
    ellipse is parameterised by `f0 + f1 cos(t) + f2 sin(t)`, where `f0` is the bias vector, `f1`
    and `f2` are the first and second column of the weight matrix respectively, of the affine
    transformation. `f1` and `f2` are called first and second axes of the ellipse.

    Note that this representation is not unique, the same ellipse can be represented by an infinite
    number of affine transforms of the unit circle. To make the representation more useful, we
    further assert that when f1 and f2 are perpendicular (linearly independent), the ellipse is
    normalised. In other words, the weight matrix is a unitary matrix multiplied by a diagonal
    matrix. After normalisation there is only a finite number of ways to represent the same
    ellipse. You can normalise either at initialisation time, or later by invoking member function
    `normalised`.

    Parameters
    ----------
    aff_tfm : Aff2d
        an affine transformation of the unit circle
    make_normalised : bool
        whether or not to adjust the affine transformation (via an SVD of its linear part) to make
        a normalised representation of the ellipse. Default is True.

    Raises
    ------
    ValueError
        if `aff_tfm` is not an instance of :class:`~mt.geo2d.Aff2d`

    Attributes
    ----------
    aff_tfm : Aff2d
        the affine transformation (after normalisation if requested)
    f0 : numpy.ndarray
        the origin (center), shape (2,)
    f1, f2 : numpy.ndarray
        the first and second axes, shape (2,)
    area : float
        the area of the ellipse's interior

    Examples
    --------
    >>> from mt import np, glm, geo2d
    >>> a = geo2d.Aff2d(offset=np.array([5., 5.]), linear=glm.vec4(3, 2, 0, 0))  # scales (3,2)
    >>> e = geo2d.Ellipse(a)
    >>> e
    Ellipse(aff_tfm=Aff2d(offset=vec2( 5, 5 ), linear=mat2x2(( 3, 0 ), ( 0, 2 ))))
    >>> e.f0, e.f1, e.f2
    (array([5., 5.], dtype=float32), array([3., 0.], dtype=float32), array([0., 2.], dtype=float32))
    >>> round(e.area, 3)  # pi * 3 * 2
    18.85

    Notes
    -----
    Ellipse follows the convention of Ellipsoid closely.

    See Also
    --------
    mt.geond.ellipsoid.Ellipsoid
    """

    def __init__(self, aff_tfm, make_normalised=True):
        """Initialises an ellipse with an affine transformation. See the class docstring."""
        if not isinstance(aff_tfm, Aff2d):
            raise ValueError("Only an instance of class `Aff2d` is accepted.")
        if make_normalised:
            v = glm.svd2(aff_tfm.linear)
            linear = glm.rot2(v.z) * glm.mat2diag(v.xy)
            aff_tfm = Aff2d(offset=aff_tfm.offset, linear=linear)
        self.aff_tfm = aff_tfm

    def __repr__(self):
        return f"Ellipse(aff_tfm={self.aff_tfm})"

    @property
    def f0(self):
        """numpy.ndarray: the origin (center), the bias of the affine transformation."""
        return self.aff_tfm.bias

    @property
    def f1(self):
        """numpy.ndarray: the first axis, the first column of the weight matrix."""
        return self.aff_tfm.weight[:, 0]

    @property
    def f2(self):
        """numpy.ndarray: the second axis, the second column of the weight matrix."""
        return self.aff_tfm.weight[:, 1]

    @property
    def area(self):
        """The absolute area of the ellipse's interior, `pi*abs(det)` of the linear part."""
        return np.pi * abs(self.aff_tfm.det)

    def normalised(self):
        """Returns an equivalent ellipse where f1 and f2 are perpendicular (linearly
        independent).

        Returns
        -------
        Ellipse
            the same ellipse, represented with a weight matrix being a rotation times a diagonal
            matrix
        """
        return Ellipse(self.aff_tfm, make_normalised=True)


# ----- casting -----


register_cast(
    Ellipse,
    Hyperellipsoid,
    lambda x: Hyperellipsoid(cast(x.aff_tfm, Aff), make_normalised=False),
)
register_cast(
    Hyperellipsoid,
    Ellipse,
    lambda x: Ellipse(cast(x.aff_tfm, Aff2d), make_normalised=False),
)
register_castable(Hyperellipsoid, Ellipse, lambda x: x.ndim == 2)


def cast_Ellipse_to_Moments2d(obj):
    """Extracts Moments2d from an Ellipse instance.

    The moments are those of the ellipse's interior: the moments of the unit disk transformed by
    the ellipse's affine transformation.

    Note that it currently fails, because the transformation of Moments2d by Aff2d
    (:func:`~mt.geo2d.transform_Aff2d_on_Moments2d`) raises AttributeError.
    """
    a = np.pi / 4
    moments = Moments2d(np.pi, [0, 0], [[a, 0], [0, a]])  # unit circle's moments
    return transform(obj.aff_tfm, moments)  # transform


register_cast(Ellipse, Moments2d, cast_Ellipse_to_Moments2d)


def approx_Moments2d_to_Ellipse(obj):
    """Approximates a Moments2d instance with a normalised Ellipse that has the same mean and
    covariance as the mean and covariance of the instance.

    Parameters
    ----------
    obj : Moments2d
        the moments to approximate

    Returns
    -------
    Ellipse
        the normalised ellipse whose interior has the mean and covariance of `obj` (the area
        `m0` of the moments is not used)

    Examples
    --------
    >>> from mt.geo2d import Moments2d, approx_Moments2d_to_Ellipse
    >>> m = Moments2d(1, [3, 4], [[13, 12], [12, 17]])  # mean (3,4), covariance diag(4,1)
    >>> approx_Moments2d_to_Ellipse(m)
    Ellipse(aff_tfm=Aff2d(offset=vec2( 3, 4 ), linear=mat2x2(( 4, 0 ), ( 0, 2 ))))
    """
    # A
    AAT = obj.cov * 4
    w, v = np.linalg.eig(AAT)
    A = v @ np.diag(np.sqrt(w))

    # aff_tfm
    aff_tfm = Aff2d(offset=obj.mean, linear=A)
    return Ellipse(aff_tfm)


register_approx(Moments2d, Ellipse, approx_Moments2d_to_Ellipse)


# ----- bounding -----


def upper_bound_Ellipse_to_Rect(obj):
    """Returns a bounding axis-aligned rectangle of the ellipse.

    Parameters
    ----------
    obj : Ellipse
        the ellipse to be upper-bounded

    Returns
    -------
    Rect
        a bounding Rect of the ellipse (the tightest axis-aligned one)

    Examples
    --------
    >>> from mt import np, glm
    >>> from mt.geo2d import Aff2d, Ellipse, upper_bound_Ellipse_to_Rect
    >>> e = Ellipse(Aff2d(offset=np.array([5., 5.]), linear=glm.vec4(3, 2, 0, 0)))
    >>> upper_bound_Ellipse_to_Rect(e)
    Rect(x=2.0, y=3.0, w=6.0, h=4.0)
    """
    weight = obj.aff_tfm.weight
    mx = np.linalg.norm(weight[0])
    my = np.linalg.norm(weight[1])
    cx, cy = obj.aff_tfm.bias
    return Rect(cx - mx, cy - my, cx + mx, cy + my)


register_upper_bound(Ellipse, Rect, upper_bound_Ellipse_to_Rect)


def lower_bound_Rect_to_Ellipse(x):
    """Returns an axis-aligned ellipse bounded by the given axis-aligned rectangle x.

    Parameters
    ----------
    x : Rect
        the rectangle from which the enclosed ellipse is computed

    Returns
    -------
    Ellipse
        the axis-aligned ellipse enclosed by the rectangle (touching the four sides)

    Raises
    ------
    ValueError
        if `x` is not a :class:`~mt.geo2d.Rect`

    Examples
    --------
    >>> from mt.geo2d import Rect, lower_bound_Rect_to_Ellipse
    >>> lower_bound_Rect_to_Ellipse(Rect(0, 0, 6, 4))
    Ellipse(aff_tfm=Aff2d(offset=vec2( 3, 2 ), linear=mat2x2(( 3, 0 ), ( 0, 2 ))))
    """
    if not isinstance(x, Rect):
        raise ValueError(f"Input type must be a `Rect`, '{x.__class__}' given.")
    return Ellipse(Aff2d(linear=glm.vec4(x.w / 2, x.h / 2, 0, 0), offset=x.center_pt))


register_lower_bound(Rect, Ellipse, lower_bound_Rect_to_Ellipse)


# ----- transform functions -----


def transform_Aff2d_on_Ellipse(aff_tfm, obj):
    """Affine-transforms an Ellipse. The resultant Ellipse has affine transformation
    `aff_tfm*obj.aff_tfm`.

    Parameters
    ----------
    aff_tfm  : Aff2d
        an 2D affine transformation
    obj : Ellipse
        an ellipse

    Returns
    -------
    Ellipse
        the affine-transformed ellipse

    Examples
    --------
    >>> from mt import np, glm
    >>> from mt.geo import transform
    >>> from mt.geo2d import Aff2d, Ellipse, translate2d
    >>> e = Ellipse(Aff2d(offset=np.array([5., 5.]), linear=glm.vec4(3, 2, 0, 0)))
    >>> transform(translate2d(1, 1), e)
    Ellipse(aff_tfm=Aff2d(offset=vec2( 6, 6 ), linear=mat2x2(( 3, 0 ), ( 0, 2 ))))
    """
    return Ellipse(aff_tfm * obj.aff_tfm)


register_transform(Aff2d, Ellipse, transform_Aff2d_on_Ellipse)
