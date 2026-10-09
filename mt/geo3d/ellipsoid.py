"""There are many definitions of an ellipsoid. In our case, an ellipsoid is an affine transform of
the unit sphere x^2+y^2+z^2=1.

The module provides :class:`Ellipsoid`, its upper bound by a :class:`mt.geo3d.Box`, its lower bound
from a box, and the function to transform it with an :class:`mt.geo3d.Aff3d`.
"""

from mt import np, glm
from mt.base.casting import register_cast, register_castable, cast

from ..geo import (
    GeometricObject,
    ThreeD,
    register_transform,
    register_upper_bound,
    register_lower_bound,
)
from ..geond import Hyperellipsoid
from .affine import Aff3d
from .box import Box


__all__ = ["Ellipsoid", "transform_Aff3d_on_Ellipsoid"]


class Ellipsoid(ThreeD, GeometricObject):
    r"""Ellipsoid, defined as an affine transform the unit sphere x^2+y^2+z^2=1.

    If the unit sphere is parameterised by `(cos(t_1), sin(t_1)*cos(t_2), sin(t_1)*sin(t_2))` where
    `t_1 \in [0,\pi)` and `t_2 \in [0,2\pi)` then the ellipsoid is parameterised by
    `f0 + f1 cos(t_1) + f2 sin(t_1) cos(t_2) + f3 sin(t_1) sin(t_2)`, where `f0` is the bias
    vector, `f1, f2, f3` are the columns of the weight matrix from left to right respectively, of
    the affine transformation. They are also called called axes of the ellipsoid.

    Note that this representation is not unique, the same ellipsoid can be represented by an
    infinite number of affine transforms of the unit sphere. To make the representation more useful
    we further assert that when the axes are perpendicular (linearly independent), the ellipsoid is
    normalised. In other words, the weight matrix is a unitary matrix multiplied by a diagonal
    matrix. After normalisation there is almost surely a finite number of ways to represent the
    same ellipsoid, except for singular cases. You can normalise either at initialisation time, or
    later by invoking member function `normalised`.

    Parameters
    ----------
    aff_tfm : Aff3d
        a 3D affine transformation. A `ValueError` is raised if it is not an instance of
        :class:`mt.geo3d.Aff3d`.
    make_normalised : bool
        whether or not to adjust the affine transformation to make a normalised representation of
        the ellipsoid, using the singular value decomposition of its linear part.

    Attributes
    ----------
    aff_tfm : Aff3d
        the (normalised) affine transformation of the unit sphere
    volume : float
        the volume of the ellipsoid's interior
    ndim : int
        3

    See Also
    --------
    mt.geo3d.Aff3d : the 3D affine transformation
    mt.geo3d.Box : an axis-aligned box, which can bound an ellipsoid
    mt.geond.Hyperellipsoid : the ellipsoid in any number of dimensions

    Notes
    -----
    - With `make_normalised=True` (the default), the constructor currently raises `TypeError`
      because it builds an :class:`Aff3d` with an unknown keyword `bias`. Use
      `make_normalised=False` until it is fixed. For the same reason `normalised()` and the
      affine transformation of an ellipsoid (:func:`transform_Aff3d_on_Ellipsoid`) fail.
    - The upper bound to a :class:`Box` refers to an undefined name `off`, and the lower bound
      from a :class:`Box` needs a cast from `Dlt` to `Aff3d` that is not registered, so both
      raise an error.
    - `volume` is `4*pi/3*det`, which is negative if the determinant of the linear part is.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo3d import Aff3d, Ellipsoid
    >>> a = Aff3d(offset=np.array([1., 2., 3.]), linear=np.diag([2., 3., 4.]))
    >>> e = Ellipsoid(a, make_normalised=False)
    >>> e
    Ellipsoid(aff_tfm=Aff3d(offset=vec3( 1, 2, 3 ), linear=mat3x3(( 2, 0, 0 ), ...))))
    >>> e.ndim
    3
    >>> round(e.volume, 3)
    100.531
    >>> e.aff_tfm.matrix
    array([[2., 0., 0., 1.],
           [0., 3., 0., 2.],
           [0., 0., 4., 3.],
           [0., 0., 0., 1.]], dtype=float32)
    """

    # ----- construction -----

    def __init__(self, aff_tfm, make_normalised=True):
        """Initialises an ellipsoid with an affine transformation.

        Parameters
        ----------
        aff_tfm : Aff3d
            a 3D affine transformation of the unit sphere
        make_normalised : bool
            whether or not to normalise the representation. See the class docstring.

        Raises
        ------
        ValueError
            if `aff_tfm` is not an instance of :class:`Aff3d`
        """
        if not isinstance(aff_tfm, Aff3d):
            raise ValueError("Only an instance of class `Aff3d` is accepted.")
        if make_normalised:
            U, S, VT = glm.svd3(aff_tfm.linear)
            aff_tfm = Aff3d(bias=aff_tfm.bias, linear=U * glm.mat3diag(S))
        self.aff_tfm = aff_tfm

    def __repr__(self):
        return f"Ellipsoid(aff_tfm={self.aff_tfm})"

    @property
    def volume(self):
        """The volume of the ellipsoid's interior, computed as `4*pi/3*det` where `det` is the
        determinant of the linear part of the affine transformation."""
        return (4 * np.pi / 3) * self.aff_tfm.det

    @property
    def surface_area(self):
        """The surface area of the ellipsoid.

        Raises
        ------
        NotImplementedError
            always, as it is not implemented yet
        """
        raise NotImplementedError(
            "It's quite complicated to compute the surface area of an ellipsoid. I will only be "
            "implemented when there's a real demand. MT has to check with the Wiki page first."
        )

    def normalised(self):
        """Returns an equivalent ellipsoid where f1, f2 and f3 are perpendicular (linearly
        independent).

        Returns
        -------
        Ellipsoid
            a new ellipsoid with a normalised affine transformation
        """
        return Ellipsoid(self.aff_tfm, make_normalised=True)


# ----- casting -----


register_cast(
    Ellipsoid,
    Hyperellipsoid,
    lambda x: Hyperellipsoid(x.aff_tfm, make_normalised=False),
)
register_cast(
    Hyperellipsoid, Ellipsoid, lambda x: Ellipsoid(x.aff_tfm, make_normalised=False)
)
register_castable(Hyperellipsoid, Ellipsoid, lambda x: x.ndim == 3)


# ----- bounding -----


def upper_bound_Ellipsoid_to_Box(obj):
    """Returns a bounding axis-aligned box of the ellipsoid.

    The box is meant to be centered at the offset of the affine transformation, with half side
    lengths equal to the norms of the rows of the weight matrix. It is registered as the upper
    bound from :class:`Ellipsoid` to :class:`Box`.

    Parameters
    ----------
    obj : Ellipsoid
        the ellipsoid to be upper-bounded

    Returns
    -------
    Box
        a bounding Box of the ellipsoid

    Notes
    -----
    The current body refers to the undefined name `off`, so calling it raises `NameError`.
    """
    weight = obj.aff_tfm.weight
    c = off.aff_tfm.bias
    m = np.array([np.linalg.norm(weight[i]) for i in range(3)])
    return Box(min_coords=c - m, max_coords=c + m)


register_upper_bound(Ellipsoid, Box, upper_bound_Ellipsoid_to_Box)


def lower_bound_Box_to_Ellipsoid(x):
    """Returns an axis-aligned ellipsoid bounded by the given axis-aligned box.

    The ellipsoid is meant to be the one inscribed in the box. It is registered as the lower bound
    from :class:`Box` to :class:`Ellipsoid`.

    Parameters
    ----------
    x : Box
        the box from which the enclosed ellipsoid is computed

    Returns
    -------
    Ellipsoid
        the axis-aligned ellipsoid enclosed by the box

    Raises
    ------
    ValueError
        if `x` is not a :class:`Box`

    Notes
    -----
    The current body casts the box's dilatation to :class:`Aff3d`, which has no registered cast, so
    a valid box raises `NotImplementedError`.
    """
    if not isinstance(x, Box):
        raise ValueError(f"Input type must be a `Box`, '{x.__class__}' given.")
    return Ellipsoid(cast(x.dlt_tfm, Aff3d))


register_lower_bound(Box, Ellipsoid, lower_bound_Box_to_Ellipsoid)


# ----- transform functions -----


def transform_Aff3d_on_Ellipsoid(aff_tfm, obj):
    """Affine-transforms an Ellipsoid. The resultant Ellipsoid has affine transformation
    `aff_tfm*obj.aff_tfm`.

    It is registered as the function applying an :class:`Aff3d` to an :class:`Ellipsoid`.

    Parameters
    ----------
    aff_tfm  : Aff3d
        an affine transformation
    obj : Ellipsoid
        an ellipsoid

    Returns
    -------
    Ellipsoid
        the affine-transformed ellipsoid

    Notes
    -----
    The result is built with the default `make_normalised=True`, which currently raises
    `TypeError` (see :class:`Ellipsoid`).
    """
    return Ellipsoid(aff_tfm * obj.aff_tfm)


register_transform(Aff3d, Ellipsoid, transform_Aff3d_on_Ellipsoid)
