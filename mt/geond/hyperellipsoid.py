"""There are many definitions of a hyperellipsoid. In our case, a hyperellipsoid is an affine
transform of the unit hypersphere x^2+y^2+...=1.

The module provides :class:`Hyperellipsoid`, its upper bound by a :class:`mt.geond.Hyperbox`, its
lower bound from a hyperbox, and the function to transform it with an affine map.
"""

from mt import np
from mt.base.casting import register_cast, cast
from mt.base.deprecated import deprecated_func

from ..geo import (
    GeometricObject,
    register_transform,
    register_transformable,
    register_upper_bound,
    register_lower_bound,
)
from .affine import Aff
from .hyperbox import Hyperbox


__all__ = ["Hyperellipsoid", "transform_Aff_on_Hyperellipsoid"]


class Hyperellipsoid(GeometricObject):
    """Hyperellipsoid, defined as an affine transform the unit hypersphere x^2+y^2+z^2+...=1.

    If the unit hypersphere is parameterised by `(cos(t_1), sin(t_1)*cos(t_2),
    sin(t_1)*sin(t_2)*cos(t_3), ...)` where `t_1, t_2, ... \\in [0,\\pi)` but the last `t_{dim-1}
    \\in [0,2\\pi)` then the hyperellipsoid is parameterised by `f0 + f1 cos(t_1) + f2 sin(t_1)
    cos(t_2) + f3 sin(t_1) sin(t_2) sin(t_3) cos(t_4) + ...`, where `f0` is the bias vector, `f1,
    f2, ...` are the columns of the weight matrix from left to right respectively, of the affine
    transformation. They are also called called axes of the hyperellipsoid.

    Note that this representation is not unique, the same hyperellipsoid can be represented by an
    infinite number of affine transforms of the unit hypersphere. To make the representation more
    useful and more unique, we further assert that when the axes are perpendicular (linearly
    independent), the hyperellipsoid is normalised. In other words, the weight matrix is a unitary
    matrix multiplied by a diagonal matrix. After normalisation there can still be an infinite
    number of ways to represent the same hyperellipsoid, but only in singular cases. You can
    normalise either at initialisation time, or later by invoking member function `normalised`.

    Parameters
    ----------
    aff_tfm : Aff
        an affine transformation. A `ValueError` is raised if it is not an instance of
        :class:`Aff`.
    make_normalised : bool
        whether or not to adjust the affine transformation to make a normalised representation of
        the hyperellipsoid. Normalising replaces the weight matrix by `U @ diag(S)` from its
        singular value decomposition, with the singular values `S` sorted in decreasing order, so
        the axes (columns) can be permuted compared to the input.

    Attributes
    ----------
    aff_tfm : Aff
        the (normalised) affine transformation of the unit hypersphere

    See Also
    --------
    Aff : affine transformation
    Hyperbox : axis-aligned box, an upper or lower bound of a hyperellipsoid
    mt.geo3d.Ellipsoid : the 3D version

    Notes
    -----
    - `ndim` currently returns the bias vector `aff_tfm.bias` instead of its length.
    - The upper bound to a :class:`Hyperbox` refers to undefined names (`off`, `self`) and so
      raises `NameError`.
    - Because of the `ndim` issue above, the transformable check of an :class:`Aff` applied on a
      hyperellipsoid returns an array rather than a bool, so `aff << hyperellipsoid` fails. Calling
      :func:`transform_Aff_on_Hyperellipsoid` directly works.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geond.hyperellipsoid import Aff, Hyperellipsoid
    >>> a = Aff(bias=np.array([2,3]), weight=np.diag([4,5]))
    >>> e = Hyperellipsoid(a)
    >>> e
    Hyperellipsoid(aff_tfm=Aff(weight_diagonal=[0. 0.], bias=[2 3]))
    >>> e.aff_tfm.matrix
    array([[0., 4., 2.],
           [5., 0., 3.],
           [0., 0., 1.]])
    >>> Hyperellipsoid(a, make_normalised=False).aff_tfm.matrix
    array([[4., 0., 2.],
           [0., 5., 3.],
           [0., 0., 1.]])
    """

    # ----- base adaptation -----

    @property
    def ndim(self):
        return self.aff_tfm.bias

    # ----- construction ------

    def __init__(self, aff_tfm, make_normalised=True):
        """Initialises a hyperellipsoid with an affine transformation.

        Parameters
        ----------
        aff_tfm : Aff
            an affine transformation of the unit hypersphere
        make_normalised : bool
            whether or not to normalise the representation. See the class docstring.

        Raises
        ------
        ValueError
            if `aff_tfm` is not an instance of :class:`Aff`
        """
        if not isinstance(aff_tfm, Aff):
            raise ValueError("Only an instance of class `Aff` is accepted.")
        if make_normalised:
            U, S, VT = np.linalg.svd(aff_tfm.weight, full_matrices=False)
            aff_tfm = Aff(bias=aff_tfm.bias, weight=U @ np.diag(S))
        self.aff_tfm = aff_tfm

    def __repr__(self):
        return f"Hyperellipsoid(aff_tfm={self.aff_tfm})"

    # ----- useful functions

    def normalised(self):
        """Returns an equivalent hyperellipsoid where f1 and f2 are perpendicular (linearly
        independent).

        Returns
        -------
        Hyperellipsoid
            a new hyperellipsoid with a normalised affine transformation
        """
        return Hyperellipsoid(self.aff_tfm, make_normalised=True)


# ----- bounding -----


def upper_bound_Hyperellipsoid_to_Hyperbox(obj):
    """Returns a bounding axis-aligned box of the hyperellipsoid.

    The box is centered at the bias of the affine transformation, with half side lengths equal to
    the norms of the rows of the weight matrix. It is registered as the upper bound from
    :class:`Hyperellipsoid` to :class:`Hyperbox`.

    Parameters
    ----------
    obj : Hyperellipsoid
        the hyperellipsoid to be upper-bounded

    Returns
    -------
    Hyperbox
        a bounding Hyperbox of the hyperellipsoid

    Notes
    -----
    The current body refers to the undefined names `off` and `self`, so calling it raises
    `NameError`.
    """
    weight = obj.aff_tfm.weight
    c = off.aff_tfm.bias
    m = np.array([np.linalg.norm(weight[i]) for i in range(self.ndim)])
    return Hyperbox(min_coords=c - m, max_coords=c + m)


register_upper_bound(Hyperellipsoid, Hyperbox, upper_bound_Hyperellipsoid_to_Hyperbox)


def lower_bound_Hyperbox_to_Hyperellipsoid(x):
    """Returns an axis-aligned hyperellipsoid bounded by the given axis-aligned box.

    The returned hyperellipsoid is the one inscribed in the box, obtained by casting the box's
    dilatation to an affine transformation. It is registered as the lower bound from
    :class:`Hyperbox` to :class:`Hyperellipsoid`.

    Parameters
    ----------
    x : Hyperbox
        the box from which the enclosed hyperellipsoid is computed

    Returns
    -------
    Hyperellipsoid
        the axis-aligned hyperellipsoid enclosed by the box

    Raises
    ------
    ValueError
        if `x` is not an instance of :class:`Hyperbox`

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo import lower_bound
    >>> from mt.geond import Hyperbox
    >>> from mt.geond.hyperellipsoid import Hyperellipsoid
    >>> b = Hyperbox(np.array([0., 0.]), np.array([2., 4.]))
    >>> e = lower_bound(b, Hyperellipsoid)
    >>> e.aff_tfm.bias
    array([1., 2.])
    >>> e.aff_tfm.weight
    array([[0., 1.],
           [2., 0.]])
    """
    if not isinstance(x, Hyperbox):
        raise ValueError(
            f"Input type must be a `Hyperbox`, '{x.__class__}' given."
        )
    return Hyperellipsoid(cast(x.dlt_tfm, Aff))


register_lower_bound(Hyperbox, Hyperellipsoid, lower_bound_Hyperbox_to_Hyperellipsoid)


# ----- transform functions -----


def transform_Aff_on_Hyperellipsoid(aff_tfm, obj):
    """Affine-transforms a hyperellipsoid. The resultant Hyperellipsoid has affine transformation
    `aff_tfm*obj.aff_tfm`.

    It is registered as the function applying an :class:`Aff` to a :class:`Hyperellipsoid`.

    Parameters
    ----------
    aff_tfm  : Aff
        an affine transformation
    obj : Hyperellipsoid
        a hyperellipsoid

    Returns
    -------
    Hyperellipsoid
        the affine-transformed hyperellipsoid
    """
    return Hyperellipsoid(aff_tfm * obj.aff_tfm)


register_transform(Aff, Hyperellipsoid, transform_Aff_on_Hyperellipsoid)
register_transformable(
    Aff, Hyperellipsoid, lambda aff_tfm, obj: aff_tfm.ndim == obj.ndim
)
