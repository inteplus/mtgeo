"""Hyperrectangle.

The module provides :class:`Hyperbox`, an axis-aligned n-dimensional box stored as a
:class:`mt.geond.Dlt`, and registers the function that joins the volumes of two hyperboxes (see
:func:`mt.geo.join_volume`).
"""

from mt import np
from ..geo import GeometricObject, join_volume, register_join_volume
from .dilatation import Dlt
from .iou import iou_impl

__all__ = ["Hyperbox"]


class Hyperbox(GeometricObject):
    """Axis-aligned n-dimensional hyperrectangle.

    An axis-aligned n-dimensional box/hyperrectangle is defined as the set of points of the
    hypercube [-1,1]^n transformed by a dilatation.

    Note that is is a many-to-one representation. For each box, there are up to 2^n dilatations that
    map the hypbercube [-1,1]^n to it.

    The box is "valid" if all the scales of its dilatation are non-negative, i.e. `min_coords <=
    max_coords` coordinate-wise. A box built from `min_coords > max_coords` keeps negative scales
    unless `force_valid` is True. Note that the properties `min_coords` and `max_coords` always
    return the element-wise minimum and maximum of the two corners, so they are sorted even for an
    invalid box, whereas `signed_volume` and `is_valid()` expose the validity.

    Besides the properties below, the box has the following methods.

    - `is_valid()`: whether all the scales are non-negative.
    - `validated()`: returns a valid box built from `min_coords` and `max_coords`.
    - `intersect(other)`: returns the box with the intersection of the coordinate ranges.
    - `box_union(other)`: returns the smallest box containing both boxes.
    - `iou(other, epsilon=1e-7)`: returns the intersection over union, the denominator being
      stabilised by `epsilon`.

    Parameters
    ----------
    min_coords : 1d array or Dlt
        array of minimum coordinate values for all the dimensions, or the dilatation itself. In
        case of the latter, the `max_coords` argument is ignored.
    max_coords : 1d array
        array of maximum coordinate values for all the dimensions. If it is None, then `min_coords`
        is meant to represent `max_coords` and the minimum coordinate values are meant to be 0.
    force_valid : bool
        whether or not to sort out `min_coords` and `max_coords` to make the the minus point meet
        the min_coords and the plus point meet the max_coords.

    Attributes
    ----------
    dlt_tfm : Dlt
        the dilatation equivalent, which can be get/set
    dim : int
        number of dimensions
    minus_pt : point
        the (-1,)^n point after being transformed by the dilatation
    plus_pt : point
        the (+1,)^n point after being transformed by the dilatation
    min_coords : point
        minimum coordinates
    max_coords : point
        maximum coordinates
    center_pt : point
        center point
    size : size/point
        box size
    signed_volume : float
        the product of `max_coords - min_coords`
    volume : float
        the absolute value of `signed_volume`

    See Also
    --------
    Dlt : the dilatation behind the box
    mt.geo3d.Box : the 3D box, with named coordinates
    mt.geo.join_volume : volumes of the intersection and union of two boxes

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geond import Hyperbox
    >>> b = Hyperbox(np.array([0., 0.]), np.array([2., 4.]))
    >>> b
    Hyperbox(Dlt(offset=[1. 2.], scale=[1. 2.]))
    >>> b.min_coords, b.max_coords, b.center_pt
    (array([0., 0.]), array([2., 4.]), array([1., 2.]))
    >>> b.dim, b.volume
    (2, 8.0)
    >>> c = Hyperbox(np.array([1., 2.]), np.array([5., 3.]))
    >>> b.intersect(c)
    Hyperbox(Dlt(offset=[1.5 2.5], scale=[0.5 0.5]))
    >>> b.intersect(c).volume
    1.0
    >>> b.box_union(c).max_coords
    array([5., 4.])
    >>> round(b.iou(c), 4)
    0.0909
    >>> from mt.geo import join_volume
    >>> join_volume(b, c)
    (1.0, 7.0, 3.0, 11.0)
    >>> b.intersect(Hyperbox(np.array([5., 5.]), np.array([6., 6.]))).volume
    0.0
    >>> bad = Hyperbox(np.array([2., 4.]), np.array([0., 0.]))
    >>> bad.is_valid()
    False
    >>> Hyperbox(np.array([2., 4.]), np.array([0., 0.]), force_valid=True).is_valid()
    True
    """

    # ----- data encapsulation -----

    @property
    def dlt_tfm(self):
        """the dilatation (:class:`Dlt`) mapping the hypercube [-1,1]^n to the box, which can be
        get and set"""
        return self.__dlt_tfm

    @dlt_tfm.setter
    def dlt_tfm(self, dlt_tfm):
        self.__dlt_tfm = dlt_tfm

    # ----- derived properties -----

    @property
    def ndim(self):
        """The dimensionality, same as `dim`."""
        return self.dim

    @property
    def dim(self):
        """The dimensionality, i.e. the number of dimensions of the dilatation."""
        return self.dlt_tfm.dim

    @property
    def minus_pt(self):
        """The (-1,)^n point after being transformed by the dilatation, i.e. `offset - scale`."""
        return self.dlt_tfm.offset - self.dlt_tfm.scale

    @property
    def plus_pt(self):
        """The (+1,)^n point after being transformed by the dilatation, i.e. `offset + scale`."""
        return self.dlt_tfm.offset + self.dlt_tfm.scale

    @property
    def min_coords(self):
        """minimum coordinates, the element-wise minimum of `minus_pt` and `plus_pt`"""
        return np.minimum(self.minus_pt, self.plus_pt)

    @property
    def max_coords(self):
        """maximum coordinates, the element-wise maximum of `minus_pt` and `plus_pt`"""
        return np.maximum(self.minus_pt, self.plus_pt)

    @property
    def center_pt(self):
        """center point, which is the offset of the dilatation"""
        return self.dlt_tfm.offset

    @property
    def size(self):
        """box size, `abs(2 * scale)`, the side lengths of the box"""
        return abs(self.dlt_tfm.scale * 2)

    @property
    def signed_volume(self):
        """signed (hyper-)volume, `prod(max_coords - min_coords)`"""
        return (self.max_coords - self.min_coords).prod()

    @property
    def volume(self):
        """absolute/unsigned (hyper-)volume, `abs(signed_volume)`"""
        return abs(self.signed_volume)

    # ----- methods -----

    def __init__(self, min_coords, max_coords=None, force_valid=False):
        if isinstance(min_coords, Dlt):
            self.dlt_tfm = min_coords
        else:
            if max_coords is None:
                max_coords = min_coords
                min_coords = np.zeros_like(max_coords)
            self.dlt_tfm = Dlt(
                offset=(max_coords + min_coords) / 2,
                scale=(max_coords - min_coords) / 2,
            )

        if force_valid:
            min_coords = self.min_coords
            max_coords = self.max_coords
            self.dlt_tfm = Dlt(
                offset=(max_coords + min_coords) / 2,
                scale=(max_coords - min_coords) / 2,
            )

    def __repr__(self):
        return f"Hyperbox({self.dlt_tfm})"

    def is_valid(self):
        return (self.dlt_tfm.scale >= 0).all()

    def validated(self):
        """Returns a validated version of the box.

        Returns
        -------
        Hyperbox
            a box with the same extent as the original, but whose dilatation scale is
            non-negative
        """
        min_coords = self.min_coords
        max_coords = self.max_coords
        return Hyperbox(
            Dlt(offset=(max_coords + min_coords) / 2, scale=(max_coords - min_coords) / 2)
        )

    def intersect(self, other):
        """Returns the box with the intersection of the coordinate ranges.

        If the two boxes are disjoint along some axis, the intersection is empty: the returned
        box is valid and has zero extent (hence zero volume) along that axis.
        """
        min_coords = np.maximum(self.min_coords, other.min_coords)
        max_coords = np.maximum(np.minimum(self.max_coords, other.max_coords), min_coords)
        return Hyperbox(min_coords, max_coords)

    def box_union(self, other):
        return Hyperbox(
            np.minimum(self.min_coords, other.min_coords),
            np.maximum(self.max_coords, other.max_coords),
        )

    def iou(self, other, epsilon=1e-7):
        return iou_impl(
            self.intersect(other).volume, self.volume, other.volume, eps=epsilon
        )


# ----- joining volumes -----


def join_volume_Hyperbox_and_Hyperbox(obj1, obj2):
    """Joins the volumes of two Hyperboxes of the same dimension.

    It is the function registered with :func:`mt.geo.register_join_volume` for a pair of
    :class:`Hyperbox`, and it is used by :func:`mt.geo.join_volume`.

    Parameters
    ----------
    obj1 : Hyperbox
        the first hyperbox
    obj2 : Hyperbox
        the second hyperbox

    Returns
    -------
    intersection_volume : float
        the volume of the intersection of the two hyperboxes' interior regions
    obj1_only_volume : float
        the volume of the interior of obj1 that does not belong to obj2
    obj2_only_volume : float
        the volume of the interior of obj2 that does not belong to obj1
    union_volume : float
        the volume of the union of the two hyperboxes' interior regions

    Raises
    ------
    ValueError
        if the two hyperboxes do not have the same dimension
    """
    if obj1.dim != obj2.dim:
        raise ValueError(
            f"Two objects of dimension {obj1.dim} and {obj2.dim} are not in the same space."
        )
    inter = obj1.intersect(obj2)
    inter_volume = inter.volume
    obj1_volume = obj1.volume
    obj2_volume = obj2.volume
    return (
        inter_volume,
        obj1_volume - inter_volume,
        obj2_volume - inter_volume,
        obj1_volume + obj2_volume - inter_volume,
    )


register_join_volume(Hyperbox, Hyperbox, join_volume_Hyperbox_and_Hyperbox)
