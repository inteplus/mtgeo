"""The base class to represent a point.

For efficiency reasons, please try to bunch points into arrays or lists and use appropriate
representations instead of using single points implemented here.

The module provides :class:`Point`, which is dimension-agnostic and expects its subclasses to say
what `ndim` is (for example :class:`mt.geo3d.Point3d`), and the helper `castable_ndarray_Point`
that tells whether a numpy array has the shape of a single point of a given dimension.
"""


import numpy as _np
import mt.base.casting as _bc
from ..geo import GeometricObject


__all__ = ["Point", "castable_ndarray_Point"]


def castable_ndarray_Point(obj, ndim):
    return len(obj.shape) == 1 and obj.shape[0] == ndim


class Point(GeometricObject):
    """A point.

    The base class stores the coordinates as a 1D numpy array in the attribute `point`. It does not
    define `ndim` itself, so it can only be instantiated directly with `check=False`; with the
    default `check=True` the shape check calls `self.ndim`, which raises `NotImplementedError` in
    the base class. Subclasses such as :class:`mt.geo3d.Point3d` fix `ndim` via a mixin
    (:class:`mt.geo.TwoD` or :class:`mt.geo.ThreeD`).

    Parameters
    ----------
    point : iterable
        A list of point coordinates, which is an iterable of D items, where D is the `ndim` of the
        class.
    check : bool
        Whether or not to check if the shape is valid. If True and the number of coordinates is
        not `ndim`, a `ValueError` is raised.

    Attributes
    ----------
    point : `numpy.ndarray(shape=(D,))`
        The point in numpy.

    See Also
    --------
    mt.geo3d.Point3d : the 3D point, stored in a glm vector
    PointList : a list of points

    Examples
    --------
    >>> from mt.geond import Point
    >>> Point([1, 2], check=False).point
    array([1, 2])
    """

    def __init__(self, point, check=True):
        point = point if isinstance(point, _np.ndarray) else _np.array(point)
        if check and not castable_ndarray_Point(point, self.ndim):
            raise ValueError(
                f"Point in {self.ndim}D not in the right length: {len(point)}"
            )
        self.point = point
