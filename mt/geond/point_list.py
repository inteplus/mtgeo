"""The base class to represent a list of points.

The module provides :class:`PointList`, which is dimension-agnostic and expects its subclasses to
say what `ndim` is (for example :class:`mt.geo3d.PointList3d`), and the helper
`castable_ndarray_PointList` that tells whether a numpy array has the shape of a list of points of
a given dimension.
"""


import numpy as _np
from ..geo import GeometricObject


__all__ = ["PointList", "castable_ndarray_PointList"]


def castable_ndarray_PointList(obj, ndim):
    if obj.size == 0:
        return True
    return len(obj.shape) == 2 and obj.shape[1] == ndim


class PointList(GeometricObject):
    """A list of points.

    The points are stored as a 2D numpy array of shape `(N, D)` in the attribute `points`, where an
    empty list is reshaped to `(0, D)`. The base class does not define `ndim`, so it can only be
    instantiated directly with `check=False`; with the default `check=True` the shape check calls
    `self.ndim`, which raises `NotImplementedError` in the base class. Subclasses such as
    :class:`mt.geo3d.PointList3d` fix `ndim` via a mixin.

    Parameters
    ----------
    point_list : list or numpy.ndarray
        A list of points, each of which is an iterable of D items, where D is the `ndim` of the
        class.
    check : bool
        Whether or not to check if the shape is valid. If True and the shape is not `(N, ndim)`
        (an empty list is always accepted), a `ValueError` is raised.

    Attributes
    ----------
    points : `numpy.ndarray(shape=(N,D))`
        The list of points in numpy.

    See Also
    --------
    Point : a single point
    mt.geo3d.PointList3d : the 3D point list, stored in a glm array

    Examples
    --------
    >>> from mt.geond import PointList
    >>> PointList([[1, 2], [3, 4], [5, 6]], check=False).points
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """

    def __init__(self, point_list, check=True):
        points = (
            point_list if isinstance(point_list, _np.ndarray) else _np.array(point_list)
        )
        if check and not castable_ndarray_PointList(points, self.ndim):
            raise ValueError(
                f"Point list in {self.ndim}D not in the right shape: {points.shape}"
            )
        if points.size == 0:
            points = points.reshape((0, self.ndim))
        self.points = points
