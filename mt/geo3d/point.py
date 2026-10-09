"""The class to represent a 3D point.

For efficiency reasons, please try to bunch points into arrays or lists and use appropriate
representations instead of using single points implemented here.

The module provides :class:`Point3d`, a 3D point stored as a `glm.vec3`, and registers casts between
numpy arrays of shape `(3,)`, :class:`Point3d` and :class:`mt.geond.Point` (see
:func:`mt.base.casting.cast`).
"""


from mt import np, glm
import mt.base.casting as _bc
from ..geo import ThreeD
from ..geond import Point, castable_ndarray_Point


__all__ = ["Point3d"]


class Point3d(ThreeD, Point):
    """A 3D point implemented in glm.

    The coordinates are stored in single precision (float32) in a `glm.vec3`. See :class:`Point`
    for more details.

    Parameters
    ----------
    point : iterable
        the 3 coordinates `(x, y, z)` of the point, e.g. a list or a numpy array. .
    check : bool
        Whether or not to check that the point has exactly 3 coordinates, raising a `ValueError`
        otherwise. If False, the check is left to glm, which raises a `TypeError`.

    Attributes
    ----------
    point_glm : `glm.vec3`
        The point in glm.
    point : `numpy.ndarray(shape=(3,), dtype=numpy.float32)`
        The numpy copy of `point_glm`. Assigning to it updates `point_glm`.

    See Also
    --------
    mt.geond.Point : the dimension-agnostic base class
    PointList3d : a list of 3D points

    Examples
    --------
    >>> from mt.geo3d import Point3d
    >>> p = Point3d([1, 2, 3])
    >>> p.ndim
    3
    >>> p.point
    array([1., 2., 3.], dtype=float32)
    >>> p.point_glm
    vec3( 1, 2, 3 )
    >>> p.point = [4, 5, 6]
    >>> p.point_glm
    vec3( 4, 5, 6 )
    """

    def __init__(self, point, check=True):
        if check:
            arr = np.asarray(point)
            if not castable_ndarray_Point(arr, 3):
                raise ValueError(f"Point in 3D not in the right length: {arr.shape}")
        self.point = point

    @property
    def point(self):
        return np.array(self.point_glm)

    @point.setter
    def point(self, p):
        self.point_glm = glm.vec3(p)


_bc.register_castable(np.ndarray, Point3d, lambda x: castable_ndarray_Point(x, 3))
_bc.register_cast(np.ndarray, Point3d, lambda x: Point3d(x, check=False))
_bc.register_cast(Point3d, Point, lambda x: Point(x.point, check=False))
_bc.register_cast(Point, Point3d, lambda x: Point3d(x.point, check=False))
_bc.register_castable(Point, Point3d, lambda x: x.ndim == 3)
