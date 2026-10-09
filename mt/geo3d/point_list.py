"""The class to represent a list of 3D points.

The module provides :class:`PointList3d`, a list of 3D points stored as a `glm.array` of `glm.vec3`.
"""

from mt import tp, np, glm
import mt.base.casting as bc
from ..geo import ThreeD
from ..geond import PointList, castable_ndarray_PointList


__all__ = ["PointList3d"]


class PointList3d(ThreeD, PointList):
    """A list of 3D points.

    The coordinates are stored in single precision (float32) in a `glm.array` of `glm.vec3`. See
    :class:`PointList` for more details.

    Parameters
    ----------
    point_list : list or numpy.ndarray or glm.array
        A list of points, each of which is an iterable of 3 items. An empty list gives an empty
        array. A `glm.array` of `glm.vec3` is used as it is, and a `glm.array` of another type is
        reinterpreted as `glm.vec3`.

    Attributes
    ----------
    points_glm : glm.array
        The array of vec3 points
    points : `numpy.ndarray(shape=(N,3), dtype=numpy.float32)`
        The numpy view of `points_glm`. Assigning to it updates `points_glm`.

    See Also
    --------
    mt.geond.PointList : the dimension-agnostic base class
    Point3d : a single 3D point

    Notes
    -----
    The constructor has no `check` argument, unlike :class:`PointList`. The registered casts and
    the affine transformation of a point list call the constructor with `check=False` and
    therefore currently raise `TypeError`.

    Examples
    --------
    >>> from mt.geo3d import PointList3d
    >>> pl = PointList3d([[1, 2, 3], [4, 5, 6]])
    >>> pl.ndim
    3
    >>> pl.points
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)
    >>> PointList3d([]).points.shape
    (0, 3)
    """

    def __init__(self, point_list: tp.Union[list, np.ndarray, glm.array]):
        self.points = point_list

    @property
    def points(self):
        p = np.frombuffer(self.points_glm.to_bytes(), dtype=np.float32)
        return p.reshape((len(p) // 3, 3))

    @points.setter
    def points(self, point_list: tp.Union[list, np.ndarray, glm.array]):
        if len(point_list) == 0:
            self.points_glm = glm.array.zeros(0, glm.vec3)
        elif isinstance(point_list, glm.array):
            self.points_glm = point_list
            if not isinstance(self.points_glm[0], glm.vec3):
                self.points_glm = self.points_glm.reinterpret_cast(glm.vec3)
        else:
            self.points_glm = glm.array([glm.vec3(x) for x in point_list])


bc.register_castable(
    np.ndarray, PointList3d, lambda x: castable_ndarray_PointList(x, 3)
)
bc.register_cast(np.ndarray, PointList3d, lambda x: PointList3d(x, check=False))
bc.register_cast(PointList3d, PointList, lambda x: PointList(x.points, check=False))
bc.register_cast(PointList, PointList3d, lambda x: PointList3d(x.points, check=False))
bc.register_castable(PointList, PointList3d, lambda x: x.ndim == 3)
