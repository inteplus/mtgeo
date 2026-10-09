"""A 2D polygon."""

from mt import np
import mt.base.casting as _bc
from ..geo import register_join_volume
from ..geond import castable_ndarray_PointList
from .point_list import PointList2d
from .shapely import HasShapely, join_volume_shapely
from .rect import Rect


__all__ = ["Polygon"]


class Polygon(HasShapely, PointList2d):
    """A 2D polygon, represented as a point list of vertices in either clockwise or
    counter-clockwise order.

    The polygon is implicitly closed: the last vertex connects back to the first one. The
    constructor is the one of :class:`~mt.geo2d.PointList2d`, including its `check` argument.

    Examples
    --------
    >>> from mt.geo2d import Polygon
    >>> p = Polygon([[0, 0], [4, 0], [4, 3]])
    >>> p.points
    array([[0., 0.],
           [4., 0.],
           [4., 3.]], dtype=float32)
    >>> round(p.shapely.area, 2)  # shapely area is slightly inflated by a 0.0001 buffer
    6.0

    See Also
    --------
    PointList2d
        base class
    """

    # ----- internal representations -----

    @property
    def shapely(self):
        """Shapely representation for fast intersection operations.

        It is a shapely polygon of the vertices, buffered by 0.0001 to turn any non-simple polygon
        into a simple one. The result is cached in the instance.
        """
        if not hasattr(self, "_shapely"):
            import shapely.geometry as _sg

            self._shapely = _sg.Polygon(self.points)
            self._shapely = self._shapely.buffer(
                0.0001
            )  # to clean up any (multi and/or non-simple) polygon into a simple polygon
        return self._shapely


_bc.register_castable(np.ndarray, Polygon, lambda x: castable_ndarray_PointList(x, 2))
_bc.register_cast(np.ndarray, Polygon, lambda x: Polygon(x, check=False))

_bc.register_cast(PointList2d, Polygon, lambda x: Polygon(x.points, check=False))


# ----- joining volumes -----


register_join_volume(Rect, Polygon, join_volume_shapely)
register_join_volume(Polygon, Rect, join_volume_shapely)
register_join_volume(Polygon, Polygon, join_volume_shapely)
