'''A 2D polygon.'''

from mt import np
import mt.base.casting as _bc
from ..geo import register_join_volume
from ..geond import castable_ndarray_PointList
from .point_list import PointList2d
from .shapely import HasShapely, join_volume_shapely
from .rect import Rect
from .rrect import RRect


__all__ = ['Polygon']


class Polygon(HasShapely, PointList2d):
    '''A 2D polygon, represented as a point list of vertices in either clockwise or counter-clockwise order.

    Parameters
    ----------
    point_list : list
        A list of 2D points, each of which is an iterable of 2 items.
    check : bool
        Whether or not to check if the shape is valid

    Attributes
    ----------
    points : `numpy.ndarray(shape=(N,2))`
        The list of 2D vertices in numpy.
    '''

    # ----- internal representations -----

    @property
    def shapely(self):
        '''Shapely representation for fast intersection operations.'''
        if not hasattr(self, '_shapely'):
            import shapely.geometry as _sg
            self._shapely = _sg.Polygon(self.points)
            self._shapely = self._shapely.buffer(0.0001) # to clean up any (multi and/or non-simple) polygon into a simple polygon
        return self._shapely


_bc.register_castable(np.ndarray, Polygon, lambda x: castable_ndarray_PointList(x,2))
_bc.register_cast(np.ndarray, Polygon, lambda x: Polygon(x, check=False))

_bc.register_cast(PointList2d, Polygon, lambda x: Polygon(x.points, check=False))


# ----- joining volumes -----


register_join_volume(Rect, Polygon, join_volume_shapely)
register_join_volume(Polygon, Rect, join_volume_shapely)
register_join_volume(RRect, Polygon, join_volume_shapely)
register_join_volume(Polygon, RRect, join_volume_shapely)
register_join_volume(Polygon, Polygon, join_volume_shapely)
