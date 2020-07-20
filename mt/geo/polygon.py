'''A 2D polygon.'''


import numpy as _np
import mt.base.cast as _bc
from .point_list import PointList2d, castable_ndarray_PointList


__all__ = ['Polygon']


class Polygon(PointList2d):
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
    pass


_bc.register_castable(_np.ndarray, Polygon, lambda x: castable_ndarray_PointList(x,2))
_bc.register_cast(_np.ndarray, Polygon, lambda x: Polygon(x, check=False))

_bc.register_cast(PointList2d, Polygon, lambda x: Polygon(x.points, check=False))
