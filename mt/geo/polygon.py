'''A 2D polygon.'''


import numpy as _np
import mt.base.casting as _bc
from .point_list import PointList2d, castable_ndarray_PointList
from .polygon_integral import * # for now assume that you want to do polygon integration whenever polygon is imported

import shapely.geometry as _sg # we need shapely to intersect _sg.Polygons and _sg.boxes
from .rect import Rect
from .join_volume import *


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


# ----- joining volumes -----


def join_volume_Polygon_Rect(obj1, obj2):
    '''Joins the areas of two objects of types Polygons or Rect.

    Parameters
    ----------
    obj1 : Rect or Polygon
        the first 2D geometry object
    obj2 : Rect or Polygon
        the second 2D geometry object

    Returns
    -------
    intersection_area : float
        the area of the intersection of the two objects' interior regions
    obj1_only_area : float
        the area of the interior of obj1 that does not belong to obj2
    obj2_only_area : float
        the area of the interior of obj2 that does not belong to obj1
    union_area : float
        the area of the union of the two objects' interior regions
    '''
    if isinstance(obj1, Rect):
        out_obj1 = _sg.box(obj1.min_x, obj1.min_y, obj1.max_x, obj1.max_y)
    elif isinstance(obj1, Polygon):
        out_obj1 = _sg.Polygon(obj1.points).buffer(0) # to clean up
    else:
        raise ValueError("The first object is neither a Rect nor a Polygon. Got '{}'.".format(type(obj1)))

    if isinstance(obj2, Rect):
        out_obj2 = _sg.box(obj2.min_x, obj2.min_y, obj2.max_x, obj2.max_y)
    elif isinstance(obj2, Polygon):
        out_obj2 = _sg.Polygon(obj2.points).buffer(0) # to clean up
    else:
        raise ValueError("The second object is neither a Rect nor a Polygon. Got '{}'.".format(type(obj2)))

    inter_area = obj1.intersection(obj2).area
    obj1_area = obj1.area
    obj2_area = obj2.area
    return (inter_area, obj1_area - inter_area, obj2_area - inter_area, obj1_area + obj2_area - inter_area)
register_join_volume(Rect, Polygon, join_volume_Polygon_Rect)
register_join_volume(Polygon, Rect, join_volume_Polygon_Rect)
register_join_volume(Polygon, Polygon, join_volume_Polygon_Rect)
