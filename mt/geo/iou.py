'''Intersection over Union. In the current primitive form, it is treated as a separate module.'''


import shapely.geometry as _sg # we need shapely to intersect _sg.Polygons and _sg.boxes
from .rect import Rect
from .polygon import Polygon


def iou(geo2d_obj1, geo2d_obj2):
    '''Computes the Intersection-over-Union ratio of two 2D geometry objects. Right now we only accept Rect and Polygon.

    Parameters
    ----------
    geo2d_obj1 : Rect or Polygon
        the first 2D geometry object
    geo2d_obj2 : Rect or Polygon
        the second 2D geometry object

    Returns
    -------
    float
        the IoU ratio of the two objects, regardless of whether they are sepcified in clockwise or counter-clockwise order
    '''
    
    # convert the first object into shapely format
    if isinstance(geo2d_obj1, Rect):
        obj1 = _sg.box(geo2d_obj1.min_x, geo2d_obj1.min_y, geo2d_obj1.max_x, geo2d_obj1.max_y)
    elif isinstance(geo2d_obj1, Polygon):
        obj1 = _sg.Polygon(geo2d_obj1.points)
    else:
        raise ValueError("The first object is neither a Rect nor a Polygon. Got '{}'.".format(type(geo2d_obj1)))

    # convert the second object into shapely format
    if isinstance(geo2d_obj2, Rect):
        obj2 = _sg.box(geo2d_obj2.min_x, geo2d_obj2.min_y, geo2d_obj2.max_x, geo2d_obj2.max_y)
    elif isinstance(geo2d_obj2, Polygon):
        obj2 = _sg.Polygon(geo2d_obj2.points)
    else:
        raise ValueError("The second object is neither a Rect nor a Polygon. Got '{}'.".format(type(geo2d_obj2)))

    o1 = obj1.area
    o2 = obj2.area
    ii = obj1.intersection(obj2).area

    return 0.0 if abs(ii) < 1E-7 else ii/(o1+o2-ii)
