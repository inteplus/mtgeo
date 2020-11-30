'''Intersection over Union. In the current primitive form, it is treated as a separate module.'''


import shapely.geometry as _sg # we need shapely to intersect _sg.Polygons and _sg.boxes
from .rect import Rect
from .polygon import Polygon


def iou(geo2d_obj1, geo2d_obj2):
    '''Computes the Intersection-over-Union ratio of two (sets of non-overlapping) 2D geometry objects. Right now we only accept Rect and Polygon.

    Parameters
    ----------
    geo2d_obj1 : Rect or Polygon or list of non-overlapping geometric objects of these types
        the first 2D geometry object
    geo2d_obj2 : Rect or Polygon or list of non-overlapping geometric objects of these types
        the second 2D geometry object

    Returns
    -------
    float
        the IoU ratio of the two objects, regardless of whether they are sepcified in clockwise or counter-clockwise order
    '''
    # make sure each object is a list
    inputs = [obj if isinstance(obj, list) else [obj] for obj in [geo2d_obj1, geo2d_obj2]]

    # convert each item into shapely format
    outputs = []
    for i, group in enumerate(inputs):
        out_group = []
        for j, obj in enumerate(group):
            if isinstance(obj, Rect):
                out_obj = _sg.box(obj.min_x, obj.min_y, obj.max_x, obj.max_y)
            elif isinstance(obj, Polygon):
                out_obj = _sg.Polygon(obj.points).buffer(0) # to clean up
            else:
                raise ValueError("The {}-th object of the {} argument is neither a Rect nor a Polygon. Got '{}'.".format(j+1, 'first' if i==0 else 'second', type(obj)))
            out_group.append(out_obj)
        outputs.append(out_group)

    # compute moments
    sum_left = 0
    sum_right = 0
    sum_inner = 0
    for obj1 in outputs[0]:
        for obj2 in outputs[1]:
            sum_left += obj1.area
            sum_right += obj2.area
            sum_inner += obj1.intersection(obj2).area

    # result
    return 0.0 if abs(sum_inner) < 1E-7 else sum_inner/(sum_left + sum_right - sum_inner)
