'''Intersection over Union. In the current primitive form, it is treated as a separate module.

See :func:`iou`.
'''


from ..geond.iou import iou_impl


__all__ = ['iou']


def iou(geo2d_obj1, geo2d_obj2):
    '''Computes the Intersection-over-Union ratio of two (sets of non-overlapping) 2D geometry
    objects. Right now we only accept Rect and Polygon (any object with a `shapely` property, such
    as RRect, also works).

    The areas are computed with shapely, whose representations are slightly buffered (by 0.0001),
    so the result is accurate up to a small tolerance. Only `list` is recognised as a collection,
    not tuples. For lists, each list is treated as the union of its (non-overlapping) objects: the
    areas are summed per list and the intersection area is summed over every pair of objects (one
    from each list).

    Parameters
    ----------
    geo2d_obj1 : Rect or Polygon or list of non-overlapping geometric objects of these types
        the first 2D geometry object
    geo2d_obj2 : Rect or Polygon or list of non-overlapping geometric objects of these types
        the second 2D geometry object

    Returns
    -------
    float
        the IoU ratio of the two objects, regardless of whether they are specified in clockwise or
        counter-clockwise order

    Examples
    --------
    >>> from mt.geo2d import Rect, Polygon, iou
    >>> round(iou(Rect(0, 0, 2, 2), Rect(1, 0, 3, 2)), 3)
    0.333
    >>> round(iou(Rect(0, 0, 2, 2), Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])), 3)
    1.0
    >>> a = Rect(0, 0, 1, 1)
    >>> round(iou([a], [a]), 3)
    1.0
    >>> round(iou([a, Rect(2, 0, 3, 1)], [a]), 3)  # union of two unit squares vs one of them
    0.5

    See Also
    --------
    mt.geo2d.Rect.iou : a method for Rect against Rect
    '''
    # make sure each object is a list
    inputs = [obj if isinstance(obj, list) else [obj] for obj in [geo2d_obj1, geo2d_obj2]]

    # compute moments
    sum_left = 0
    sum_right = 0
    sum_inner = 0
    for obj1 in inputs[0]:
        sum_left += obj1.shapely.area
    for obj2 in inputs[1]:
        sum_right += obj2.shapely.area
    for obj1 in inputs[0]:
        for obj2 in inputs[1]:
            sum_inner += obj1.shapely.intersection(obj2.shapely).area

    # result
    return iou_impl(sum_inner, sum_left, sum_right)
