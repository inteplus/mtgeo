"""Support for shapely: a mixin for geometry objects convertible to shapely, and area joining.

Note that the `shapely` property of :class:`~mt.geo2d.Rect`, :class:`~mt.geo2d.Polygon` and
:class:`~mt.geo2d.RRect` is buffered by 0.0001 to clean up invalid polygons, so the areas computed
from it are slightly larger than the exact areas.
"""

__all__ = ["HasShapely", "join_volume_shapely"]


class HasShapely(object):
    """Mixin to assert that the geometric object has the 'shapely' property.

    Subclasses must implement the property `shapely`, returning a shapely geometry. See
    :class:`~mt.geo2d.Rect`, :class:`~mt.geo2d.Polygon` and :class:`~mt.geo2d.RRect`.
    """

    @property
    def shapely(self):
        """Shapely representation for fast intersection operations. Note that shapely treats
        signed and unsigned regions equally.

        Raises
        ------
        NotImplementedError
            always, in this base class. Subclasses must override it.
        """
        raise NotImplementedError("Implement me.")


# ----- joining volumes -----


def join_volume_shapely(
    obj1: HasShapely, obj2: HasShapely, intersection_grid_size=None
):
    """Joins the areas of two 2D geometry objects supporting shapely.

    This function is registered with :func:`mt.geo.join_volume` for pairs of Rect, Polygon and
    RRect.

    Parameters
    ----------
    obj1 : HasShapely
        the first 2D geometry object
    obj2 : HasShapely
        the second 2D geometry object
    intersection_grid_size : float or None
        grid size for approximating intersection area. If None, exact computation is used.

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

    Examples
    --------
    >>> from mt.geo2d import Rect
    >>> from mt.geo2d.shapely import join_volume_shapely
    >>> inter, only1, only2, union = join_volume_shapely(Rect(0, 0, 2, 2), Rect(1, 1, 3, 3))
    >>> round(inter, 2), round(only1, 2), round(only2, 2), round(union, 2)
    (1.0, 3.0, 3.0, 7.0)
    """

    inter_area = obj1.shapely.intersection(
        obj2.shapely, grid_size=intersection_grid_size
    ).area
    obj1_area = obj1.shapely.area
    obj2_area = obj2.shapely.area
    return (
        inter_area,
        obj1_area - inter_area,
        obj2_area - inter_area,
        obj1_area + obj2_area - inter_area,
    )
