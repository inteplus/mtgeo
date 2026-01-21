"""Support for shapely."""

__all__ = ["HasShapely", "join_volume_shapely"]


class HasShapely(object):
    """Mixin to assert that the geometric object has the 'shapely' property."""

    @property
    def shapely(self):
        """Shapely representation for fast intersection operations. Noe that shapely treats
        signed and unsigned regions equally."""
        raise NotImplementedError("Implement me.")


# ----- joining volumes -----


def join_volume_shapely(
    obj1: HasShapely, obj2: HasShapely, intersection_grid_size=None
):
    """Joins the areas of two 2D geometry objects supporting shapely.

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
