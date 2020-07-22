'''The base class to represent a point. 

For efficiency reasons, please try to bunch points into arrays or lists and use appropriate representations instead of using single points implemented here.
'''


import numpy as _np
import mt.base.casting as _bc
from .object import GeometricObject, TwoD, ThreeD


__all__ = ['Point', 'Point2d', 'Point3d', 'castable_ndarray_Point']


def castable_ndarray_Point(obj, ndim):
    return len(obj.shape) == 1 and obj.shape[0] == ndim


class Point(GeometricObject):
    '''A point.

    Parameters
    ----------
    point : iterable
        A list of point coordinates, which is an iterable of D items, where D is the `ndim` of the class.
    check : bool
        Whether or not to check if the shape is valid

    Attributes
    ----------
    point : `numpy.ndarray(shape=(D,))`
        The point in numpy.
    '''

    def __init__(self, point, check=True):
        point = point if isinstance(point, _np.ndarray) else _np.array(point)
        if check and not castable_ndarray_Point(point, self.ndim):
            raise ValueError("Point in {}D not in the right length: {}".format(self.ndim, len(point)))
        self.point = point


class Point2d(TwoD, Point):
    '''A 2D point. See Point for more details.'''
    pass
_bc.register_castable(_np.ndarray, Point2d, lambda x: castable_ndarray_Point(x,2))
_bc.register_cast(_np.ndarray, Point2d, lambda x: Point2d(x, check=False))
_bc.register_cast(Point2d, Point, lambda x: Point(x.point, check=False))
_bc.register_cast(Point, Point2d, lambda x: Point2d(x.point, check=False))
_bc.register_castable(Point, Point2d, lambda x: x.ndim==2)


class Point3d(ThreeD, Point):
    '''A 3D point. See Point for more details.'''
    pass
_bc.register_castable(_np.ndarray, Point3d, lambda x: castable_ndarray_Point(x, 3))
_bc.register_cast(_np.ndarray, Point3d, lambda x: Point3d(x, check=False))
_bc.register_cast(Point3d, Point, lambda x: Point(x.point, check=False))
_bc.register_cast(Point, Point3d, lambda x: Point3d(x.point, check=False))
_bc.register_castable(Point, Point3d, lambda x: x.ndim==3)
