'''The base class to represent a point. 

For efficiency reasons, please try to bunch points into arrays or lists and use appropriate representations instead of using single points implemented here.
'''


import numpy as _np
import mt.base.casting as _bc
from ..geo import ThreeD
from ..geond import Point, castable_ndarray_Point


__all__ = ['Point3d']


class Point3d(ThreeD, Point):
    '''A 3D point. See Point for more details.'''
    pass
_bc.register_castable(_np.ndarray, Point3d, lambda x: castable_ndarray_Point(x, 3))
_bc.register_cast(_np.ndarray, Point3d, lambda x: Point3d(x, check=False))
_bc.register_cast(Point3d, Point, lambda x: Point(x.point, check=False))
_bc.register_cast(Point, Point3d, lambda x: Point3d(x.point, check=False))
_bc.register_castable(Point, Point3d, lambda x: x.ndim==3)
