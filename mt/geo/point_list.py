'''The base class to represent a list of points.'''


import numpy as _np
import mt.base.cast as _bc
from .object import GeometricObject, TwoD, ThreeD


__all__ = ['PointList', 'PointList2d', 'PointList3d', 'castable_ndarray_PointList']


def castable_ndarray_PointList(obj, ndim):
    if obj.size == 0:
        return True
    return len(obj.shape) == 2 and obj.shape[1] == ndim


class PointList(GeometricObject):
    '''A list of points.

    Parameters
    ----------
    point_list : list
        A list of points, each of which is an iterable of D items, where D is the `ndim` of the class.
    check : bool
        Whether or not to check if the shape is valid

    Attributes
    ----------
    points : `numpy.ndarray(shape=(N,D))`
        The list of points in numpy.
    '''

    def __init__(self, point_list, check=True):
        points = point_list if isinstance(point_list, _np.ndarray) else _np.array(point_list)
        if check and not castable_ndarray_PointList(points, self.ndim):
            raise ValueError("Point list in {}D not in the right shape: {}".format(self.ndim, points.shape))
        if points.size == 0:
            points = points.reshape((0,self.ndim))
        self.points = points


class PointList2d(TwoD, PointList):
    '''A list of 2D points. See PointList for more details.'''
    pass
_bc.register_castable(_np.ndarray, PointList2d, lambda x: castable_ndarray_PointList(x,2))
_bc.register_cast(_np.ndarray, PointList2d, lambda x: PointList2d(x, check=False))


class PointList3d(ThreeD, PointList):
    '''A list of 3D points. See PointList for more details.'''
    pass
_bc.register_castable(_np.ndarray, PointList3d, lambda x: castable_ndarray_PointList(x, 3))
_bc.register_cast(_np.ndarray, PointList3d, lambda x: PointList3d(x, check=False))
