'''The base class to represent a list of points.'''


import numpy as np
import mt.base.casting as _bc
from ..geo import TwoD
from ..geond import PointList, castable_ndarray_PointList


__all__ = ['PointList2d']


class PointList2d(TwoD, PointList):
    '''A list of 2D points. See PointList for more details.'''
    pass
_bc.register_castable(np.ndarray, PointList2d, lambda x: castable_ndarray_PointList(x,2))
_bc.register_cast(np.ndarray, PointList2d, lambda x: PointList2d(x, check=False))
_bc.register_cast(PointList2d, PointList, lambda x: PointList(x.points, check=False))
_bc.register_cast(PointList, PointList2d, lambda x: PointList2d(x.points, check=False))
_bc.register_castable(PointList, PointList2d, lambda x: x.ndim==2)
