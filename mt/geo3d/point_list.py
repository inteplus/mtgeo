'''The base class to represent a list of points.'''


import numpy as np
import mt.base.casting as _bc
from ..geo import ThreeD
from ..geond import PointList, castable_ndarray_PointList


__all__ = ['PointList3d']


class PointList3d(ThreeD, PointList):
    '''A list of 3D points. See PointList for more details.'''
    pass
_bc.register_castable(np.ndarray, PointList3d, lambda x: castable_ndarray_PointList(x, 3))
_bc.register_cast(np.ndarray, PointList3d, lambda x: PointList3d(x, check=False))
_bc.register_cast(PointList3d, PointList, lambda x: PointList(x.points, check=False))
_bc.register_cast(PointList, PointList3d, lambda x: PointList3d(x.points, check=False))
_bc.register_castable(PointList, PointList3d, lambda x: x.ndim==3)
