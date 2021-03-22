'''Raw moments up to 2nd order of 2D points.'''

from mt import np
import sys as _sys

import mt.base.casting as _bc

from ..geo import TwoD
from ..geond import Moments, moments_from_pointlist
from .point_list import PointList2d


__all__ = ['Moments2d']


class Moments2d(TwoD, Moments):
    '''Raw moments up to 2nd order of points living in 2D. See Moments for more details.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo2d.moments import Moments2d
    >>> gm.Moments2d(10, np.array([2,3]), np.array([[1,2],[3,4]]))
    Moments2d(m0=10.0, mean=[0.2 0.3], cov=[[0.06, 0.14], [0.24, 0.31000000000000005]])
    '''

    def __repr__(self):
        return "Moments2d(m0={}, mean={}, cov={})".format(self.m0, self.mean, self.cov.tolist())
_bc.register_cast(Moments2d, Moments, lambda x: Moments(x.m0, x.m1, x.m2))
_bc.register_cast(Moments, Moments2d, lambda x: Moments2d(x.m0, x.m1, x.m2))
_bc.register_castable(Moments, Moments2d, lambda x: x.ndim==2)


_bc.register_cast(PointList2d, Moments2d, moments_from_pointlist)
