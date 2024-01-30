'''Raw moments up to 2nd order of 3D points.'''

from mt import np
import sys as _sys

import mt.base.casting as _bc

from ..geo import ThreeD
from ..geond import Moments, moments_from_pointlist
from .point_list import PointList3d


__all__ = ['Moments3d']


class Moments3d(ThreeD, Moments):
    '''Raw moments up to 2nd order of points living in 3D. See Moments for more details.'''

    def __repr__(self):
        return "Moments3d(m0={}, mean={}, cov={})".format(self.m0, self.mean, self.cov.tolist())
_bc.register_cast(Moments3d, Moments, lambda x: Moments(x.m0, x.m1, x.m2))
_bc.register_cast(Moments, Moments3d, lambda x: Moments3d(x.m0, x.m1, x.m2))
_bc.register_castable(Moments, Moments3d, lambda x: x.ndim==3)


_bc.register_cast(PointList3d, Moments3d, moments_from_pointlist)
