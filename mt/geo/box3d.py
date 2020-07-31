import numpy as _np

from mt.base.deprecated import deprecated_func
from .dilatation import Dlt
from .object import ThreeD
from .box import Box

__all__ = ['Box3d']

class Box3d(Box):
    '''Axis-aligned 3D box.

    An axis-aligned 3D box is defined as the set of points of the cube `[-1,1]^3` transformed by a dilatation.

    Note that is is a many-to-one representation. For each box, there are up to 8 dilatations that map the cube `[-1,1]^3` to it.

    # MT-TODO: continue from here, make it like Rect, with x, y, z coordinates

    Parameters
    ----------
    min_coords : 1d array or Dlt
        array of minimum coordinate values for all the dimensions, or the dilatation if self. In case of the latter, the `max_coords` argument is ignored.
    max_coords : 1d array
        array of maximum coordinate values for all the dimensions. If it is None, then `min_coords` represents `max_coords` and the minimum coordinate values are assumed 0.
    force_valid : bool
        whether or not to sort out `min_coords` and `max_coords` to make the the minus point meet the min_coords and the plus point meet the max_coords.

    Attributes
    ----------
    dlt_tfm : Dlt
        the dilatation equivalent, which can be get/set
    dim : int
        number of dimensions
    minus_pt : point
        the (-1,)^n point after being transformed by the dilatation
    plus_pt : point
        the (+1,)^n point after being transformed by the dilatation
    min_coords : point
        minimum coordinates
    max_coords : point
        maximum coordinates
    center_pt : point
        center point
    size : size/point
        box size
    '''

    # ----- data encapsulation -----

    @property
    def dlt_tfm(self):
        '''the dilatation'''
        return self.__dlt_tfm

    @dlt_tfm.setter
    def dlt_tfm(self, dlt_tfm):
        self.__dlt_tfm = dlt_tfm

    # ----- derived properties -----

    @classmethod
    def ndim(self):
        '''The dimensionality'''
        return self.dim

    @property
    def dim(self):
        '''The dimensionality'''
        return self.dlt_tfm.dim

    @property
    def minus_pt(self):
        '''The (-1,)^n point after being transformed by the dilatation.'''
        return self.dlt_tfm.offset - self.dlt_tfm.scale

    @property
    def plus_pt(self):
        '''The (+1,)^n point after being transformed by the dilatation.'''
        return self.dlt_tfm.offset + self.dlt_tfm.scale

    @property
    def min_coords(self):
        '''minimum coordinates'''
        return _np.minimum(self.minus_pt, self.plus_pt)

    @property
    def max_coords(self):
        '''maximum coordinates'''
        return _np.maximum(self.minus_pt, self.plus_pt)

    @property
    def center_pt(self):
        '''center point'''
        return self.dlt_tfm.offset

    @property
    def size(self):
        '''box size'''
        return _np.abs(self.dlt_tfm.scale*2)

    # ----- methods -----

    def __init__(self, min_coords, max_coords=None, force_valid=False):
        if isinstance(min_coords, Dlt):
            self.dlt_tfm = min_coords
        else:
            if max_coords is None:
                max_coords = min_coords
                min_coords = _np.zeros(dim)
            self.dlt_tfm = Dlt(offset=(max_coords + min_coords)/2, scale=(max_coords-min_coords)/2)

        if force_valid:
            min_coords = self.min_coords
            max_coords = self.max_coords
            self.dlt_tfm = Dlt(offset=(max_coords + min_coords)/2, scale=(max_coords-min_coords)/2)

    def __repr__(self):
        return "Box({})".format(self.dlt_tfm)

    def is_valid(self):
        return (self.dlt_tfm.scale >= 0).all()

    def validated(self):
        '''Returns a validated version of the box.'''
        min_coords = self.min_coords
        max_coords = self.max_coords
        return Box(Dlt(offset=(max_coords + min_coords)/2, scale=max_coords-min_coords))

    def intersect(self, other):
        return Box(_np.maximum(self.min_coords, other.min_coords), _np.minimum(self.max_coords, other.max_coords))

    def union(self, other):
        return Box(_np.minimum(self.min_coords, other.min_coords), _np.maximum(self.max_coords, other.max_coords))


class box(Box):

    __doc__ = Box.__doc__

    @deprecated_func("0.4.2", suggested_func='mt.geo.box.Box.__init__', removed_version="0.6.0")
    def __init__(self, *args, **kwargs):
        super(box, self).__init__(*args, **kwargs)
