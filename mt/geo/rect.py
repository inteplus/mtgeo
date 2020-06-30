'''A 2D rectangle.'''

import numpy as _np

from .box import box

__all__ = ['rect']

class rect(box):
    '''A 2D rectangle,

    Note we do not care if the rectangle is open or partially closed or closed.'''

    # ----- derived properties -----

    @property
    def min_x(self):
        return self.min_coords[0]

    @property
    def min_y(self):
        return self.min_coords[1]

    @property
    def max_x(self):
        return self.max_coords[0]

    @property
    def max_y(self):
        return self.max_coords[1]

    @property
    def x(self):
        return self.min_x

    @property
    def y(self):
        return self.min_y

    @property
    def w(self):
        return self.max_x - self.min_x

    @property
    def h(self):
        return self.max_y - self.min_y

    @property
    def cx(self):
        return (self.min_x + self.max_x)/2

    @property
    def cy(self):
        return (self.min_y + self.max_y)/2

    @property
    def area(self):
        return self.w*self.h

    # ----- methods -----

    def __init__(self, min_x, min_y, max_x, max_y, force_valid=False):
        super(rect, self).__init__(_np.array([min_x, min_y]), _np.array([max_x, max_y]), force_valid = force_valid)

    def __repr__(self):
        return "rect(x={}, y={}, w={}, h={})".format(self.x, self.y, self.w, self.h)

    def intersect(self, other):
        res = super(rect, self).intersect(other)
        return rect(res.min_coords[0], res.min_coords[1], res.max_coords[0], res.max_coords[1])

    def union(self, other):
        res = super(rect, self).union(other)
        return rect(res.min_coords[0], res.min_coords[1], res.max_coords[0], res.max_coords[1])

    def move(self, offset):
        '''Moves the rect by a given offset vector.'''
        return rect(self.min_x + offset[0], self.min_y + offset[1], self.max_x + offset[0], self.max_y + offset[1])
