'''A 3D box.

The module provides :class:`Box`, an axis-aligned 3D box, with its casts to
:class:`mt.geond.Hyperbox` and :class:`Moments3d`, its approximation from :class:`Moments3d` and
the function that joins the volumes of two boxes (see :func:`mt.geo.join_volume`).
'''

from mt import np
from mt.base.casting import *

from ..geo import ThreeD, register_approx, register_join_volume
from ..geond import EPSILON, Hyperbox
from .moments import Moments3d


__all__ = ['Box', 'cast_Hyperbox_to_Box', 'cast_Box_to_Moments3d', 'approx_Moments3d_to_Box']


class Box(ThreeD, Hyperbox):
    '''A 3D box.

    An axis-aligned box defined by its lowest and highest corners, with the coordinate order
    `(x, y, z)`. The box inherits from :class:`mt.geond.Hyperbox` and adds named coordinates
    (`x`, `lx`, `cx`, `min_x`, `max_x`, and the same for y and z), the moments of the box
    seen as a solid region, and `move`. See :class:`mt.geond.Hyperbox` for the other properties and
    methods (`volume`, `is_valid()`, `intersect`, `box_union`, `iou`, ...).

    Note we do not care if the box is open or partially closed or closed.

    Parameters
    ----------
    min_x, min_y, min_z : float
        the lowest coordinates of the box
    max_x, max_y, max_z : float
        the highest coordinates of the box
    force_valid : bool
        whether or not to swap the corners so that each `min` is not greater than its `max`. If
        False, a box with `min > max` is kept as it is (with negative scales, see
        :class:`mt.geond.Hyperbox`), although `min_x`, `max_x`, etc. always return sorted values.

    See Also
    --------
    mt.geond.Hyperbox : the base class, in any number of dimensions
    mt.geo3d.Ellipsoid : an ellipsoid, which can be bounded by a box

    Notes
    -----
    - `surface_area` currently raises `AttributeError` because of a typo in the code.
    - `intersect` of two disjoint boxes returns a box spanning the gap between them rather than an
      empty box, so `volume` and the volumes returned by `join_volume` are not meaningful in that
      case.
    - `box_union` and `validated` return a :class:`mt.geond.Hyperbox`, not a `Box`.

    Examples
    --------
    >>> from mt.geo3d import Box
    >>> b = Box(0, 0, 0, 1, 2, 3)
    >>> b
    Box(x=0.0, y=0.0, z=0.0, lx=1.0, ly=2.0, lz=3.0)
    >>> b.ndim
    3
    >>> b.min_coords, b.max_coords
    (array([0., 0., 0.]), array([1., 2., 3.]))
    >>> b.cx, b.cy, b.cz
    (0.5, 1.0, 1.5)
    >>> b.volume
    6.0
    >>> b.move([1, 1, 1])
    Box(x=1.0, y=1.0, z=1.0, lx=1.0, ly=2.0, lz=3.0)
    >>> c = Box(0.5, 1, 1, 2, 3, 2)
    >>> b.intersect(c)
    Box(x=0.5, y=1.0, z=1.0, lx=0.5, ly=1.0, lz=1.0)
    >>> round(b.iou(c), 4)
    0.0588
    >>> from mt.geo import join_volume
    >>> join_volume(b, c)
    (0.5, 5.5, 2.5, 8.5)
    >>> b.moment_x, b.moment_xx
    (3.0, 2.0)
    '''

    
    # ----- derived properties -----

    
    @property
    def min_x(self):
        '''lowest x-coordinate, `min_coords[0]`.'''
        return self.min_coords[0]

    @property
    def min_y(self):
        '''lowest y-coordinate, `min_coords[1]`.'''
        return self.min_coords[1]

    @property
    def min_z(self):
        '''lowest z-coordinate, `min_coords[2]`.'''
        return self.min_coords[2]

    @property
    def max_x(self):
        '''highest x-coordinate, `max_coords[0]`.'''
        return self.max_coords[0]

    @property
    def max_y(self):
        '''highest y-coordinate, `max_coords[1]`.'''
        return self.max_coords[1]

    @property
    def max_z(self):
        '''highest z-coordinate, `max_coords[2]`.'''
        return self.max_coords[2]

    @property
    def x(self):
        '''left, same as `min_x`.'''
        return self.min_x

    @property
    def y(self):
        '''top, same as `min_y`.'''
        return self.min_y

    @property
    def z(self):
        '''lowest z-coordinate, same as `min_z`.'''
        return self.min_z

    @property
    def lx(self):
        '''length in x-axis, `max_x - min_x`'''
        return self.max_x - self.min_x

    @property
    def ly(self):
        '''length in y-axis, `max_y - min_y`'''
        return self.max_y - self.min_y

    @property
    def lz(self):
        '''length in z-axis, `max_z - min_z`'''
        return self.max_z - self.min_z

    @property
    def cx(self):
        '''Center x-coordinate.'''
        return (self.min_x + self.max_x)/2

    @property
    def cy(self):
        '''Center y-coordinate.'''
        return (self.min_y + self.max_y)/2

    @property
    def cz(self):
        '''Center z-coordinate.'''
        return (self.min_z + self.max_z)/2

    @property
    def surface_area(self):
        '''Surface area, intended to be `2*(lx*ly + ly*lz + lz*lx)`.

        The current code has a typo (`max_coords.self.min_coords`) so it raises `AttributeError`.
        '''
        l = np.abs(self.max_coords.self.min_coords)
        return (l[0]*l[1]+l[1]*l[2]+l[2]*l[0])*2

    
    # ----- moments -----


    @property
    def moment_x(self):
        '''Returns the integral of x over the box's interior, `signed_volume*cx`.'''
        return self.signed_volume*self.cx

    @property
    def moment_y(self):
        '''Returns the integral of y over the box's interior, `signed_volume*cy`.'''
        return self.signed_volume*self.cy

    @property
    def moment_z(self):
        '''Returns the integral of z over the box's interior, `signed_volume*cz`.'''
        return self.signed_volume*self.cz

    @property
    def moment_xy(self):
        '''Returns the integral of x*y over the box's interior.'''
        return self.moment_x*self.cy

    @property
    def moment_yz(self):
        '''Returns the integral of y*z over the box's interior.'''
        return self.moment_y*self.cz

    @property
    def moment_zx(self):
        '''Returns the integral of z*x over the box's interior.'''
        return self.moment_z*self.cx

    @property
    def moment_xx(self):
        '''Returns the integral of x*x over the box's interior, `signed_volume*E[x^2]` with `E[x^2]`
        the average of `x^2` over the x-range.'''
        return self.signed_volume*(
            self.min_x*self.min_x+self.min_x*self.max_x+self.max_x*self.max_x
        )/3

    @property
    def moment_yy(self):
        '''Returns the integral of y*y over the box's interior, `signed_volume*E[y^2]` with `E[y^2]`
        the average of `y^2` over the y-range.'''
        return self.signed_volume*(
            self.min_y*self.min_y+self.min_y*self.max_y+self.max_y*self.max_y
        )/3

    @property
    def moment_zz(self):
        '''Returns the integral of z*z over the box's interior, `signed_volume*E[z^2]` with `E[z^2]`
        the average of `z^2` over the z-range.'''
        return self.signed_volume*(
            self.min_z*self.min_z+self.min_z*self.max_z+self.max_z*self.max_z
        )/3

    
    # ----- methods -----

    
    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z, force_valid=False):
        super(Box, self).__init__(
            np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z]),
            force_valid = force_valid
        )

    def __repr__(self):
        return f"Box(x={self.x}, y={self.y}, z={self.z}, lx={self.lx}, ly={self.ly}, lz={self.lz})"

    def intersect(self, other):
        res = super(Box, self).intersect(other)
        return Box(
            res.min_coords[0], res.min_coords[1], res.min_coords[2],
            res.max_coords[0], res.max_coords[1], res.max_coords[2]
        )

    def move(self, offset):
        '''Moves the Box by a given offset vector.

        Parameters
        ----------
        offset : sequence of 3 floats
            the displacement `(dx, dy, dz)` to add to the coordinates

        Returns
        -------
        Box
            a new box translated by `offset`. The original box is not modified.
        '''
        return Box(
            self.min_x + offset[0], self.min_y + offset[1], self.min_z + offset[2],
            self.max_x + offset[0], self.max_y + offset[1], self.max_z + offset[2]
        )


# ----- casting -----
        

register_cast(Box, Hyperbox, lambda x: Hyperbox(x.dlt_tfm))
register_castable(Hyperbox, Box, lambda x: x.dim==3)

def cast_Hyperbox_to_Box(x):
    '''Casts a Hyperbox to a Box.

    It is registered as the cast function from :class:`mt.geond.Hyperbox` to :class:`Box`; the
    hyperbox must have 3 dimensions.

    Parameters
    ----------
    x : mt.geond.Hyperbox
        a 3D hyperbox

    Returns
    -------
    Box
        the box with the same `min_coords` and `max_coords`

    Examples
    --------
    >>> import numpy as np
    >>> from mt.base.casting import cast
    >>> from mt.geond import Hyperbox
    >>> from mt.geo3d import Box
    >>> cast(Hyperbox(np.zeros(3), np.array([1., 2., 3.])), Box)
    Box(x=0.0, y=0.0, z=0.0, lx=1.0, ly=2.0, lz=3.0)
    '''
    min_coords = x.min_coords
    max_coords = x.max_coords
    return Box(
        min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2]
    )
register_cast(Hyperbox, Box, cast_Hyperbox_to_Box)


def cast_Box_to_Moments3d(obj):
    m0 = obj.signed_volume
    m1 = [obj.moment_x, obj.moment_y]
    mxy = obj.moment_xy
    myz = obj.moment_yz
    mzx = obj.moment_zx
    m2 = [[obj.moment_xx, mxy, mzx], [mxy, obj.moment_yy, myz], [mzx, myz, obj.moment_zz]]
    return Moments3d(m0, m1, m2)
register_cast(Box, Moments3d, cast_Box_to_Moments3d)


# ----- approximation -----


def approx_Moments3d_to_Box(obj):
    '''Approximates a Moments3d instance with a box such that the mean aligns with the box's center,
    and the covariance matrix of the instance is closest to the moment convariance matrix of the
    box.

    It is registered as the approximation from :class:`Moments3d` to :class:`Box`.

    Parameters
    ----------
    obj : Moments3d
        the moments to approximate

    Raises
    ------
    NotImplementedError
        always, as the function is not implemented yet
    '''
    raise NotImplementedError("I don't know if the coefficients are correct. Need to figure out.")
register_approx(Moments3d, Box, approx_Moments3d_to_Box)


# ----- joining volumes -----


def join_volume_Box_and_Box(obj1, obj2):
    '''Joins the volumes of two Boxes.

    It is the function registered with :func:`mt.geo.register_join_volume` for a pair of
    :class:`Box`, and it is used by :func:`mt.geo.join_volume`.

    Parameters
    ----------
    obj1 : Box
        the first box
    obj2 : Box
        the second box

    Returns
    -------
    intersection_volume : float
        the volume of the intersection of the two boxes' interior regions
    obj1_only_volume : float
        the volume of the interior of obj1 that does not belong to obj2
    obj2_only_volume : float
        the volume of the interior of obj2 that does not belong to obj1
    union_volume : float
        the volume of the union of the two boxes' interior regions

    Examples
    --------
    >>> from mt.geo import join_volume
    >>> from mt.geo3d import Box
    >>> join_volume(Box(0, 0, 0, 1, 1, 1), Box(0, 0, 0, 2, 1, 1))
    (1.0, 0.0, 1.0, 2.0)
    '''
    inter = obj1.intersect(obj2)
    inter_volume = inter.volume
    obj1_volume = obj1.volume
    obj2_volume = obj2.volume
    return (
        inter_volume, obj1_volume - inter_volume, obj2_volume - inter_volume,
        obj1_volume + obj2_volume - inter_volume
    )
register_join_volume(Box, Box, join_volume_Box_and_Box)
