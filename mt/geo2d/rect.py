"""A 2D rectangle."""

import math

from mt import np, tp
from mt.base.casting import *

from ..geo import TwoD, register_approx, register_join_volume, join_volume
from ..geond import Hyperbox
from .moments import Moments2d
from .shapely import HasShapely


__all__ = [
    "Rect",
    "cast_Hyperbox_to_Rect",
    "cast_Rect_to_Moments2d",
    "approx_Moments2d_to_Rect",
]


class Rect(HasShapely, TwoD, Hyperbox):
    """A 2D axis-aligned rectangle, defined by its minimum and maximum corners.

    Coordinates are `(x, y)`. In image coordinates `x` is the column (to the right) and `y` is the
    row (downward), so `(min_x, min_y)` is the top-left corner. The coordinate properties
    (`min_x`, `max_x`, ...) always return sorted values even if the corners were given in the
    wrong order, so width, height and area are never negative.

    Note we do not care if the rectangle is open or partially closed or closed.

    Parameters
    ----------
    min_x : float
        the x-coordinate of the first corner (the left side)
    min_y : float
        the y-coordinate of the first corner (the top side)
    max_x : float
        the x-coordinate of the opposite corner (the right side)
    max_y : float
        the y-coordinate of the opposite corner (the bottom side)
    force_valid : bool
        whether or not to normalise the internal representation so that the first corner is the
        minimum one. It does not change the values the properties return.

    Examples
    --------
    >>> from mt.geo2d import Rect
    >>> r = Rect(10, 20, 30, 60)
    >>> r
    Rect(x=10.0, y=20.0, w=20.0, h=40.0)
    >>> r.area, r.circumference
    (800.0, 120.0)
    >>> r.center_pt
    array([20., 40.])
    >>> r.to_json()
    [10.0, 20.0, 30.0, 60.0]

    See Also
    --------
    RRect : rotated rectangle
    mt.geond.Hyperbox : base class
    """

    # ----- internal representations -----

    @property
    def shapely(self):
        """Shapely representation for fast intersection operations.

        It is a shapely box, buffered by 0.0001 (so its area is slightly larger than the
        exact one). The result is cached in the instance.
        """
        if not hasattr(self, "_shapely"):
            import shapely.geometry as _sg

            self._shapely = _sg.box(self.min_x, self.min_y, self.max_x, self.max_y)
            self._shapely = self._shapely.buffer(
                0.0001
            )  # to clean up any (multi and/or non-simple) polygon into a simple polygon
        return self._shapely

    # ----- derived properties -----

    @property
    def min_x(self):
        """lowest x-coordinate."""
        return self.min_coords[0]

    @property
    def min_y(self):
        """lowest y-coordinate."""
        return self.min_coords[1]

    @property
    def max_x(self):
        """highest x-coordinate."""
        return self.max_coords[0]

    @property
    def max_y(self):
        """highest y-coordinate."""
        return self.max_coords[1]

    @property
    def x(self):
        """left, same as min_x."""
        return self.min_x

    @property
    def y(self):
        """top (in image coordinates), same as min_y."""
        return self.min_y

    @property
    def w(self):
        """width, i.e. max_x - min_x."""
        return self.max_x - self.min_x

    @property
    def h(self):
        """height, i.e. max_y - min_y."""
        return self.max_y - self.min_y

    @property
    def cx(self):
        """Center x-coordinate."""
        return (self.min_x + self.max_x) / 2

    @property
    def cy(self):
        """Center y-coordinate."""
        return (self.min_y + self.max_y) / 2

    @property
    def area(self):
        """Absolute area (width times height)."""
        return abs(self.w * self.h)

    @property
    def circumference(self):
        """Circumference (perimeter), i.e. twice the sum of width and height."""
        return (abs(self.w) + abs(self.h)) * 2

    @property
    def min_pt(self):
        """Corner point with minimum coordinates, as a numpy array `[min_x, min_y]`."""
        return np.array([self.min_x, self.min_y])

    @property
    def max_pt(self):
        """Corner point with maximum coordinates, as a numpy array `[max_x, max_y]`."""
        return np.array([self.max_x, self.max_y])

    @property
    def center_pt(self):
        """Center point, as a numpy array `[cx, cy]`."""
        return np.array([self.cx, self.cy])

    # ----- moments -----

    @property
    def signed_area(self):
        """Returns the signed area of the rectangle, `w*h`. As `w` and `h` are never negative for a
        Rect, it equals :attr:`area`."""
        return self.w * self.h

    @property
    def moment1(self):
        """First-order moment, i.e. the numpy array `[moment_x, moment_y]`."""
        if not hasattr(self, "_moment1"):
            self._moment1 = self.signed_area * self.center_pt
        return self._moment1

    @property
    def moment_x(self):
        """Returns the integral of x over the rectangle's interior."""
        return self.moment1[0]

    @property
    def moment_y(self):
        """Returns the integral of y over the rectangle's interior."""
        return self.moment1[1]

    @property
    def moment2(self):
        """Second-order moment, i.e. the symmetric 2x2 numpy array `[[xx, xy], [xy, yy]]` of
        integrals of `x*x`, `x*y` and `y*y` over the rectangle's interior."""
        if not hasattr(self, "_moment2"):
            moment_xx = (
                self.signed_area
                * (
                    self.min_x * self.min_x
                    + self.min_x * self.max_x
                    + self.max_x * self.max_x
                )
                / 3
            )
            moment_xy = self.moment_x * self.cy  # self.sign*self.cx*self.cy
            moment_yy = (
                self.signed_area
                * (
                    self.min_y * self.min_y
                    + self.min_y * self.max_y
                    + self.max_y * self.max_y
                )
                / 3
            )
            self._moment2 = np.array([[moment_xx, moment_xy], [moment_xy, moment_yy]])
        return self._moment2

    @property
    def moment_xy(self):
        """Returns the integral of x*y over the rectangle's interior."""
        return self.moment2[0][1]

    @property
    def moment_xx(self):
        """Returns the integral of x*x over the rectangle's interior."""
        return self.moment2[0][0]

    @property
    def moment_yy(self):
        """Returns the integral of y*y over the rectangle's interior."""
        return self.moment2[1][1]

    # ----- serialization -----

    def to_json(self):
        """Returns a list [min_x, min_y, max_x, max_y].

        See Also
        --------
        from_json : the reverse operation
        """
        return [self.min_x, self.min_y, self.max_x, self.max_y]

    @staticmethod
    def from_json(json_obj):
        """Creates a Rect from a JSON-like object.

        Parameters
        ----------
        json_obj : list
            list [min_x, min_y, max_x, max_y]

        Returns
        -------
        Rect
            output rectangle

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect.from_json([1, 2, 4, 6])
        Rect(x=1.0, y=2.0, w=3.0, h=4.0)
        """
        return Rect(json_obj[0], json_obj[1], json_obj[2], json_obj[3])

    def to_tensor(self):
        """Returns a tensorflow tensor [min_x, min_y, max_x, max_y] representing the rect."""
        from mt import tf

        return tf.convert_to_tensor(self.to_json())

    # ----- methods -----

    def __init__(self, min_x, min_y, max_x, max_y, force_valid=False):
        """Initialises a rectangle from two opposite corners. See the class docstring."""
        super(Rect, self).__init__(
            np.array([min_x, min_y]), np.array([max_x, max_y]), force_valid=force_valid
        )

    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

    def intersect(self, other):
        """Returns the intersection of this rectangle with another one, as a Rect.

        Notes
        -----
        If the two rectangles do not overlap, no empty rectangle is returned: the result is the
        rectangle spanning the gap between them (the coordinates are sorted), which is not
        meaningful. Check for overlap beforehand if that can happen.

        Parameters
        ----------
        other : Rect
            the other rectangle

        Returns
        -------
        Rect
            the intersection

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect(0, 0, 10, 10).intersect(Rect(5, 5, 20, 20))
        Rect(x=5.0, y=5.0, w=5.0, h=5.0)
        """
        res = super(Rect, self).intersect(other)
        return Rect(
            res.min_coords[0], res.min_coords[1], res.max_coords[0], res.max_coords[1]
        )

    def box_union(self, other):
        """Returns the smallest rectangle containing this rectangle and another one.

        Parameters
        ----------
        other : Rect
            the other rectangle

        Returns
        -------
        Rect
            the bounding rectangle of the two

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect(0, 0, 10, 10).box_union(Rect(5, 5, 20, 20))
        Rect(x=0.0, y=0.0, w=20.0, h=20.0)
        """
        res = super(Rect, self).box_union(other)
        return Rect(
            res.min_coords[0], res.min_coords[1], res.max_coords[0], res.max_coords[1]
        )

    def iou(self, other, epsilon=1e-7):
        """Computes the Intersection-over-Union ratio with another geometry object.

        It uses :func:`mt.geo.join_volume`, so `other` can be any object that has a registered
        join with Rect (Rect, Polygon, RRect). See :meth:`intersect` for the caveat about
        non-overlapping Rect pairs.

        Parameters
        ----------
        other : Rect or Polygon or RRect
            the other geometry object
        epsilon : float
            a small value added to the union area to avoid division by zero

        Returns
        -------
        float
            the IoU ratio, between 0 and 1

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> round(Rect(0, 0, 10, 10).iou(Rect(5, 0, 15, 10)), 3)
        0.333
        """
        inter, _, _, union = join_volume(self, other)
        return inter / (union + epsilon)

    def move(self, offset):
        """Moves the Rect by a given offset vector.

        Parameters
        ----------
        offset : iterable
            the `(dx, dy)` displacement

        Returns
        -------
        Rect
            a new moved rectangle

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect(0, 0, 10, 10).move([5, -1])
        Rect(x=5.0, y=-1.0, w=10.0, h=10.0)
        """
        return Rect(
            self.min_x + offset[0],
            self.min_y + offset[1],
            self.max_x + offset[0],
            self.max_y + offset[1],
        )

    # ----- approximation -----

    def approx_to_square(self) -> "Rect":
        """Returns an approximated square of the same area, centered at the same point.

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect(0, 0, 8, 2).approx_to_square()
        Rect(x=2.0, y=-1.0, w=4.0, h=4.0)
        """
        side = np.sqrt(self.area)
        cx, cy = self.cx, self.cy
        r = side / 2
        return Rect(cx - r, cy - r, cx + r, cy + r)

    def enlarge_to_square(self) -> "Rect":
        """Returns an enlarged square, centered at the same point, with side being the maximum of
        width and height.

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect(0, 0, 8, 2).enlarge_to_square()
        Rect(x=0.0, y=-3.0, w=8.0, h=8.0)
        """
        r = max(self.w, self.h) * 0.5
        cx, cy = self.cx, self.cy
        return Rect(cx - r, cy - r, cx + r, cy + r)

    def enlarge_to_square_crop(
        self, imgres: tp.Iterable[int], min_length: int = 224
    ) -> "Rect":
        """Returns an enlarged square containing the rectangle.

        The enlarged square is guaranteed to reside completey in a given image resolution and to
        have length equal or greater than a minimal length.

        Parameters
        ----------
        imgres : iterable
            tuple [width, height] defining the image resolution where the square must stay fully
            inside
        min_length : int
            minimal number of pixels in each dimension of the output square

        Returns
        -------
        Rect
            a square satisfying the conditions above, with all coordinates being integer. The side
            is `max(ceil(max(w, h)), min_length)`; the square is centered on the rectangle when
            possible, otherwise shifted to stay inside the image.

        Examples
        --------
        >>> from mt.geo2d import Rect
        >>> Rect(0, 0, 10, 20).enlarge_to_square_crop([100, 100], min_length=8)
        Rect(x=0.0, y=0.0, w=20.0, h=20.0)
        >>> Rect(90, 0, 100, 10).enlarge_to_square_crop([100, 100], min_length=20)
        Rect(x=80.0, y=0.0, w=20.0, h=20.0)
        """
        l = math.ceil(max(self.w, self.h))
        l = int(max(l, min_length))

        mx = int(self.cx - l / 2)
        mx = max(mx, 0)
        mx = min(mx, imgres[0] - l)

        my = int(self.cy - l / 2)
        my = max(my, 0)
        my = min(my, imgres[1] - l)

        return Rect(mx, my, mx + l, my + l)


# ----- casting -----


register_cast(Rect, Hyperbox, lambda x: Hyperbox(x.dlt_tfm))
register_castable(Hyperbox, Rect, lambda x: x.dim == 2)


def cast_Hyperbox_to_Rect(x):
    """Casts a 2D Hyperbox to a Rect."""
    min_coords = x.min_coords
    max_coords = x.max_coords
    return Rect(min_coords[0], min_coords[1], max_coords[0], max_coords[1])


register_cast(Hyperbox, Rect, cast_Hyperbox_to_Rect)


def cast_Rect_to_Moments2d(obj):
    """Casts a Rect to the Moments2d of its interior (signed area, first and second moments)."""
    return Moments2d(obj.signed_area, obj.moment1, obj.moment2)


register_cast(Rect, Moments2d, cast_Rect_to_Moments2d)


# ----- approximation -----


def approx_Moments2d_to_Rect(obj):
    """Approximates a Moments2d instance with a rect such that the mean aligns with the rect's
    center, and the covariance matrix of the instance is closest to the moment covariance matrix of
    the rect.

    Parameters
    ----------
    obj : Moments2d
        the moments to be approximated

    Returns
    -------
    Rect
        an axis-aligned rectangle with the same area (`abs(m0)`) and mean, whose covariance fits
        the diagonal of the covariance of `obj`
    """
    cx, cy = obj.mean
    cov = obj.cov

    # w = half width, h = half height
    size = abs(obj.m0)
    hw3 = cov[0][0] * size * 0.75  # should be >= 0
    wh3 = cov[1][1] * size * 0.75  # should be >= 0
    wh = np.sqrt(np.sqrt(wh3 * hw3))
    h = np.sqrt(wh3 / wh)
    w = np.sqrt(hw3 / wh)
    return Rect(cx - w, cy - h, cx + w, cy + h)


register_approx(Moments2d, Rect, approx_Moments2d_to_Rect)


# ----- joining volumes -----


def join_volume_Rect_and_Rect(obj1, obj2):
    """Joins the areas of two Rects.

    This function is registered with :func:`mt.geo.join_volume` for the pair (Rect, Rect). See
    :meth:`Rect.intersect` for the caveat about non-overlapping rectangles.

    Parameters
    ----------
    obj1 : Rect
        the first rectangle
    obj2 : Rect
        the second rectangle

    Returns
    -------
    intersection_area : float
        the area of the intersection of the two Rects' interior regions
    obj1_only_area : float
        the area of the interior of obj1 that does not belong to obj2
    obj2_only_area : float
        the area of the interior of obj2 that does not belong to obj1
    union_area : float
        the area of the union of the two Rects' interior regions

    Examples
    --------
    >>> from mt.geo2d import Rect
    >>> from mt.geo2d.rect import join_volume_Rect_and_Rect
    >>> join_volume_Rect_and_Rect(Rect(0, 0, 2, 2), Rect(1, 1, 3, 3))
    (1.0, 3.0, 3.0, 7.0)
    """
    inter = obj1.intersect(obj2)
    inter_area = inter.area
    obj1_area = obj1.area
    obj2_area = obj2.area
    return (
        inter_area,
        obj1_area - inter_area,
        obj2_area - inter_area,
        obj1_area + obj2_area - inter_area,
    )


register_join_volume(Rect, Rect, join_volume_Rect_and_Rect)
