'''Utilities for converting a rectangle into an octagon bounded by it.

An octagon is a cheap polygonal approximation of the ellipse inscribed in the rectangle.
'''

from mt import np


__all__ = ['ISQRT2', 'approx_oct']


ISQRT2 = 1/np.sqrt(2)

# octagon to approximate a circle originating at (0,0) with radius 1
std_oct = np.array([
    (1,       0),
    (ISQRT2,  ISQRT2),
    (0,       1),
    (-ISQRT2, ISQRT2),
    (-1,      0),
    (-ISQRT2, -ISQRT2),
    (0,      -1),
    (ISQRT2, -ISQRT2),
])


def approx_oct(min_x, min_y, max_x, max_y):
    '''Converts a rectangle into an octagon inside it.

    The octagon is the standard octagon approximating the unit circle (vertices at angles 0, 45,
    ..., 315 degrees), scaled and translated to the ellipse inscribed in the rectangle. The first
    vertex is the middle of the right edge and the vertices go from the x-axis towards the y-axis.

    Parameters
    ----------
    min_x : float
        min x-coordinate
    min_y : float
        min y-coordinate
    max_x : float
        max x-coordinate
    max_y : float
        max y-coordinate

    Returns
    -------
    polygon : numpy array
        array of shape (8,2) of 2D (x,y) points representing an octagon inside the rectangle
        (min_x, min_y, max_x, max_y)

    Examples
    --------
    >>> from mt.geo2d.rect2oct import approx_oct
    >>> approx_oct(0, 0, 2, 4).round(3)
    array([[2.   , 2.   ],
           [1.707, 3.414],
           [1.   , 4.   ],
           [0.293, 3.414],
           [0.   , 2.   ],
           [0.293, 0.586],
           [1.   , 0.   ],
           [1.707, 0.586]])
    '''
    c = np.array([min_x + max_x, min_y + max_y])*0.5
    r = np.array([max_x, max_y]) - c
    return std_oct*r + c
