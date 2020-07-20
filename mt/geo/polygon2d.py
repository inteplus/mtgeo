'''Polygon in 2D.

A polygon is represented as a collection of 2D points in either clockwise or counter-clockwise order. It is stored in a numpy array of shape (N,2).
'''

import numpy as _np
import math as _m

from mt.base import logger
logger.warn_module_move('mt.geo.polygon2d', ['mt.geo.polygon', 'mt.geo.polygon_integral'])

from .moments2d import EPSILON, moments2d


__all__ = ['trapezium_integral', 'signed_area', 'moment_x', 'moment_y', 'moment_xy', 'moment_xx', 'moment_yy', 'to_moments2d']


def trapezium_integral(poly, func):
    '''Applies the Shoelace algorithm to integrate a function over the interior of a polygon. Shoelace algorithm views the polygon as a sum of trapeziums.

    :Parameters:
        poly : polygon
            a polygon
        func : function
            a function that takes x1, y1, x2, y2 as input and returns a scalar
    :Returns:
        retval : scalar
            the integral over the polygon's interior

    :References:
        [1] Pham et al. Fast Polygonal Integration and Its Application in Extending Haar-like Features To Improve Object Detection. CVPR, 2010.
        [2] exterior algebra

    :Notes:
        We use OpenGL's convention here. x-axis points to the right side and y-axis points upward

    :Examples:
    >>> from mt.geo.polygon2d import trapezium_integral
    >>> import numpy as np
    >>> poly = np.array([[1,2]])
    >>> trapezium_integral(poly, None)
    0
    >>> poly = np.array([[20,10],[30,20]])
    >>> trapezium_integral(poly, None)
    0
    >>> poly = np.array([[1,1],[0,0],[1,0]])
    >>> trapezium_integral(poly, lambda x1, y1, x2, y2: x1*y2-x2*y1)
    1

    '''
    retval = 0
    N = len(poly)
    if N <= 2:
        return 0
    for i in range(N):
        z1 = poly[i]
        z2 = poly[(i+1)%N]
        retval += func(z1[0], z1[1], z2[0], z2[1])
    return retval

def signed_area(poly):
    '''Returns the signed area of a polygon.

    >>> from mt.geo.polygon2d import signed_area
    >>> import numpy as np
    >>> poly = np.array([[10,10],[20,10],[20,20]])
    >>> round(signed_area(poly), 3)
    -50.0
    '''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 0.5*(x2-x1)*(y1+y2))


def moment_x(poly):
    '''Returns the integral of x over the polygon's interior.

    >>> from mt.geo.polygon2d import moment_x
    >>> import numpy as np
    >>> poly = np.array([[3,4],[2,3],[3,2]])
    >>> round(moment_x(poly), 3)
    -2.667
    '''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/6*(x2-x1)*(x1*(y1*2+y2) + x2*(y1+y2*2)))

def moment_y(poly):
    '''Returns the integral of y over the polygon's interior.

    >>> from mt.geo.polygon2d import moment_y
    >>> import numpy as np
    >>> poly = np.array([[3,3],[2,2],[3,1]])
    >>> round(moment_y(poly), 3)
    -2.0
    '''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/6*(x2-x1)*(y1*y1 + y1*y2 + y2*y2))

def moment_xy(poly):
    '''Returns the integral of x*y over the polygon's interior.

    >>> from mt.geo.polygon2d import moment_xy
    >>> import numpy as np
    >>> poly = np.array([[3,3],[2,2],[3,1]])
    >>> round(moment_xy(poly), 3)
    -5.333
    '''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/24*(x2-x1)*(y1*y1*(x1*3+x2) + y1*y2*2*(x1+x2) + y2*y2*(x1+x2*3)))

def moment_xx(poly):
    '''Returns the integral of x*x over the polygon's interior.

    >>> from mt.geo.polygon2d import moment_xx
    >>> import numpy as np
    >>> poly = np.array([[3,3],[2,2],[3,1]])
    >>> round(moment_xx(poly), 3)
    -7.167
    '''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/12*(x2-x1)*(x1*x1*(y1*3+y2) + x1*x2*2*(y1+y2) + x2*x2*(y1+y2*3)))

def moment_yy(poly):
    '''Returns the integral of y*y over the polygon's interior.

    >>> from mt.geo.polygon2d import moment_yy
    >>> import numpy as np
    >>> poly = np.array([[3,3],[2,2],[3,1]])
    >>> round(moment_yy(poly), 3)
    -4.167
    '''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/12*(x2-x1)*(y1+y2)*(y1*y1+y2*y2))


def to_moments2d(poly):
    '''Computes all moments, up to 2nd-order of the polygon's interior.

    :Parameters:
        poly : polygon
            a polygon
    :Returns:
        retval : moments2d
            the collection of moments up to 2nd order

    :Examples:
    >>> from mt.geo.polygon2d import to_moments2d
    >>> import numpy as np
    >>> poly = np.array([[3,3],[2,2],[3,1]])
    >>> m = to_moments2d(poly)
    >>> round(m.m0, 3)
    -1.0
    >>> round(m.m1.sum(), 3)
    -4.667
    >>> round(m.m2.sum(), 3)
    -22.0
    >>> round(m.mean.sum(), 3)
    4.667
    >>> round(m.cov.sum(), 3)
    -22.444
    '''
    m0 = signed_area(poly)
    m1 = [moment_x(poly), moment_y(poly)]
    mxy = moment_xy(poly)
    m2 = [[moment_xx(poly), mxy], [mxy, moment_yy(poly)]]
    return moments2d(m0, m1, m2)
