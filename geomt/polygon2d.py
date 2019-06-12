'''Polygon in 2D.

A polygon is represented as a collection of 2D points in either clockwise or counter-clockwise order. It is stored in a numpy array of shape (N,2).
'''

import numpy as _np
import math as _m
from .moments2d import EPSILON, moments2d


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

    :Examples:
    >>> from geomt.polygon2d import trapezium_integral
    >>> import numpy as np
    >>> poly = np.array([[1,2]])
    >>> trapezium_integral(poly, None)
    0
    >>> poly = np.array([[20,10],[30,20]])
    >>> trapezium_integral(poly, None)
    0

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
    '''Returns the signed area of a polygon.'''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 0.5*(x2-x1)*(y1+y2))


def moment_x(poly):
    '''Returns the integral of x over the polygon's interior.'''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/6*(x2-x1)*(x1*(y1*2+y2) + x2*(y1+y2*2)))

def moment_y(poly):
    '''Returns the integral of y over the polygon's interior.'''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/6*(x2-x1)*(y1*y1 + y1*y2 + y2*y2))

def moment_xy(poly):
    '''Returns the integral of x*y over the polygon's interior.'''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/24*(x2-x1)*(y1*y1*(x1*3+x2) + y1*y2*2*(x1+x2) + y2*y2*(x1+x2*3)))

def moment_xx(poly):
    '''Returns the integral of x*x over the polygon's interior.'''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/12*(x2-x1)*(x1*x1*(y1*3+y2) + x1*x2*2*(y1+y2) + x2*x2*(y1+y2*3)))

def moment_yy(poly):
    '''Returns the integral of y*y over the polygon's interior.'''
    return trapezium_integral(poly, lambda x1, y1, x2, y2: 1/12*(x2-x1)*(y1+y2)*(y1*y1+y2*y2))


def to_moments2d(poly):
    '''Computes all moments, up to 2nd-order of the polygon's interior.

    :Parameters:
        poly : polygon
            a polygon
    :Returns:
        retval : moments2d
            the collection of moments up to 2nd order
    '''
    m0 = signed_area(poly)
    m1 = [moment_x(poly), moment_y(poly)]
    mxy = moment_xy(poly)
    m2 = [[moment_xx(poly), mxy], [mxy, moment_yy(poly)]]
    return moments2d(m0, m1, m2)

# MT-TODO: make some doctests for each function
