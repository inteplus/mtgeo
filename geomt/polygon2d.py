'''Polygon in 2D.

A polygon is represented as a collection of 2D points in either clockwise or counter-clockwise order. It is stored in a numpy array of shape (N,2).
'''

from mt.base import logger
logger.warn_module_move('geomt.polygon2d', 'mt.geo.polygon2d')

from mt.geo.polygon2d import *
