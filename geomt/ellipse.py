'''There are many definitions of an ellipse. In our case, an ellipse is an affine transform of the unit circle x^2+y^2=1.'''

from mt.base import logger
logger.warn_module_move('geomt.ellipse', 'mt.geo.ellipse')

from mt.geo.ellipse import *
