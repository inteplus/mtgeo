import basemt.pyximportcpp as pyximport; pyximport.install()
#from .interval import interval
from .rect import rect
from .affine_transformation import *
from .dilated_isometry import *
from .isometry import *
from .dilatation import *
from .similarity2d import *
from .ellipse import ellipse

# TODO: doctest for each of the classes above
