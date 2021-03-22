from ..geond.moments import *
from ..geo2d.moments import *
from ..geo3d.moments import *

from mt.base import logger
logger.warn_module_move("mt.geo.moments", ["mt.geo2d.moments", "mt.geo3d.moments", "mt.geond.moments"])
