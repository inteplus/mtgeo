from ..geond.point import *
from ..geo2d.point import *
from ..geo3d.point import *

from mt.base import logger
logger.warn_module_move("mt.geo.point", ["mt.geo2d.point", "mt.geo3d.point", "mt.geond.point"])
