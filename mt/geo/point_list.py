from ..geond.point_list import *
from ..geo2d.point_list import *
from ..geo3d.point_list import *

from mt.base import logger
logger.warn_module_move("mt.geo.point_list", ["mt.geo2d.point_list", "mt.geo3d.point_list", "mt.geond.point_list"])
