from ..geond.iou import *
from ..geo2d.iou import *

from mt.base import logger
logger.warn_module_move("mt.geo.iou", ["mt.geond.iou", "mt.geo2d.iou"])
