'''Intersection over Union. In the current primitive form, it is treated as a separate module.

The module provides :func:`iou_impl`, which computes `inter / (vol1 + vol2 - inter + eps)` from the
volumes of the intersection and of the two objects, as used by :meth:`mt.geond.Hyperbox.iou`. Here,
"volume" means length, area, volume, etc. depending on the dimension.

Examples
--------
>>> from mt.geond.iou import iou_impl
>>> round(iou_impl(1.0, 8.0, 4.0), 4)
0.0909
>>> round(iou_impl(4.0, 4.0, 4.0), 4)
1.0
'''


__all__ = ['iou_impl']


def iou_impl(inter_volume, obj1_volume, obj2_volume, eps=1E-7):
    return inter_volume / (obj1_volume + obj2_volume - inter_volume + eps)
