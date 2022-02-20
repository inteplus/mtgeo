'''Module supporting the Lin2d class.'''

from mt import np

from ..geo import register_transform, register_transformable
from .linear_impl import Lin2d
from .moments import Moments2d
from .point_list import PointList2d
from .polygon import Polygon


__all__ = ['Lin2d', 'transform_Lin2d_on_Moments2d', 'transform_Lin2d_on_PointList2d', 'transform_Lin2d_on_Polygon']


# ----- transform functions -----


def transform_Lin2d_on_Moments2d(lin_tfm, moments):
    '''Transform a Moments2d using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Lin2d
        2D linear transformation
    moments : Moments2d
        2D moments

    Returns
    -------
    Moments2d
        linear-transformed 2D moments
    '''
    A = lin_tfm.matrix
    old_m0 = moments.m0
    old_mean = moments.mean
    old_cov = moments.cov
    new_mean = A @ old_mean
    new_cov = A @ old_cov @ A.T
    new_m0 = old_m0*abs(lin_tfm.det)
    new_m1 = new_m0*new_mean
    new_m2 = new_m0*(np.outer(new_mean, new_mean) + new_cov)
    return Moments2d(new_m0, new_m1, new_m2)
register_transform(Lin2d, Moments2d, transform_Lin2d_on_Moments2d)


def transform_Lin2d_on_ndarray(lin_tfm, point_array):
    '''Transform an array of 2D points using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Aff
        a 2D linear transformation
    point_array : numpy.ndarray with last dimension having the same length as the dimensionality of the transformation
        an array of 2D points

    Returns
    -------
    numpy.ndarray
        linear-transformed point array
    '''
    return point_array @ lin_tfm.matrix.T
register_transform(Lin2d, np.ndarray, transform_Lin2d_on_ndarray)
register_transformable(Lin2d, np.ndarray, lambda x, y: y.shape[-1] == 2)


def transform_Lin2d_on_PointList2d(lin_tfm, point_list):
    '''Transform a 2D point list using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Lin2d
        a 2D linear transformation
    point_list : PointList2d
        a 2D point list

    Returns
    -------
    PointList2d
        linear-transformed point list
    '''
    return PointList2d(point_list.points @ lin_tfm.matrix.T, check=False)
register_transform(Lin2d, PointList2d, transform_Lin2d_on_PointList2d)


def transform_Lin2d_on_Polygon(lin_tfm, poly):
    '''Transform a polygon using a 2D linear transformation.

    Parameters
    ----------
    lin_tfm : Lin2d
        a 2D linear transformation
    poly : Polygon
        a 2D polygon

    Returns
    -------
    Polygon
        linear-transformed polygon
    '''
    return Polygon(poly.points @ lin_tfm.matrix.T, check=False)
register_transform(Lin2d, Polygon, transform_Lin2d_on_Polygon)


