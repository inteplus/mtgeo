"""Raw moments up to 2nd order of 3D points."""

import glm

from mt import tp, np
import mt.base.casting as _bc

from ..geo import ThreeD
from ..geond import Moments, moments_from_pointlist, EPSILON
from .point_list import PointList3d


__all__ = ["Moments3d"]


class Moments3d(ThreeD, Moments):
    """Raw moments up to 2nd order of points living in 3D, implemnted in GLM.

    Overloaded operators are negation, multiplication with a scalar and true division with a scalar.

    Parameters
    ----------
    m0 : float
        0th-order raw moment
    m1 : glm.vec3
        1st-order raw moment
    m2 : glm.mat3 or numpy.ndarray
        2nd-order raw moment. If the input is numpy.ndarray, a row-major matrix is expected.
        Otherwise, a column-major matrix is expected.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo2d.moments import Moments2d
    >>> gm.Moments2d(10, np.array([2,3]), np.array([[1,2],[3,4]]))
    Moments2d(m0=10.0, mean=[0.2, 0.3], cov=[[0.06, 0.14], [0.24, 0.31000000000000005]])

    See Also
    --------
    Moments
        base class
    """

    def __init__(
        self,
        m0: float,
        m1: tp.Union[np.ndarray, glm.vec3],
        m2: tp.Union[np.ndarray, glm.mat3],
    ):
        self._m0 = m0
        self._m1 = glm.vec3(m1)
        if isinstance(m2, np.ndarray):
            self._m2 = glm.mat3(
                glm.vec3(m2[:, 0]), glm.vec3(m2[:, 1], glm.vec3(m2[:, 2]))
            )
        else:
            self._m2 = glm.mat3(m2)
        self._mean = None
        self._cov = None

    @property
    def mean(self):
        """Returns the mean vector."""
        if self._mean is None:
            self._mean = glm.vec3() if glm.abs(self.m0) < EPSILON else self.m1 / self.m0
        return self._mean

    @property
    def cov(self):
        """Returns the column-major covariance matrix."""
        if self._cov is None:
            self._cov = (
                glm.mat3()
                if glm.abs(self.m0) < EPSILON
                else (self.m2 / self.m0) - glm.outerProduct(self.mean, self.mean)
            )
        return self._cov

    def __repr__(self):
        return "Moments3d(m0={}, mean={}, cov={})".format(
            self.m0, self.mean.to_list(), self.cov.to_list()
        )


_bc.register_cast(Moments3d, Moments, lambda x: Moments(x.m0, x.m1, x.m2))
_bc.register_cast(Moments, Moments3d, lambda x: Moments3d(x.m0, x.m1, x.m2))
_bc.register_castable(Moments, Moments3d, lambda x: x.ndim == 3)


_bc.register_cast(PointList3d, Moments3d, moments_from_pointlist)
