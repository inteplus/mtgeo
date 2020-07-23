'''There are many definitions of an ellipse. In our case, an ellipse is an affine transform of the unit circle x^2+y^2=1.'''

import math as _m
import numpy as _np
import numpy.linalg as _nl

from mt.base.casting import register_cast
from mt.base.deprecated import deprecated_func

from .affine2d import Aff2d, Lin2d
from .rect import rect
from .moments import EPSILON, Moments2d
from .approximation import register_approx
from .transformation import transform, register_transform
from .object import GeometricObject, TwoD


__all__ = ['ellipse', 'Ellipse', 'cast_ellipse_to_moments', 'approx_moments_to_ellipse', 'transform_Aff2d_on_Ellipse']


class Ellipse(TwoD, GeometricObject):
    '''Ellipse, defined as an affine transform the unit circle x^2+y^2=1.

    If the unit circle is parameterised by `(cos(t), sin(t))` where `t \in [0,2\pi)` then the ellipse is parameterised by `f0 + f1 cos(t) + f2 sin(t)`, where `f0` is the bias vector, `f1` and `f2` are the first and second column of the weight matrix respectively, of the affine transformation. `f1` and `f2` are called first and second axes of the ellipse.

    Note that this representation is not unique, the same ellipse can be represented by an infinite number of affine transforms of the unit circle. To make the representation unique, we further assert that when f1 and f2 are perpendicular (linearly independent), the ellipse is normalised, and use the normalised version as a unique representation. You can normalise either at initialisation time, or later by invoking member function `normalised`.

    Parameters
    ----------
    aff_tfm : Aff2d
        an affine transformation
    make_normalised : bool
        whether or not to adjust the affine transformation to make a normalised representation of the ellipse

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geo.ellipse import Aff2d, Ellipse, Lin2d
    >>> a = Aff2d(offset=np.array([2,3]), linear=Lin2d(scale=np.array([0.3,7]), shear=0.3, angle=0.3))
    >>> Ellipse(a)
    Ellipse(aff_tfm=Aff2d(offset=[2 3], linear=Lin2d(scale=[-0.3  7. ], shear=-0.30000000000000787, angle=-1.5702443840777238)))
    '''

    def __init__(self, aff_tfm, make_normalised=True):
        '''Initialises an ellipse with an affine transformation.'''
        if not isinstance(aff_tfm, Aff2d):
            raise ValueError("Only an instance of class `Aff2d` is accepted.")
        if make_normalised:
            U, S, VT = _nl.svd(aff_tfm.weight, full_matrices=False)
            aff_tfm = Aff2d(offset=aff_tfm.offset, linear=Lin2d.from_matrix(U @ _np.diag(S)))            
        self.aff_tfm = aff_tfm

    def __repr__(self):
        return "Ellipse(aff_tfm={})".format(self.aff_tfm)

    @property
    def f0(self):
        '''origin'''
        return self.aff_tfm.bias

    @property
    def f1(self):
        '''first axis'''
        return self.aff_tfm.weight[:,0]

    @property
    def f2(self):
        '''second axis'''
        return self.aff_tfm.weight[:,1]

    @property
    def area(self):
        '''The absolute area of the ellipse's interior.'''
        return _m.pi*abs(self.aff_tfm.det)

    def normalised(self):
        '''Returns an equivalent ellipse where f1 and f2 are perpendicular (linearly independent).'''
        return Ellipse(self.aff_tfm.offset, make_normalised=True)

    @deprecated_func("0.3.8", suggested_func=["mt.geo.transformation.transform", "mt.geo.ellipse.transform_Aff2d_on_Ellipse"], removed_version="0.6.0", docstring_prefix="        ")
    def transform(self, aff_tfm):
        '''Affine-transforms the ellipse. The resultant ellipse has affine transformation `aff_tfm*self.aff_tfm`.'''
        if not isinstance(aff_tfm, Aff2d):
            raise ValueError("Only an instance of class `Aff2d` is accepted.")
        return Ellipse(aff_tfm*self.aff_tfm)

    # ----- bounding rect -----

    def to_bounding_rect(self, rotated=False):
        '''Returns a bounding rectangle of the ellipse.

        Parameters
        ----------
        rotated : bool
            If true, find a rotated bounding rectangle via eignevalue decomposition of the affine transformation. Otherwise, find the axis-aligned bounding rectangle.

        Returns
        -------
        For now, `rotated=True` is not yet implemented. We would just return an instance of rect.
        '''
        if rotated:
            raise NotImplementedError("Sorry we have not implemented rotated rect.")

        weight = self.aff_tfm.weight
        mx = _nl.norm(weight[0])
        my = _nl.norm(weight[1])
        cx, cy = self.aff_tfm.bias
        return rect(cx-mx, cy-my, cx+mx, cy+my)

    @staticmethod
    def from_bounding_rect(x):
        '''Returns an axis-aligned ellipse bounded by the given axis-aligned rectangle x.'''
        if not isinstance(x, rect):
            raise ValueError("Input type a `rect`, '{}' given.".format(x.__class__))
        return Ellipse(Aff2d(linear=Lin2d(scale=[x.w/2, x.h/2]), offset=x.center_pt))


ellipse = Ellipse # for now


# ----- casting -----

def cast_ellipse_to_moments(obj):
    a = _m.pi/4
    moments = Moments2d(_m.pi, [0,0], [[a,0],[0,a]]) # unit circle's moments
    return transform(obj.aff_tfm, moments) # transform
register_cast(Ellipse, Moments2d, cast_ellipse_to_moments)


def approx_moments_to_ellipse(obj):
    '''Approximates a Moments2d instance with a normalised Ellipse that has the same mean and covariance as the mean and covariance of the instance.'''
    # A
    AAT = obj.cov*4
    w, v = _nl.eig(AAT)
    A = v @ _np.diag(_np.sqrt(w))

    # aff_tfm
    aff_tfm = Aff2d(offset=obj.mean, linear=Lin2d.from_matrix(A))
    return Ellipse(aff_tfm)
register_approx(Moments2d, Ellipse, approx_moments_to_ellipse)


# ----- transform functions -----


def transform_Aff2d_on_Ellipse(afm_tfm, obj):
    '''Affine-transforms an Ellipse. The resultant Ellipse has affine transformation `aff_tfm*obj.aff_tfm`.

    Parameters
    ----------
    aff_tfm  : Aff2d
        an 2D affine transformation
    obj : Ellipse
        an ellipse

    Returns
    -------
    Ellipse
        the affine-transformed ellipse
    '''
    return Ellipse(aff_tfm*obj.aff_tfm)
register_transform(Aff2d, Ellipse, transform_Aff2d_on_Ellipse)
