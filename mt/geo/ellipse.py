'''There are many definitions of an ellipse. In our case, an ellipse is an affine transform of the unit circle x^2+y^2=1.'''

import numpy as _np
import numpy.linalg as _nl
from .affine_transformation import aff
from .rect import rect
from .moments2d import EPSILON, moments2d


__all__ = ['ellipse']


class ellipse(object):
    '''Ellipse, defined as an affine transform the unit circle x^2+y^2=1.

    Note that this representation is not unique, the same ellipse can be represented by an infinite number of affine transforms of the unit circle.

    If the unit circle is parameterised by `(cos(t), sin(t)) where t \in [0,2\pi)` then the ellipse is parameterised by `f0 + f1 cos(t) + f2 sin(t), where f0 is the bias vector, f1 and f2 are the first and second column of the weight matrix respectively, of the affine transformation. f1 and f2 are called conjugate diameters of the ellipse.

    '''

    def __init__(self, aff_tfm):
        '''Initialises an ellipse with an affine transformation.'''
        if not isinstance(aff_tfm, aff):
            raise ValueError("Only an instance of class `aff` is accepted.")
        if aff_tfm.dim != 2:
            raise ValueError("Only 2D affine transformation is accepted.")
        self.aff_tfm = aff_tfm

    def __repr__(self):
        return "ellipse(aff_tfm={})".format(self.aff_tfm)

    @property
    def f0(self):
        return self.aff_tfm.bias

    @property
    def f1(self):
        return self.aff_tfm.weight[:,0]

    @property
    def f2(self):
        return self.aff_tfm.weight[:,1]

    def transform(self, aff_tfm):
        '''Affine-transforms the ellipse. The resultant ellipse has affine transformation `aff_tfm*self.aff_tfm`.'''
        return ellipse(aff_tfm*self.aff_tfm)

    # ----- bounding rect -----

    def to_bounding_rect(self, rotated=False):
        '''Returns a bounding rectangle of the ellipse.

        :Parameters:
            rotated : bool
                If true, find a rotated bounding rectangle via eignevalue decomposition of the affine transformation. Otherwise, find the axis-aligned bounding rectangle.
        :Returns:
            For now, `rotated=True` is not yet implemented. We would just return an instance of rect.
        '''
        if rotated:
            raise NotImplementedError("Sorry we have not implemented rotated rect.")

        mx = _nl.norm(self.aff_tfm.weight[0])
        my = _nl.norm(self.aff_tfm.weight[1])
        cx = self.aff_tfm.bias[0]
        cy = self.aff_tfm.bias[1]
        return rect(cx-mx, cy-my, cx+mx, cy+my)

    @staticmethod
    def from_bounding_rect(x):
        '''Returns an axis-aligned ellipse bounded by the given axis-aligned rectangle x.'''
        if not isinstance(x, rect):
            raise ValueError("Input type a `rect`, '{}' given.".format(x.__class__))
        return ellipse(aff(weight=_np.array([[x.w/2,0],[0,x.h/2]]), bias=x.center_pt))

    # ----- moments2d -----

    def to_moments2d(self):
        '''Returns a moments2d capturing the pdf representing the interior of the ellipse.'''
        # MT-TODO
        pass

    @staticmethod
    def from_moments2d(mom):
        '''Creates an ellipse whose moments match with the given moments.

        :Parameters:
            mom : moments2d
                a collection of moments up to 2nd order in 2D

        :Returns:
            retval : ellipse
                an ellipse whose interior generates moments that match with the given collection of moments
        '''
        # MT-TODO
        return None
