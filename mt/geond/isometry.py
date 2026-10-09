from mt import np
import mt.base.casting as _bc

from .affine import Aff
from .dilated_isometry import Dliso


__all__ = ['Iso']


class Iso(Dliso):
    '''Isometry = orthogonal transformation followed by translation.

    An isometry is a (Euclidean-)metric-preserving transformation. In other words, it is an affine
    transformation but the linear part is a unitary matrix. It maps a point `x` to
    `unitary @ x + offset`. It is a :class:`Dliso` whose scale is fixed to 1 (assigning `scale`
    raises `TypeError`). The product of two isometries is an isometry.

    Parameters
    ----------
    offset : numpy.ndarray
        the 1D translation vector. Defaults to the zero vector of length 2.
    unitary : numpy.ndarray
        the 2D unitary (orthogonal) matrix, assumed but not checked to be unitary. Defaults to the
        2x2 identity matrix.

    See Also
    --------
    Dliso : dilated isometry, which also has a uniform scale

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geond.isometry import Iso
    >>> R = np.array([[0., -1.], [1., 0.]])  # rotation by 90 degrees
    >>> i = Iso(offset=np.array([1., 2.]), unitary=R)
    >>> i
    Iso(offset=[1. 2.], unitary_diagonal=[0. 0.])
    >>> i << np.array([1., 0.])
    array([1., 3.])
    >>> i * ~i
    Iso(offset=[0. 0.], unitary_diagonal=[1. 1.])

    References:
        [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky MT is trying to advertise his paper!)
    '''

    # ----- base adaptation -----

    @property
    def scale(self):
        return 1
    scale.__doc__ = Dliso.scale.__doc__

    @scale.setter
    def scale(self, scale):
        raise TypeError("Scale value is read-only.")

    @property
    def linear(self):
        '''Returns the linear part of the affine transformation matrix of the isometry, which is
        the unitary matrix.'''
        return self.unitary

    # ----- methods -----

    def __init__(self, offset=np.zeros(2), unitary=np.identity(2)):
        self.offset = offset
        self.unitary = unitary

    def __repr__(self):
        return f"Iso(offset={self.offset}, unitary_diagonal={self.unitary.diagonal()})"

    # ----- base adaptation -----

    def multiply(self, other):
        if not isinstance(other, Iso):
            return super(Iso, self).__mul__(other)
        return Iso(offset=self << other.offset, unitary=np.dot(self.unitary, other.unitary))
    multiply.__doc__ = Dliso.multiply.__doc__

    def invert(self):
        # slow, and assuming the unitary matrix is invertible
        invUnitary = np.linalg.inv(self.unitary)
        return Iso(offset=np.dot(invUnitary, -self.offset), unitary=invUnitary)
    invert.__doc__ = Dliso.invert.__doc__


# ----- casting -----


_bc.register_cast(Iso, Dliso, lambda x: Dliso(offset=x.offset, unitary=x.unitary))
_bc.register_cast(Iso, Aff, lambda x: Aff(bias=x.offset, weight=x.weight))
