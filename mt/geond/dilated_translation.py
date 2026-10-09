import math

from mt import np
import mt.base.casting as _bc
from ..geo import register_approx, register_transform, register_transformable
from .dilated_isometry import Dliso
from .dilatation import Dlt


__all__ = ["Dltra", "approx_Dliso_to_Dltra", "approx_Dlt_to_Dltra"]


class Dltra(Dliso):
    """Dilated translation = Translation following a uniform scaling.

    A dilated translation can be seen as both a dilatation and a dilated isometry. It maps a point
    `x` to `scale * x + offset`. It is a :class:`Dliso` whose unitary matrix is fixed to the
    identity matrix (assigning `unitary` raises `TypeError`). The product of two dilated
    translations is a dilated translation, and so is the inverse returned by `~`. Such
    transformations form a Lie group, so they have a matrix logarithm :meth:`logm`, a matrix
    exponential :meth:`expm` and real powers :meth:`pow`.

    Parameters
    ----------
    offset : numpy.ndarray
        the 1D translation vector. Defaults to the zero vector of length 2.
    scale : float
        the positive uniform scale. Defaults to 1.

    See Also
    --------
    Dliso : dilated isometry
    Dlt : dilatation with a different scale per dimension

    Examples
    --------
    >>> import numpy as np
    >>> from mt.geond.dilated_translation import Dltra
    >>> t = Dltra(offset=np.array([1., 2.]), scale=2.)
    >>> t
    Dltra(offset=[1. 2.], scale=2.0)
    >>> t << np.array([1., 1.])
    array([3., 4.])
    >>> t * t
    Dltra(offset=[3. 6.], scale=4.0)
    >>> t.matrix
    array([[2., 0., 1.],
           [0., 2., 2.],
           [0., 0., 1.]])
    >>> ~t
    Dltra(offset=[-0.5 -1. ], scale=0.5)
    >>> t.pow(2)
    Dltra(offset=[3. 6.], scale=4.0)

    References
    ----------
    .. [1] Pham et al, Distances and Means of Direct Similarities, IJCV, 2015. (not true but cheeky
       MT is trying to advertise his paper!)
    """

    # ----- base adaptation -----

    @property
    def dim(self):
        """Returns the dimension of the transformation, which is the length of the offset."""
        return self.offset.shape[0]

    @property
    def unitary(self):
        """The unitary matrix of the dilated translation, which is always the identity matrix.

        The setter raises `TypeError`.
        """
        return np.identity(self.ndim)  # returns identity

    @unitary.setter
    def unitary(self, unitary):
        raise TypeError("Unitary matrix is the identity matrix and is read-only.")

    # ----- derived properties -----

    @property
    def linear(self):
        """Returns the linear part of the affine transformation matrix of the dilated
        translation, which is the identity matrix multiplied by the scale."""
        return np.identity(self.ndim) * self.scale

    # ----- methods -----

    def __init__(self, offset=np.zeros(2), scale=1):
        self.offset = offset
        self.scale = scale

    def __repr__(self):
        return f"Dltra(offset={self.offset}, scale={self.scale})"

    # ----- base adaptation -----

    def multiply(self, other):
        if not isinstance(other, Dltra):
            return super(Dltra, self).multiply(other)
        return Dltra(self << other.offset, self.scale * other.scale)

    multiply.__doc__ = Dliso.multiply.__doc__

    def invert(self):
        invScale = 1 / self.scale
        return Dltra(self.offset * (-invScale), invScale)

    invert.__doc__ = Dliso.invert.__doc__

    # ----- matrix exponential -----

    def logm(self) -> tuple:
        """Returns the matrix logarithm of the transformation matrix.

        The logarithm of the homogeneous matrix is `[[u*I, v], [0, 0]]`, where `u = log(scale)` and
        `v = u/(scale-1) * offset` (or `v = offset` and `u = 0` when `scale == 1`). It is the
        inverse of :meth:`expm`.

        Returns
        -------
        u : float
            the linear part of the matrix logarithm in the form of a single scalar
        v : np.ndarray
            the translation part of the matrix logarithm in the form of a vector

        See Also
        --------
        expm : the inverse operation
        """
        if self.scale == 1:  # special case, deform to translation only
            return 0, self.offset
        u = math.log(self.scale)
        v = (u / (self.scale - 1)) * self.offset
        return u, v

    @classmethod
    def expm(cls, u: float, v: np.ndarray):
        """Returns the matrix exponential of a matrix living in the Lie algebra of Dltra.

        This is the inverse of :meth:`logm`: given `u` and `v` as returned by it, it returns the
        dilated translation with scale `exp(u)` and offset `(exp(u)-1)/u * v` (or offset `v` and
        scale 1 when `u == 0`).

        Parameters
        ----------
        u : float
            the linear part of the matrix in the form of a single scalar
        v : np.ndarray
            the translation part of the matrix in the form of a vector

        Returns
        -------
        Dltra
            the matrix exponential, a dilated translation

        See Also
        --------
        logm : the inverse operation

        Examples
        --------
        >>> import numpy as np
        >>> from mt.geond.dilated_translation import Dltra
        >>> t = Dltra(offset=np.array([1., 2.]), scale=2.)
        >>> u, v = t.logm()
        >>> round(u, 4)
        0.6931
        >>> Dltra.expm(u, v)
        Dltra(offset=[1. 2.], scale=2.0)
        """
        if u == 0:  # special case, deform to translation only
            return Dltra(offset=v)  # scale == 1
        s = math.exp(u)
        t = ((s - 1) / u) * v
        return Dltra(offset=t, scale=s)

    def pow(self, k: float):
        """Raises the Dltra to a scalar power, computed as `expm(k*u, k*v)` from `logm()`.

        Parameters
        ----------
        k : float
            the exponent. `pow(2)` equals `self*self`, `pow(0.5)` is a square root.

        Returns
        -------
        Dltra
            the power of the transformation
        """
        u, v = self.logm()
        return Dltra.expm(u * k, v * k)


# ----- casting -----


_bc.register_cast(Dltra, Dliso, lambda x: Dliso(offset=x.offset, scale=x.scale, unitary=x.unitary))
_bc.register_cast(
    Dltra,
    Dlt,
    lambda x: Dlt(offset=x.offset, scale=np.full(x.dim, x.scale), check_shapes=False),
)


# ----- approximation ------


def approx_Dliso_to_Dltra(obj):
    """Approximates a Dliso instance with a Dltra by ignoring the unitary part.

    It is registered as the approximation from :class:`Dliso` to :class:`Dltra`.

    Parameters
    ----------
    obj : Dliso
        a dilated isometry

    Returns
    -------
    Dltra
        a dilated translation with the same offset and scale
    """
    return Dltra(offset=obj.offset, scale=obj.scale)


register_approx(Dliso, Dltra, approx_Dliso_to_Dltra)


def approx_Dlt_to_Dltra(dlt):
    """Approximates a Dlt instance with a Dltra.

    We approximate the scale components with their arithmetic mean in this case for efficiency and
    numerical stability, breaking theoretical concerns. So use this function with care. It is
    registered as the approximation from :class:`Dlt` to :class:`Dltra`.

    Parameters
    ----------
    dlt : Dlt
        a dilatation

    Returns
    -------
    Dltra
        a dilated translation
    """
    return Dltra(offset=dlt.offset, scale=dlt.scale.mean())


register_approx(Dlt, Dltra, approx_Dlt_to_Dltra)


# ----- transform functions ------


def transform_Dltra_on_ndarray(dlt_tfm, point_array):
    """Transform an array of points using a dilated translation.

    It computes `point_array * scale + offset`. It is the function registered to apply a
    :class:`Dltra` to a `numpy.ndarray`.

    Parameters
    ----------
    dlt_tfm : Dltra
        a dilated translation
    point_array : numpy.ndarray
        a point array whose last dimension has the same length as the dimensionality of the
        transformation

    Returns
    -------
    numpy.ndarray
        dilated-translated point array, of the same shape as the input
    """
    return point_array * dlt_tfm.scale + dlt_tfm.bias


register_transform(Dltra, np.ndarray, transform_Dltra_on_ndarray)
register_transformable(Dltra, np.ndarray, lambda x, y: x.ndim == y.shape[-1])
