'''The base class to represent a geometric object, plus mixins fixing its dimensionality.'''


__all__ = ['GeometricObject', 'TwoD', 'ThreeD']


class GeometricObject(object):
    '''A geometric object which lives in a d-dimensional Euclidean space.

    A GeometricObject is a geometric object which lives in a d-dimensional Euclidean space. It can
    be a point, a point set, a polygon, a circle, a sphere, a parallelogram, a paralleloid, etc.
    This class represents the base class, in which the only property available is `ndim`, telling
    the number of dimensions. The user needs to implement `ndim`, or use some mixin classes like
    :class:`TwoD` or :class:`ThreeD`.

    Examples
    --------
    >>> from mt.geo import GeometricObject, ThreeD
    >>> class Blob(ThreeD, GeometricObject):
    ...     pass
    >>> Blob().ndim
    3
    >>> GeometricObject().ndim
    Traceback (most recent call last):
        ...
    NotImplementedError
    '''
    
    # ----- properties -----
    
    @property
    def ndim(self):
        '''int: the number of dimensions of the Euclidean space in which the object lives.

        Raises
        ------
        NotImplementedError
            in the base class. Subclasses must override it or inherit a mixin that does.
        '''
        raise NotImplementedError


class TwoD(object):
    '''Mixin to assert that the geometric object lives in 2D Euclidean space.

    Put it before :class:`GeometricObject` in the list of base classes so that its `ndim` takes
    precedence, as in `class Foo(TwoD, GeometricObject)`.

    Examples
    --------
    >>> from mt.geo import TwoD
    >>> TwoD().ndim
    2
    '''

    @property
    def ndim(self):
        '''the dimensionality, which is 2'''
        return 2


class ThreeD(object):
    '''Mixin to assert that the geometric object lives in 3D Euclidean space.

    Put it before :class:`GeometricObject` in the list of base classes so that its `ndim` takes
    precedence, as in `class Foo(ThreeD, GeometricObject)`.

    Examples
    --------
    >>> from mt.geo import ThreeD
    >>> ThreeD().ndim
    3
    '''

    @property
    def ndim(self):
        '''the dimensionality, which is 3'''
        return 3
