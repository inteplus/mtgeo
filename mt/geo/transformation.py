'''Base tools to represent a geometric transformer and the act of transforming geometric objects
(transformation).

A transformer (for example an affine map) is an object that can be applied to geometric objects of
various types. The function to apply a transformer of type T to an object of type O is registered
with :func:`register_transform` and invoked via :func:`transform` or, more conveniently, the `<<`
operator provided by the :class:`Transformer` mixin. The lookup walks through the base classes of
the transformer in method resolution order, so a function registered for a base transformer class
also serves its subclasses, whereas the type of the object must match exactly.

Examples
--------
>>> from mt.geo import LieTransformer, register_transform
>>> class Scale(LieTransformer):
...     def __init__(self, k):
...         self.k = k
...     def invert(self):
...         return Scale(1 / self.k)
...     def multiply(self, other):
...         return Scale(self.k * other.k)
>>> register_transform(Scale, float, lambda tfm, x: tfm.k * x)
>>> Scale(2.0) << 3.0
6.0
>>> Scale(2.0) >> 3.0
1.5
>>> (Scale(2.0) * Scale(3.0)).k
6.0
>>> (~Scale(4.0)).k
0.25
'''


from inspect import getmro

from .object import GeometricObject


__all__ = [
    'transform', 'transformable', 'register_transform', 'register_transformable',
    'Transformer', 'InvertibleTransformer', 'LieTransformer'
]


# map: (tfm_type, obj_type) -> transform function
_transform_registry = {}

_transformable_registry = {}
# map: (tfm_type, obj_type) -> transformable function


def transformable(tfm, obj):
    '''Checks if an object can be transformed using a transformer.

    The base classes of `type(tfm)` are visited in method resolution order. For each one, if a
    function was registered with :func:`register_transformable` for the key
    `(base class, type(obj))`, it is called as `f(tfm, obj)` and its result is returned; else if a
    function was registered with :func:`register_transform` for that key, True is returned. If no
    base class matches, False is returned.

    Parameters
    ----------
    tfm : Transformer
        a transformer
    obj : object
        object to transform from

    Returns
    -------
    bool
        whether or not the object is transformable using the transformer. It is False, and no
        exception is raised, if nothing has been registered.

    See Also
    --------
    transform : performs the transformation
    register_transformable : registers a per-instance predicate

    Examples
    --------
    >>> from mt.geo import Transformer, transformable
    >>> transformable(Transformer(), 3.0)
    False
    '''
    tfm_type = type(tfm)
    obj_type = type(obj)
    for cur_type in getmro(tfm_type): # go through every base class, in method resolution order
        key = (cur_type, obj_type)
        if key in _transformable_registry:
            return _transformable_registry[key](tfm, obj)
        if key in _transform_registry:
            return True
    return False


def transform(tfm, obj):
    '''Transforms an object using a given transformer.

    The base classes of `type(tfm)` are visited in method resolution order and the first function
    registered with :func:`register_transform` for `(base class, type(obj))` is called as
    `f(tfm, obj)`. Unlike the `<<` operator, this function does not consult the functions
    registered with :func:`register_transformable`.

    Parameters
    ----------
    tfm : Transformer
        the transformer. It is typically an instance of a class using the :class:`Transformer`
        mixin.
    obj : object
        object to transform from

    Returns
    -------
    object
        the transformed version of `obj`, as returned by the registered function

    Raises
    ------
    NotImplementedError
        if no transform function has been registered, using function `register_transform`, for the
        transformer and the object

    See Also
    --------
    transformable : checks whether an object can be transformed
    register_transform : registers a transform function

    Examples
    --------
    >>> from mt.geo import Transformer, transform
    >>> transform(Transformer(), 3.0)
    Traceback (most recent call last):
        ...
    NotImplementedError: Transformer ... has no function to transform object 3.0.
    '''
    obj_type = type(obj)
    for cur_type in getmro(type(tfm)): # go through every base class, in method resolution order
        key = (cur_type, obj_type)
        if key in _transform_registry:
            return _transform_registry[key](tfm, obj)
    raise NotImplementedError(
        f"Transformer {tfm} has no function to transform object {obj}."
    )


def register_transform(tfm_type, obj_type, func):
    '''Registers a function to transform an object of a given type using a transformer of a given
    type.

    Parameters
    ----------
    tfm_type : type
        type or class of the transformer. Subclasses of it are served by the same function.
    obj_type : type
        type or class of the object to transform from. It is matched exactly.
    func : function
        transform function `f(tfm, obj) -> obj` to register, taking a transformer instance and
        an object instance and returning the transformed object

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(tfm_type, obj_type)`

    See Also
    --------
    transform : uses the registered function
    '''

    key = (tfm_type, obj_type)
    if key in _transform_registry:
        raise SyntaxError(
            f"Transform function using transformer type {tfm_type} applying to object type "
            f"{obj_type} has been registered before."
        )
    _transform_registry[key] = func


def register_transformable(tfm_type, obj_type, func):
    '''Registers a function to check if we can transform an object of a given type using a
    transformer of another given type.

    Parameters
    ----------
    tfm_type : type
        type or class of the transformer. Subclasses of it are served by the same function.
    obj_type : type
        type or class of the object to transform from. It is matched exactly.
    func : function
        transformable function `f(tfm, obj) -> bool` to register, taking a transformer instance
        and an object instance, for instance to check that their dimensions agree

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(tfm_type, obj_type)`

    See Also
    --------
    transformable : uses the registered function
    '''

    key = (tfm_type, obj_type)
    if key in _transformable_registry:
        raise SyntaxError(
            f"Transformable function using transformer type {tfm_type} applying to object type "
            f"{obj_type} has been registered before."
        )
    _transformable_registry[key] = func



class Transformer(object):
    '''A mixin asserting that this is a class of transformers.

    A transformer is an object capable of transforming geometric objects of certain types into other
    geometric objects. It is usually parametrised by a transformation matrix. The mixin assumes that
    the act of transforming an object using the inheriting class is conducted via calling the global
    :func:`transform` function. The mixin overloads the left shift operator, so that `tfm << obj`
    transforms `obj` by `tfm`.

    See Also
    --------
    InvertibleTransformer : adds inversion and the right shift operator
    LieTransformer : adds multiplication

    Examples
    --------
    >>> from mt.geo import Transformer, register_transform
    >>> class Negate(Transformer):
    ...     pass
    >>> register_transform(Negate, float, lambda tfm, x: -x)
    >>> Negate() << 3.0
    -3.0
    >>> Negate() << 'abc'
    Traceback (most recent call last):
        ...
    ValueError: Object abc cannot by transformed by transformer type <class '...Negate'>
    '''
    
    # ----- operators -----
    
    def __lshift__(self, obj):
        '''Transforms an object or raise a ValueError if the object is not transformable by the
        transformer.

        Parameters
        ----------
        obj : object
            object to transform

        Returns
        -------
        object
            `transform(self, obj)`

        Raises
        ------
        ValueError
            if `transformable(self, obj)` is False
        '''
        if not transformable(self, obj):
            raise ValueError(
                f"Object {obj} cannot by transformed by transformer type {type(self)}"
            )
        return transform(self, obj)


class InvertibleTransformer(Transformer):
    '''A mixin asserting that this is a class of invertible transformers. 

    The mixin assumes that a function `self.invert()` to return the inverted transformer has been
    implemented. It further overloads the right shift operator and the inverse operator: `~tfm` is
    the inverted transformer and `tfm >> obj` is `(~tfm) << obj`, which undoes `tfm << obj`.

    See Also
    --------
    Transformer : the base mixin
    LieTransformer : adds multiplication
    '''

    # ----- abstract -----
    
    def invert(self):
        '''Returns the inverse of the transformer. Subclasses must implement it.

        Returns
        -------
        InvertibleTransformer
            the inverted transformer

        Raises
        ------
        NotImplementedError
            in the mixin. Subclasses must override it.
        '''
        raise NotImplementedError(f"You must implement the function {type(self)}.invert().")

    # ----- operators -----
    
    def __rshift__(self, obj):
        '''Inversely transforms an object or raises a ValueError if the object is not transformable
        by the inverted transformer.

        Parameters
        ----------
        obj : object
            object to transform inversely

        Returns
        -------
        object
            `(~self) << obj`
        '''
        return (~self) << obj

    def __invert__(self):
        '''Returns the inverted transformer, invoking the function `self.invert()`.'''
        return self.invert()

class LieTransformer(InvertibleTransformer):
    '''A mixin asserting that this is a class of transformers living in a Lie group, supporting the
    associative multiplication operator and the inversion operator.

    Beside being an invertible transformer, the mixin additionally assumes that a function
    `self.multiply(other)` implementing the multiplication operator is available. It further
    overloads the multiplication, true division and modular operators, and provides a conjugate
    function. For transformers `a` and `b`, `a*b` is `a.multiply(b)`, `a/b` is `a*(~b)` and `a%b`
    is `(~a)*b`.

    See Also
    --------
    InvertibleTransformer : the base mixin
    '''
    
    # ----- abstract -----
    
    def multiply(self, other):
        '''Multiplies the transformer with another transformer. Subclasses must implement it.

        Parameters
        ----------
        other : LieTransformer
            the other transformer, usually of the same type

        Returns
        -------
        LieTransformer
            the product `self*other`

        Raises
        ------
        NotImplementedError
            in the mixin. Subclasses must override it.
        '''
        raise NotImplementedError(
            f"You must implement the function {type(self)}.multiply()."
        )

    # ----- operators -----
    
    def __mul__(self, other):
        '''a*b = Lie operator'''
        return self.multiply(other)

    def __truediv__(self, other):
        '''a/b = a*(~b)'''
        return self*(~other)

    def __mod__(self, other):
        '''a%b = (~a)*b'''
        return (~self)*other

    def conjugate(self, other):
        '''Conjugate: `self.conjugate(other) = self*other*self^{-1}`

        Parameters
        ----------
        other : LieTransformer
            the transformer to conjugate

        Returns
        -------
        LieTransformer
            `self*(other/self)`
        '''
        return self*(other/self)
