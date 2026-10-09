'''Registry to store all the functions to upper-bound and lower-bound a geometric object of type A
with an object of type B.

An upper bound of an object is an object of type B whose interior contains the interior of the
original object (for example, a box enclosing an ellipsoid). A lower bound is an object of type B
whose interior lies inside the interior of the original object (for example, an ellipsoid inscribed
in a box). Each registry is keyed by the exact pair `(src_type, dst_type)`; lookup uses `type(obj)`
so subclasses do not inherit registrations.

Examples
--------
>>> from mt.geo import register_upper_bound, upper_bound, upper_boundable
>>> class Disc:
...     def __init__(self, radius):
...         self.radius = radius
>>> class Square:
...     def __init__(self, side):
...         self.side = side
>>> register_upper_bound(Disc, Square, lambda d: Square(2 * d.radius))
>>> upper_boundable(Disc(1), Square)
True
>>> upper_bound(Disc(1.5), Square).side
3.0
'''


__all__ = [
    'upper_bound', 'upper_boundable', 'register_upper_bound', 'register_upper_boundable',
    'lower_bound', 'lower_boundable', 'register_lower_bound', 'register_lower_boundable'
]


# ----- upper-bound functions -----


# map: (src_type, dst_type) -> upper-bound function
_upper_bound_registry = {}

_upper_boundable_registry = {}
# map: (src_type, dst_type) -> upper-boundable function


def upper_boundable(obj, dst_type):
    '''Checks if an object can be upper-bounded to a target type.

    If a function was registered with :func:`register_upper_boundable` for the key
    `(type(obj), dst_type)`, it is called with `obj` and its result is returned. Otherwise the
    object is deemed upper-boundable if and only if a function was registered with
    :func:`register_upper_bound` for that key. The lookup uses the exact type of `obj`.

    Parameters
    ----------
    obj : object
        object to upper-bound
    dst_type : type
        type or class to upper-bound to

    Returns
    -------
    bool
        whether or not the object is upper-boundable to `dst_type`. It is False, and no exception
        is raised, if nothing has been registered for the pair of types.

    See Also
    --------
    upper_bound : performs the bounding
    register_upper_boundable : registers a per-instance predicate
    register_upper_bound : registers a bounding function

    Examples
    --------
    >>> from mt.geo import upper_boundable
    >>> upper_boundable(3, str)
    False
    '''
    key = (type(obj), dst_type)
    if key in _upper_boundable_registry:
        return _upper_boundable_registry[key](obj)
    return key in _upper_bound_registry


def upper_bound(obj, dst_type):
    '''Upper-bounds an object to a target type.

    Calls the function registered with :func:`register_upper_bound` for the key
    `(type(obj), dst_type)`. The result is an object of type `dst_type` whose interior
    contains the interior of `obj`. The lookup uses the exact type of `obj`.

    Parameters
    ----------
    obj : object
        object to upper-bound
    dst_type : type
        type or class to upper-bound to

    Returns
    -------
    object
        the upper-bounded version of `obj`, as returned by the registered function

    Raises
    ------
    NotImplementedError
        if the upper-bound function has not been registered using function `register_upper_bound`

    See Also
    --------
    upper_boundable : checks whether an object can be upper-bounded
    register_upper_bound : registers a bounding function

    Examples
    --------
    >>> from mt.geo import upper_bound
    >>> upper_bound(3, str)
    Traceback (most recent call last):
        ...
    NotImplementedError: Upper-bound function from type <class 'int'> to ... not found.
    '''
    key = (type(obj), dst_type)
    if key in _upper_bound_registry:
        return _upper_bound_registry[key](obj)
    raise NotImplementedError(
        f"Upper-bound function from type {key[0]} to type {key[1]} not found."
    )


def register_upper_bound(src_type, dst_type, func):
    '''Registers a function to upper-bound an object from type A to type B.

    Parameters
    ----------
    src_type : type
        type or class to upper-bound from
    dst_type : type
        type or class to upper-bound to
    func : function
        upper-bound function `f(obj) -> obj2` to register, taking an instance of `src_type` and
        returning an instance of `dst_type`

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(src_type, dst_type)`

    See Also
    --------
    upper_bound : uses the registered function
    '''

    key = (src_type, dst_type)
    if key in _upper_bound_registry:
        raise SyntaxError(
            f"Upper-bound function from type {src_type} to "
            f"type {dst_type} has been registered before."
        )
    _upper_bound_registry[key] = func


def register_upper_boundable(src_type, dst_type, func):
    '''Registers a function to check if we can upper-bound an object from type A to type B.

    Parameters
    ----------
    src_type : type
        type or class to upper-bound from
    dst_type : type
        type or class to upper-bound to
    func : function
        upper-boundable function `f(obj) -> bool` to register, taking an instance of `src_type`

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(src_type, dst_type)`

    See Also
    --------
    upper_boundable : uses the registered function
    '''

    key = (src_type, dst_type)
    if key in _upper_boundable_registry:
        raise SyntaxError(
            f"Upper-boundable function from type {src_type} to "
            f"type {dst_type} has been registered before."
        )
    _upper_boundable_registry[key] = func


# ----- upper-bound functions -----


# map: (src_type, dst_type) -> lower-bound function
_lower_bound_registry = {}

_lower_boundable_registry = {}
# map: (src_type, dst_type) -> lower-boundable function


def lower_boundable(obj, dst_type):
    '''Checks if an object can be lower-bounded to a target type.

    If a function was registered with :func:`register_lower_boundable` for the key
    `(type(obj), dst_type)`, it is called with `obj` and its result is returned. Otherwise the
    object is deemed lower-boundable if and only if a function was registered with
    :func:`register_lower_bound` for that key. The lookup uses the exact type of `obj`.

    Parameters
    ----------
    obj : object
        object to lower-bound
    dst_type : type
        type or class to lower-bound to

    Returns
    -------
    bool
        whether or not the object is lower-boundable to `dst_type`. It is False, and no exception
        is raised, if nothing has been registered for the pair of types.

    See Also
    --------
    lower_bound : performs the bounding
    register_lower_boundable : registers a per-instance predicate
    register_lower_bound : registers a bounding function

    Examples
    --------
    >>> from mt.geo import lower_boundable
    >>> lower_boundable(3, str)
    False
    '''
    key = (type(obj), dst_type)
    if key in _lower_boundable_registry:
        return _lower_boundable_registry[key](obj)
    return key in _lower_bound_registry


def lower_bound(obj, dst_type):
    '''Lower-bounds an object to a target type.

    Calls the function registered with :func:`register_lower_bound` for the key
    `(type(obj), dst_type)`. The result is an object of type `dst_type` whose interior
    lies inside the interior of `obj`. The lookup uses the exact type of `obj`.

    Parameters
    ----------
    obj : object
        object to lower-bound
    dst_type : type
        type or class to lower-bound to

    Returns
    -------
    object
        the lower-bounded version of `obj`, as returned by the registered function

    Raises
    ------
    NotImplementedError
        if the lower-bound function has not been registered using function `register_lower_bound`

    See Also
    --------
    lower_boundable : checks whether an object can be lower-bounded
    register_lower_bound : registers a bounding function

    Examples
    --------
    >>> from mt.geo import lower_bound
    >>> lower_bound(3, str)
    Traceback (most recent call last):
        ...
    NotImplementedError: Lower-bound function from type <class 'int'> to ... not found.
    '''
    key = (type(obj), dst_type)
    if key in _lower_bound_registry:
        return _lower_bound_registry[key](obj)
    raise NotImplementedError(
        f"Lower-bound function from type {key[0]} to type {key[1]} not found."
    )


def register_lower_bound(src_type, dst_type, func):
    '''Registers a function to lower-bound an object from type A to type B.

    Parameters
    ----------
    src_type : type
        type or class to lower-bound from
    dst_type : type
        type or class to lower-bound to
    func : function
        lower-bound function `f(obj) -> obj2` to register, taking an instance of `src_type` and
        returning an instance of `dst_type`

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(src_type, dst_type)`

    See Also
    --------
    lower_bound : uses the registered function
    '''

    key = (src_type, dst_type)
    if key in _lower_bound_registry:
        raise SyntaxError(
            f"Lower-bound function from type {src_type} to "
            f"type {dst_type} has been registered before."
        )
    _lower_bound_registry[key] = func


def register_lower_boundable(src_type, dst_type, func):
    '''Registers a function to check if we can lower-bound an object from type A to type B.

    Parameters
    ----------
    src_type : type
        type or class to lower-bound from
    dst_type : type
        type or class to lower-bound to
    func : function
        lower-boundable function `f(obj) -> bool` to register, taking an instance of `src_type`

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(src_type, dst_type)`

    See Also
    --------
    lower_boundable : uses the registered function
    '''

    key = (src_type, dst_type)
    if key in _lower_boundable_registry:
        raise SyntaxError(
            f"Lower-boundable function from type {src_type} to "
            f"type {dst_type} has been registered before."
        )
    _lower_boundable_registry[key] = func
