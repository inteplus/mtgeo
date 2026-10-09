'''Registry to store all the approximations functions to approximate a geometric object from type A
to type B.

The registry is keyed by the exact pair `(src_type, dst_type)`. Libraries register a function via
:func:`register_approx` (and optionally :func:`register_approximable`, a predicate telling whether a
particular instance can be approximated), and users call :func:`approx` and :func:`approximable`
without knowing which function is behind them. Lookup uses `type(obj)` exactly, so subclasses do
not inherit the registrations of their base classes.

Examples
--------
>>> from mt.geo import approx, approximable, register_approx
>>> class Square:
...     def __init__(self, side):
...         self.side = side
>>> class Circle:
...     def __init__(self, radius):
...         self.radius = radius
>>> register_approx(Square, Circle, lambda sq: Circle(sq.side / 2))
>>> approximable(Square(2), Circle)
True
>>> approximable(Circle(1), Square)
False
>>> approx(Square(2), Circle).radius
1.0
'''


__all__ = ['approx', 'approximable', 'register_approx', 'register_approximable']


# map: (src_type, dst_type) -> approx function
_approx_registry = {}

_approximable_registry = {}
# map: (src_type, dst_type) -> approximable function


def approximable(obj, dst_type):
    '''Checks if an object can be approximated to a target type.

    If a function was registered with :func:`register_approximable` for the key
    `(type(obj), dst_type)`, it is called with `obj` and its result is returned. Otherwise the
    object is deemed approximable if and only if an approx function was registered with
    :func:`register_approx` for that key. The lookup uses the exact type of `obj`.

    Parameters
    ----------
    obj : object
        object to approximate
    dst_type : type
        type or class to approx to

    Returns
    -------
    bool
        whether or not the object is approximable to `dst_type`. It is False, and no exception is
        raised, if nothing has been registered for the pair of types.

    See Also
    --------
    approx : performs the approximation
    register_approximable : registers a per-instance predicate
    register_approx : registers an approx function

    Examples
    --------
    >>> from mt.geo import approximable
    >>> approximable(3, str)
    False
    '''
    key = (type(obj), dst_type)
    if key in _approximable_registry:
        return _approximable_registry[key](obj)
    return key in _approx_registry


def approx(obj, dst_type):
    '''Approximates an object to a target type.

    Calls the function registered with :func:`register_approx` for the key
    `(type(obj), dst_type)`. The lookup uses the exact type of `obj`.

    Parameters
    ----------
    obj : object
        object to approximate
    dst_type : type
        type or class to approx to

    Returns
    -------
    object
        the approx version of `obj`, as returned by the registered function

    Raises
    ------
    NotImplementedError
        if the approx function has not been registered using function `register_approx`

    See Also
    --------
    approximable : checks whether an object can be approximated
    register_approx : registers an approx function

    Examples
    --------
    >>> from mt.geo import approx
    >>> approx(3, str)
    Traceback (most recent call last):
        ...
    NotImplementedError: Approximate function from type <class 'int'> to ... not found.
    '''
    key = (type(obj), dst_type)
    if key in _approx_registry:
        return _approx_registry[key](obj)
    raise NotImplementedError(
        f"Approximate function from type {key[0]} to type {key[1]} not found."
    )


def register_approx(src_type, dst_type, func):
    '''Registers a function to approximate an object from type A to type B.

    Parameters
    ----------
    src_type : type
        type or class to approximate from
    dst_type : type
        type or class to approximate to
    func : function
        approx function `f(obj) -> obj2` to register, taking an instance of `src_type` and
        returning an instance of `dst_type`

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(src_type, dst_type)`

    See Also
    --------
    approx : uses the registered function
    '''

    key = (src_type, dst_type)
    if key in _approx_registry:
        raise SyntaxError(
            f"Approximate function from type {src_type} to "
            f"type {dst_type} has been registered before."
        )
    _approx_registry[key] = func


def register_approximable(src_type, dst_type, func):
    '''Registers a function to check if we can approximate an object from type A to type B.

    Parameters
    ----------
    src_type : type
        type or class to approximate from
    dst_type : type
        type or class to approximate to
    func : function
        approximable function `f(obj) -> bool` to register, taking an instance of `src_type`

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(src_type, dst_type)`

    See Also
    --------
    approximable : uses the registered function
    '''

    key = (src_type, dst_type)
    if key in _approximable_registry:
        raise SyntaxError(
            f"Approximable function from type {src_type} to "
            f"type {dst_type} has been registered before."
        )
    _approximable_registry[key] = func
