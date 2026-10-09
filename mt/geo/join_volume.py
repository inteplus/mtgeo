'''Registry to store all the functions to join the interior volumes of two geometric objects in the
same space.

The registry is keyed by the ordered pair of exact types `(type1, type2)`; the pair `(B, A)` is not
looked up when only `(A, B)` is registered.

Examples
--------
>>> from mt.geo import join_volume, register_join_volume
>>> class Segment:
...     def __init__(self, a, b):
...         self.a, self.b = a, b
>>> def join_segments(s1, s2):
...     inter = max(0.0, min(s1.b, s2.b) - max(s1.a, s2.a))
...     len1, len2 = s1.b - s1.a, s2.b - s2.a
...     return inter, len1 - inter, len2 - inter, len1 + len2 - inter
>>> register_join_volume(Segment, Segment, join_segments)
>>> join_volume(Segment(0, 2), Segment(1, 4))
(1, 1, 2, 4)
'''


__all__ = ['join_volume', 'register_join_volume']


# map: (type1, type2) -> join_volume function
_join_volume_registry = {}


def join_volume(obj1, obj2):
    '''Joins the interior volumes of two geometric objects in the same space.

    This is the building block for intersection-over-union style measures. It calls the function
    registered with :func:`register_join_volume` for the key `(type(obj1), type(obj2))`. In 1D the
    "volume" is a length, in 2D an area, in 3D a volume, and so on.

    Parameters
    ----------
    obj1 : object
        first object to join
    obj2 : object
        second object to join

    Returns
    -------
    intersection_volume : float
        the volume of the intersection of the two objects' interior regions
    obj1_only_volume : float
        the volume of the interior of obj1 that does not belong to obj2
    obj2_only_volume : float
        the volume of the interior of obj2 that does not belong to obj1
    union_volume : float
        the volume of the union of the two objects' interior regions

    Raises
    ------
    NotImplementedError
        if the join_volume function has not been registered using function `register_join_volume`

    See Also
    --------
    register_join_volume : registers a join function
    '''
    key = (type(obj1), type(obj2))
    if key in _join_volume_registry:
        return _join_volume_registry[key](obj1, obj2)
    raise NotImplementedError(
        f"Join_volume function between type {key[0]} and type {key[1]} not found."
    )


def register_join_volume(type1, type2, func):
    '''Registers a function to join_volume an object of type A and object of type B.

    Parameters
    ----------
    type1 : type
        type or class of the first object to join_volume
    type2 : type
        type or class of the second object to join_volume
    func : function
        join_volume function `f(obj1, obj2) -> (intersection, obj1_only, obj2_only, union)` to
        register. See :func:`join_volume` for the meaning of the 4 returned volumes.

    Raises
    ------
    SyntaxError
        if a function has already been registered for the pair `(type1, type2)`

    See Also
    --------
    join_volume : uses the registered function
    '''

    key = (type1, type2)
    if key in _join_volume_registry:
        raise SyntaxError(
            f"Join_volume function between type {type1} and "
            f"type {type2} has been registered before."
        )
    _join_volume_registry[key] = func
