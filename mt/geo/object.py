'''The base class to represent a geometry object.'''


class GeometryObject(object):
    '''A geometry object which lives in a d-dimensional Euclidean space.

    A GeometryObject is a geometry object which lives in a d-dimensional Euclidean space. It can be a point, a point set, a polygon, a circle, a sphere, a parallelogram, a paralleloid, etc. This class represents the base class, in which the only property available is `ndim`, telling the number of dimensions. Other than that, the class is expected to have a static function called `convertible(obj)` which tells whether a given (probably geometry) object is convertible to it, and a static function called `from(obj)` to convert the object to an instance of the class.
    '''

    @property
    def ndim(self):
        '''Returns the number of dimensions in which the geometry object lives.'''
        raise NotImplementedError
    
    @staticmethod
    def convertible(obj):
        '''Checks whether an object is convertible into an instance of the class.'''
        raise NotImplementedError

    @staticmethod
    def from(obj):
        '''Converts an object into an instance of the class.'''
        raise NotImplementedError
