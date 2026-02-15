#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Vector algebra using object-oriented Python.
...
License: MIT
"""

import math
import cmath

class Vector3D:
    """Class representing a real-valued 3D Cartesian vector."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """Initialise vector components."""
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Readable representation."""
        return f"Vector3D({self.x}, {self.y}, {self.z})"

    def __abs__(self):
        """Return the magnitude of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other):
        """Vector addition."""
        return self.__class__(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Vector subtraction."""
        return self.__class__(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return self.__class__( - self.x, - self.y, - self.z)

    def __mul__(self, scalar):
        return self.__class__(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return self.__class__(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        """Dot (scalar) product."""
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other):
        """Cross (vector) product."""
        return self.__class__(self.y*other.z - self.z*other.y,
                              self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)


class ComplexVector3D(Vector3D):
    """3D vector with complex-valued components."""

    def __init__(self, x=0+0j, y=0+0j, z=0+0j):
        super().__init__(x, y, z)

    def __abs__(self):
        """Return the magnitude of the vector."""
        return math.sqrt(self.dot(self).real)

    def dot(self, other):
        """Complex dot product"""
        return self.x.conjugate()*other.x + self.y.conjugate()*other.y + self.z.conjugate()*other.z


class PlaneWaveField():
    """Plane-wave vector field A exp(i k·x)."""

    def __init__(self, k: ComplexVector3D, amplitude: ComplexVector3D):
        self.k = k
        self.A = amplitude

    def value(self, x: Vector3D):
        """Return value of the field at x."""
        return self.A * cmath.exp(1j * self.k.dot(x))

    def _d_dx(self, x: Vector3D, h):
        """Evaluate central differences at x in x direction."""
        return (self.value(Vector3D(x.x + h, x.y, x.z))
                - self.value(Vector3D(x.x - h, x.y, x.z))) * (1/(2*h))

    def _d_dy(self, x: Vector3D, h):
        """Evaluate central differences at x in y direction."""
        return (self.value(Vector3D(x.x, x.y + h, x.z))
                - self.value(Vector3D(x.x, x.y - h, x.z))) * (1/(2*h))

    def _d_dz(self, x: Vector3D, h):
        """Evaluate central differences at x in z direction."""
        return (self.value(Vector3D(x.x, x.y, x.z + h))
                - self.value(Vector3D(x.x, x.y, x.z - h))) * (1/(2*h))

    def divergence(self, x: Vector3D, h=1e-5):
        """Evaluate divergence, using central differences."""
        return self._d_dx(x, h).x + self._d_dy(x, h).y + self._d_dz(x, h).z

    def curl(self, x: Vector3D, h=1e-5):
        """Evaluate the curl, using central differences."""
        dF_dx = self._d_dx(x, h)
        dF_dy = self._d_dy(x, h)
        dF_dz = self._d_dz(x, h)
        return ComplexVector3D(dF_dy.z - dF_dz.y, dF_dz.x - dF_dx.z, dF_dx.y - dF_dy.x)


def triangle_area(a, b, c):
    """Return area of triangle defined by points a, b, c."""
    ab = b - a
    ac = c - a
    return 0.5 * abs(ab.cross(ac))

def angle_between(u, v):
    """Return angle (radians) between vectors u and v."""
    cos_of_angle = u.dot(v) / (abs(u) * abs(v))
    return math.acos(cos_of_angle)


if __name__ == '__main__':

    #Define triangles on assignment sheet.
    triangles = [
        (Vector3D(0,0,0), Vector3D(1,0,0), Vector3D(0,1,0)),
        (Vector3D(-1,-1,-1), Vector3D(0,-1,-1), Vector3D(-1,0,-1)),
        (Vector3D(1,0,0), Vector3D(0,0,1), Vector3D(0,0,0)),
        (Vector3D(0,0,0), Vector3D(1,-1,0), Vector3D(0,0,1))
    ]

    #Evaluate area of triangles (2a).
    for i, (a, b, c) in enumerate(triangles, start=1):
        area = triangle_area(a, b, c)
        print(f"Triangle {i} : area = {area:.5f}")

    #Evaluate internal angles of triangles (2b)
    for i, (a, b, c) in enumerate(triangles, start=1):
        ab, ac, bc = b - a, c - a, c - b
        angles = [angle_between(ab, ac), angle_between(-ab, bc),angle_between(-ac, -bc)]
        angles_deg = [math.degrees(a) for a in angles]
        print(f"Triangle {i} angles (deg): {angles_deg}")



    #Instanciate Hansen vectors
    k = ComplexVector3D(0, 0, math.pi)
    M = PlaneWaveField(k, ComplexVector3D(1, 0, 0))
    N = PlaneWaveField(k, ComplexVector3D(0, 1j, 0))

    #Define test points
    test_points = [
    Vector3D(0.1,  0.2,  0.3),
    Vector3D(-0.4, 0.1,  0.7),
    Vector3D(0.25, -0.6, 0.9),
    Vector3D(1.1,  0.3, -0.2),
    Vector3D(-0.8, -0.4, 0.5)
    ]

    #Test Maxwell equations
    for i, x in enumerate(test_points, start=1):
        print(f"Testing point {i}: x = {x}")
        print(f"∇·M = {M.divergence(x):.3e}")
        print(f"∇·N = {N.divergence(x):.3e}")
        print(f"|∇×N − M*|k|| = {abs(N.curl(x) - (M.value(x) * abs(k))):.3e}")
        print(f"|∇×M − N*|k|| = {abs(M.curl(x) - (N.value(x) * abs(k))):.3e}")
        