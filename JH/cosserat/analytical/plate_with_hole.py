#!/usr/bin/env python

from scipy.special import kn


def stress_concentration(nu, R, l, N):
    """Calculates the stress concentration at r=R theta=pi/2
    for a circular hole in a plane stress or plane
    strain Cosserat body. This is a based on the
    Mathematica solution provided by Elena Atroshchenko.
    A full stress field is also available but not
    yet implemented.

    nu - Poisson's ratio
    R - Radius of the circular inclusion
    l - Intrinsic length scale
    N - Coupling parameter
    """

    # Dependant parameters
    b = l
    c = l/N

    F = 8.0 * (1.0 - nu) * (b**2/c**2) * \
              (1.0/(4.0 + R**2/c**2 +
                   (2.0 * R/c) * (kn(0, R/c)/kn(1, R/c))))

    stress_concentration = (3.0 + F)/(1.0 + F)

    return stress_concentration

if __name__ == "__main__":
    print stress_concentration(0.3, 0.216, 0.216/1.063, 0.93)
