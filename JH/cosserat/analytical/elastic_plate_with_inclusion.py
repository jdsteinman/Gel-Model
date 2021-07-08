#!/usr/bin/env python
import numpy as np

def stress_concentration(g, nu_1, nu_2):
    """Calculates the stress concentration at r=R theta=pi/2
    for a circular inclusion in a plane
    strain elastic body. This is from the book
    Fundamentals of Rock Mechanics by Jaeger et al. on p. 228

    g - Shear Modulus Ratio g_1/g_2 (g is \beta in book)
    nu_1 - Poisson's ratio inclusion
    nu_2 - Poisson's ratio matrix
    """

    # Plane strain conditions
    # Usually kappa but shortened to k
    k_1 = 3.0 - 4.0*nu_1
    k_2 = 3.0 - 4.0*nu_2

    # eq. 8.181
    B = ((k_1 - 1.0) - g*(k_2 - 1.0))/(2.0*g + (k_1 - 1.0))
    C = (g - 1.0)/(g*k_2 + 1.0)

    # eq. 8.187 evaluated for sigma_2^\inf = 0, sigma_2^\inf = 1
    # and theta = pi/2 and r = 1.0
    stress_2 = 0.5*(1.0 + B) + 0.5*(1.0 - 3.0*C)
    # Stress in inclusion is always constant
    stress_1 = ((g*(k_2 + 2.0) + k_1)*g*(k_2 + 1.0))/ \
        (2.0*(2.0*g + k_1 - 1.0)*(g*k_2 + 1.0))

    # If inclusion is stiffer than matrix then
    # highest stress is in inclusion
    if g > 1.0:
        return stress_1
    # Otherwise stress is highest in the matrix
    else:
        return stress_2

if __name__ == "__main__":
    print stress_concentration(2.0, 0.25, 1.0/3.0)
