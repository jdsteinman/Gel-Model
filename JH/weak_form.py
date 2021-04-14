#!/usr/bin/python
# -*- coding: utf-8 -*-

import dolfin as df
import numpy as np


def constitutive_matrix_numpy(G, nu, l, N):
    # From Elena Atroshchenko's presentation slide 7
    D_numpy = np.array([
                       [
                           (2.0*(1.0 - nu))/(1.0 - 2.0*nu),
                           (2.0*nu)/(1.0 - 2.0 * nu),
                           0.0,
                           0.0,
                           0.0,
                           0.0,
                       ],
                       [
                           (2.0*nu)/(1.0 - 2.0*nu),
                           (2.0*(1.0 - nu)) / (1.0 - 2.0*nu),
                           0.0,
                           0.0,
                           0.0,
                           0.0,
                       ],
                       [
                           0.0,
                           0.0,
                           1.0/(1.0 - N**2),
                           (1.0 - 2.0*N**2)/(1.0 - N**2),
                           0.0,
                           0.0,
                       ],
                       [
                           0.0,
                           0.0,
                           (1.0 - 2.0*N**2)/(1.0 - N**2),
                           1.0/(1.0 - N**2),
                           0.0,
                           0.0,
                       ],
                       [
                           0.0,
                           0.0,
                           0.0,
                           0.0,
                           4.0*l**2,
                           0.0,
                       ],
                       [
                           0.0,
                           0.0,
                           0.0,
                           0.0,
                           0.0,
                           4.0*l**2,
                       ],
                       ])
    D_numpy *= G

    return D_numpy


def constitutive_matrix(G, nu, l, N):
    D = df.Constant(constitutive_matrix_numpy(G, nu, l, N))
    return D


def strain(v, eta):
    """ Returns a vector of strains of size (6,1) in the Voigt notation
    layout {eps_xx, eps_yy, eps_xy, eps_yx, chi_xz, chi_yz} """

    # From Elena Atroshchenko's presentation slide 8
    strain = df.as_vector([
                          v[0].dx(0),
                          v[1].dx(1),
                          v[1].dx(0) - eta,
                          v[0].dx(1) + eta,
                          eta.dx(0),
                          eta.dx(1),
                          ])

    return strain
