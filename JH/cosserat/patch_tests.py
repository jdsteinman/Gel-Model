#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Implementation of patch tests for Cosserat elasticity
in DOLFIN finite element problem solving environment.

Author: Jack S. Hale 2014 mail@jackhale.co.uk
"""

import dolfin as df

from weak_form import constitutive_matrix, strain


def main():
    case = 1
    patch_test(case)


def all_boundary(x, on_boundary):
    return on_boundary


def solution_and_loading(case):
    """ Given a case return the analytical solution and loading
    as specified in Providas and Kattis p. 2066
    http://dx.doi.org/10.1016/S0045-7949(02)00262-6
    """
    if case == 1:
        U_0 = df.Expression(('x[0] + 0.5*x[1]', 'x[0] + x[1]', '0.25'))
        p = df.Constant((0.0, 0.0))
        q = df.Constant(0.0)
    elif case == 2:
        U_0 = df.Expression(('x[0] + 0.5*x[1]',
                             'x[0] + x[1]',
                             '0.25 + 0.25*alpha'), alpha=2.0)
        p = df.Constant((0.0, 0.0))
        q = df.Constant(1.0)
    elif case == 3:
        U_0 = df.Expression(('x[0] + 0.5*x[1]',
                             'x[0] + x[1]',
                             '0.25 + 0.5*alpha*(x[0] - x[1])'), alpha=2.0)
        p = df.Constant((1.0, 1.0))
        q = df.Expression('2.0*(x[0] - x[1])')
    else:
        raise ValueError('case must be integer between 1 and 3')

    return (U_0, p, q)


def patch_test(case):
    # Mesh specified in Providas and Kattis
    mesh = df.Mesh('meshes/patch_test.xml')

    # Displacement space
    U = df.VectorFunctionSpace(mesh, 'Lagrange', 2, dim=2)
    # Micro-rotation space
    S = df.FunctionSpace(mesh, 'Lagrange', 1)
    # Combined (problem) space
    V = df.MixedFunctionSpace([U, S])

    u, psi = df.TrialFunctions(V)
    v, eta = df.TestFunctions(V)

    # Material constants
    G = 1000.0
    nu = 0.25
    l = 0.1
    N = 0.5

    D = constitutive_matrix(G, nu, l, N)
    U_0, p, q = solution_and_loading(case=case)

    from dolfin import inner, dx
    # Bilinear form relating to internal deformation of Cosserat body
    a = inner(strain(v, eta), D*strain(u, psi))*dx
    # Linear form
    L = inner(p, v)*dx + inner(q, eta)*dx

    bc = df.DirichletBC(V, U_0, all_boundary)

    U_h = df.Function(V)
    problem = df.LinearVariationalProblem(a, L, U_h, bcs=bc)
    solver = df.LinearVariationalSolver(problem)

    solver.solve()
    u_h, psi_h = U_h.split()

    # Recover derived quantities of interest
    e = strain(u_h, psi_h)
    tau = D*e

if __name__ == '__main__':
    main()
