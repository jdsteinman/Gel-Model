#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Implementation of plate with inclusion problem for elasticity
in DOLFIN finite element problem solving environment.

Author: Jack S. Hale 2014 mail@jackhale.co.uk
"""

import dolfin as df
import numpy as np

from weak_form import constitutive_matrix, strain
import analytical

def main():
    params = {}

    params['inner'] = {}
    params['outer'] = {}

    # Dependent parameters
    params['inner']['nu'] = 0.25
    params['inner']['mu'] = 1000.0

    params['outer']['nu'] = 1.0/3.0
    params['outer']['mu'] = 3.0*params['inner']['mu']

    # Derived parameters
    E = lambda mu, nu: 2.0*mu/(1.0 + nu)
    lmbda = lambda E, nu: (E*nu)/((1.0 + nu)*(1 - 2.0*nu))

    params['inner']['E'] = E(params['inner']['mu'], params['inner']['nu'])
    params['inner']['lambda'] = lmbda(params['inner']['E'],
                                      params['inner']['nu'])

    params['outer']['E'] = E(params['outer']['mu'], params['outer']['nu'])
    params['outer']['lambda'] = lmbda(params['outer']['E'],
                                      params['outer']['nu'])

    results = plate_with_inclusion(params)
    print results


def bottom_boundary(x, on_boundary):
    return df.near(x[1], 0.0) and on_boundary


def left_boundary(x, on_boundary):
    return df.near(x[0], 0.0) and on_boundary


class RightBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 20.0) and on_boundary


def plate_with_inclusion(params):
    mesh_dir = "meshes/plate_with_circular_inclusion/"
    params['mesh'] = df.Mesh(mesh_dir + "geometry.xml")

    params['mesh'] = df.Mesh(mesh_dir + "geometry.xml")
    # Inner cells - dx['0']
    # Outer cells - dx['1']
    params['physical_regions'] = \
        df.MeshFunction("size_t",
        params['mesh'],
        mesh_dir + "geometry_physical_region.xml")
    # Interface facets - ds['1']
    # Right facets - 2 (set later in code using SubDomain)
    params['facet_regions'] = \
        df.MeshFunction("size_t",
        params['mesh'],
        mesh_dir + "geometry_facet_region.xml")

    # Displacement space
    U = df.VectorFunctionSpace(params['mesh'], 'Lagrange', 2, dim=2)
    U_1, U_2 = U.split()

    u = df.TrialFunction(U)
    v = df.TestFunction(U)

    # Neumann boundary conditions; surface traction on right surface
    t = df.Constant((1.0, 0.0))
    right = RightBoundary()
    right.mark(params['facet_regions'], 2)
    ds = df.Measure("ds")(subdomain_data=params['facet_regions'])

    from dolfin import inner, grad, tr, Identity
    dx = df.Measure("dx")(subdomain_data=params['physical_regions'])

    def epsilon(v):
        return 0.5*(grad(v) + grad(v).T)

    def sigma(v, mu, lmbda):
        return 2.0*df.Constant(mu)*epsilon(v) + df.Constant(lmbda)*tr(epsilon(v))*Identity(params['mesh'].geometry().dim())

    a = inner(sigma(u, params['inner']['mu'], params['inner']['lambda']), epsilon(v))*dx(0) + \
        inner(sigma(u, params['outer']['mu'], params['outer']['lambda']), epsilon(v))*dx(1)
    # Linear form
    L = inner(t, v)*ds(2)

    # Dirichlet boundary conditions
    no_displacement = df.Constant(0.0)
    bc_bottom_U_2 = df.DirichletBC(U_2, no_displacement, bottom_boundary)
    bc_left_U_1 = df.DirichletBC(U_1, no_displacement, left_boundary)
    bcs = [bc_bottom_U_2, bc_left_U_1]

    u_h = df.Function(U)
    problem = df.LinearVariationalProblem(a, L, u_h, bcs=bcs)
    solver = df.LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "mumps"

    solver.solve()

    # Recover derived quantities of interest
    if params['inner']['mu'] > params['outer']['mu']:
        mu = params['inner']['mu']
        lmbda = params['inner']['lambda']
    else:
        mu = params['outer']['mu']
        lmbda = params['outer']['lambda']
    strain = epsilon(u_h)
    stress = sigma(u_h, mu, lmbda)

    # Have to project derived quantity to 1st order function space
    stress_xx = df.project(stress[0,0])
    on_interface = (0.0, 1.0)
    results = {}
    results["stress_concentration"] = stress_xx(on_interface)
    results["hmin"] = params['mesh'].hmin()
    results["hmax"] = params['mesh'].hmax()
    results["analytical_stress_concentration"] = \
        analytical.elastic_plate_with_inclusion.stress_concentration(params['inner']['mu']/params['outer']['mu'],
                                                                     params['inner']['nu'],
                                                                     params['outer']['nu'])

    return results


if __name__ == "__main__":
    main()
