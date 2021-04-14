#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Implementation of plate with hole problem for Cosserat elasticity
in DOLFIN finite element problem solving environment.

Author: Jack S. Hale 2014 mail@jackhale.co.uk
"""

import dolfin as df
import numpy as np

from weak_form import constitutive_matrix, strain


def main():
    mesh = df.Mesh("./meshes/plate_with_circular_hole/very_fine.xml")
    results = plate_with_hole(mesh, l=0.216/1.063, N=0.93)
    print("Stress concentration : %.4f" % results["stress_concentration"])


def bottom_boundary(x, on_boundary):
    return df.near(x[1], 0.0) and on_boundary


def left_boundary(x, on_boundary):
    return df.near(x[0], 0.0) and on_boundary


class TopBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 16.2) and on_boundary


def plate_with_hole(mesh, l, N, nu=0.3, G=1000.0):
    # Displacement space
    U = df.VectorElement('Lagrange', df.triangle, 2)
    # Micro-rotation space
    S = df.FiniteElement('Lagrange', df.triangle, 1)
    # Combined (problem) space
    V = df.FunctionSpace(mesh, df.MixedElement([U, S]))

    U, S = V.split()
    U_1, U_2 = U.split()

    u, psi = df.TrialFunctions(V)
    v, eta = df.TestFunctions(V)

    D = constitutive_matrix(G, nu, l, N)

    # Neumann boundary conditions; surface traction on top surface
    t = df.Constant((0.0, 1.0))
    boundary_parts = df.cpp.mesh.MeshFunctionSizet(mesh, mesh.topology().dim()-1)
    boundary_parts.set_all(0)
    top = TopBoundary()
    top.mark(boundary_parts, 1)

    ds = df.Measure("ds")(subdomain_data=boundary_parts)

    from dolfin import inner, dx
    # Bilinear form relating to internal deformation of Cosserat body
    a = inner(strain(v, eta), D*strain(u, psi))*dx
    # Linear form
    L = inner(t, v)*ds(1)

    # Dirichlet boundary conditions
    no_displacement = df.Constant(0.0)
    bc_bottom_U_2 = df.DirichletBC(U_2, no_displacement, bottom_boundary)
    bc_left_U_1 = df.DirichletBC(U_1, no_displacement, left_boundary)
    bc_bottom_S = df.DirichletBC(S, no_displacement, bottom_boundary)
    bc_left_S = df.DirichletBC(S, no_displacement, left_boundary)
    bcs = [bc_bottom_U_2, bc_left_U_1, bc_bottom_S, bc_left_S]

    U_h = df.Function(V)
    problem = df.LinearVariationalProblem(a, L, U_h, bcs=bcs)
    solver = df.LinearVariationalSolver(problem)

    solver.solve()
    u_h, psi_h = U_h.split()

    # Recover derived quantities of interest
    e = strain(u_h, psi_h)
    tau = D*e

    # Have to project derived quantity to 1st order function space
    Ptau_yy = df.project(tau[1])
    on_bottom_boundary = np.where(mesh.coordinates()[:, 1] == 0.0)
    on_circle = np.array([np.min(mesh.coordinates()[:, 0][on_bottom_boundary]),
                         0.0])
    results = {}
    results["stress_concentration"] = Ptau_yy(on_circle)
    results["l"] = l
    results["N"] = N
    results["hmin"] = mesh.hmin()
    results["hmax"] = mesh.hmax()

    disp_file = df.XDMFFile("./output/U_h.xdmf")
    u_h.rename("U_h","displacement")
    disp_file.write(u_h)

    return results


if __name__ == "__main__":
    main()
