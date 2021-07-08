#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Implementation of plate with inclusion problem for Cosserat elasticity
in DOLFIN finite element problem solving environment.

Author: Jack S. Hale 2014 mail@jackhale.co.uk
"""

import dolfin as df
import numpy as np
import pandas as pd

from weak_form import constitutive_matrix, strain, constitutive_matrix_numpy

df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True

class InclusionProperties(df.Expression):
    def __init__(self, cell_function, params, **kwargs):
        assert(cell_function.dim() == 2)
        self.cell_function = cell_function

        inner = params['inner']
        self.D_inner = constitutive_matrix_numpy(inner['G'],
                                            inner['nu'],
                                            inner['l'],
                                            inner['N'])

        outer = params['outer']
        self.D_outer = constitutive_matrix_numpy(outer['G'],
                                            outer['nu'],
                                            outer['l'],
                                            outer['N'])

    def value_shape(self):
        return (6, 6)

    def eval_cell(self, value, x, ufc_cell):
        if self.cell_function[ufc_cell.index] == 0:
            value[:] = self.D_inner.flatten()
        elif self.cell_function[ufc_cell.index] == 1:
            value[:] = self.D_outer.flatten()
        else:
            raise ValueError("CellFunction with value not equal to 0 or 1")

def parameter_study():
    params = {}

    params['inner'] = {}
    params['outer'] = {}

    #1
    params['inner']['nu'] = 0.25
    params['inner']['l'] = 0.001
    params['inner']['N'] = 0.001
    params['inner']['G'] = 2.0*100.0

    params['outer']['nu'] = 1.0/3.0
    params['outer']['l'] = 0.001 
    params['outer']['N'] = 0.001
    params['outer']['G'] = 100.0
    
    params['L'] = 20.0 
    mesh_dir = "meshes/plate_with_circular_inclusion/" 
    params['output_dir'] = 'output/plate_with_circular_inclusion/'

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
    result = plate_with_inclusion(params) 
    print(params)
    print(result)
    
    #2 
    params['outer']['l'] = 1.0
    params['outer']['N'] = 0.9
    params['inner']['l'] = 0.001
    params['inner']['N'] = 0.001
    result = plate_with_inclusion(params) 
    print(params)
    print(result)

    #3
    params['outer']['l'] = 0.001
    params['outer']['N'] = 0.001
    params['inner']['l'] = 1.0
    params['inner']['N'] = 0.9
    result = plate_with_inclusion(params) 
    print(params)
    print(result)

    #4
    params['outer']['l'] = 0.75 
    params['outer']['N'] = 0.75
    params['inner']['l'] = 0.75
    params['inner']['N'] = 0.5
    result = plate_with_inclusion(params)
    print(params)
    print(result)

    #5
    params['outer']['l'] = 0.75 
    params['outer']['N'] = 0.75
    params['inner']['l'] = 0.1
    params['inner']['N'] = 0.75
    result = plate_with_inclusion(params) 
    print(params)
    print(result)

    #6
    params['outer']['l'] = 1.0 
    params['outer']['N'] = 0.75
    params['inner']['l'] = 0.5 
    params['inner']['N'] = 0.9
    result = plate_with_inclusion(params) 
    print(params)
    print(result)

def main():
    params = {}

    params['inner'] = {}
    params['outer'] = {}

    # These parameters are from Elena's email
    params['inner']['nu'] = 0.25
    params['inner']['l'] = 1.0 
    params['inner']['N'] = 0.9
    params['inner']['G'] = 0.5*100.0

    params['outer']['nu'] = 1.0/3.0
    params['outer']['l'] = 0.001 
    params['outer']['N'] = 0.001
    params['outer']['G'] = 100.0

    mesh_dir = "meshes/plate_with_circular_inclusion/"
    params['output_dir'] = 'output/plate_with_circular_inclusion/'

    params['mesh'] = df.Mesh(mesh_dir + "geometry.xml")
    print(params['mesh'].num_cells())
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
    
    results = plate_with_inclusion(params)
    print(params)
    print(results)


def bottom_boundary(x, on_boundary):
    return df.near(x[1], 0.0) and on_boundary


def left_boundary(x, on_boundary):
    return df.near(x[0], 0.0) and on_boundary


class RightBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], self.L) and on_boundary


def plate_with_inclusion(params):
    mesh = params['mesh']
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

    D_expr = InclusionProperties(cell_function=params['physical_regions'],
                                 params=params, degree=0)
    Q = df.TensorFunctionSpace(mesh, "DG", 0, shape=(6,6))
    D = df.interpolate(D_expr, Q)

    # Neumann boundary conditions; surface traction on right surface
    t = df.Constant((1.0, 0.0))
    right = RightBoundary()
    right.L = params['L']
    right.mark(params['facet_regions'], 2)
    ds = df.Measure("ds")(subdomain_data=params['facet_regions'])

    from dolfin import inner
    dx = df.Measure("dx")(subdomain_data=params['physical_regions'])
    # Bilinear form relating to internal deformation of Cosserat body
    a = inner(strain(v, eta), D*strain(u, psi))*dx
    # Linear form
    L = inner(t, v)*ds(2)

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
    solver.parameters["linear_solver"] = "mumps"

    solver.solve()
    u_h, psi_h = U_h.split()

    # Recover derived quantities of interest
    e = strain(u_h, psi_h)
    tau = D*e

    # Have to project derived stress quantity to L^2 type function space
    # This gives the best looking stress field across the interface.
    S = df.VectorFunctionSpace(mesh, "DG", 0, dim=6)
    Ptau = df.project(tau, V=S, solver_type='lu')

    results = {}

    # Find maximum stress by directly inspecting underlying vector
    # of projected stress function.
    results["maximum_stress"] = Ptau.sub(0, deepcopy=True).vector().max()
    results["minimum_stress"] = Ptau.sub(0, deepcopy=True).vector().min()

    # Warning: the maximum and minimum stress could well be slightly different
    # to the stresses evaluated at \theta = pi/2 and r=a.
    # In my experience the maximum_stress is a better match to the
    # stress concentration, but the minimum stress is better set
    # by the inclusion_stress.
    just_outside_interface = (0.0, 1.0001)
    just_inside_interface = (0.0, 0.9999)
    results["stress_concentration"] = Ptau.sub(0)(just_outside_interface)
    results["inclusion_stress"] = Ptau.sub(0)(just_inside_interface)

    results["hmin"] = mesh.hmin()
    results["hmax"] = mesh.hmax()

    df.File(params['output_dir'] + "mesh.pvd") << mesh
    df.File(params['output_dir'] + "facet_regions.pvd") << params['facet_regions']
    df.File(params['output_dir'] + "displacements.pvd") << u_h
    df.File(params['output_dir'] + "microrotation.pvd") << psi_h
    df.File(params['output_dir'] + "sigma_xx.pvd") << Ptau.sub(0)
    df.File(params['output_dir'] + "sigma_yy.pvd") << Ptau.sub(1)
    df.File(params['output_dir'] + "sigma_xy.pvd") << Ptau.sub(2)
    df.File(params['output_dir'] + "sigma_yx.pvd") << Ptau.sub(3)

    # Warning: It's unlikely the below will work in parallel!
    import scipy.constants
    thetas = np.linspace(0.0, scipy.constants.pi/2.0, num=50)
    xys = np.empty((len(thetas), 2))
    xys[:, 0] = np.cos(thetas)
    xys[:, 1] = np.sin(thetas)

    u_int = np.empty((len(xys), 2))
    t_int = np.empty((len(xys), 2))
    for i, xy in enumerate(xys):
        u_int[i, :] = u_h(xy)
        t_int[i, 0] = Ptau[0](xy)
        t_int[i, 1] = Ptau[1](xy)

    np.save(params['output_dir'] + "interface_points.npy", xys)
    np.save(params['output_dir'] + "interface_thetas.npy", thetas)
    np.save(params['output_dir'] + "interface_displacements.npy", u_int)
    np.save(params['output_dir'] + "interface_tractions.npy", t_int)

    return results


if __name__ == "__main__":
    main()
