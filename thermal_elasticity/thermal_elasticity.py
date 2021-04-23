import dolfin as df
import numpy as np
import sys
import os
import time

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

"""
Written by: John Steinman
"""

class shearModulus(df.UserExpression):
    def __init__(self, cell_function, params, **kwargs):
        assert(cell_function.dim()==3)
        self.cell_function = cell_function
        self.box = params["box"]["mu"]
        self.sphere = params["sphere"]["mu"]
        self.rod = params["rod"]["mu"]

        super().__init__(**kwargs)

    def value_shape(self):
        return ()

    def eval_cell(self, value, x, ufl_cell):
        if self.cell_function[ufl_cell.index] == 301:
            value[0] = self.box
        elif self.cell_function[ufl_cell.index] == 302:
            value[0] =self.sphere
        elif self.cell_function[ufl_cell.index] == 303:
            value[0] =self.rod
        else:
            raise ValueError("CellFunction with bad value")

def main():

    params = {}

    params['inner'] = {}
    params['outer'] = {}

    params['inner']['mu'] = 325e12
    params['outer']['mu'] = 325e24

    params['output_folder'] = './output/'

    params['mesh'] = df.Mesh("./meshes/thermal_elasticity.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/thermal_elasticity_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/thermal_elasticity_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, det, tr, grad, Identity, inner, sym

    # Mesh
    mesh = params["mesh"]

    domains = params["physical_region"]
    boundaries = params["facet_region"]

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Thermal Problem
    VT = df.FunctionSpace(mesh, "CG", 1)
    T_, dT = df.TestFunction(VT), df.TrialFunction(VT)
    Delta_T = df.Function(VT, name="Temperature increase")
    aT = dot(grad(dT), grad(T_))*dx(301)+dot(grad(dT), grad(T_))*dx(302)+dot(grad(dT), grad(T_))*dx(303)
    LT = df.Constant(0)*T_*dx

    bcT = [df.DirichletBC(VT, df.Constant(-50.), boundaries, 203)]

    df.solve(aT == LT, Delta_T, bcT)

    # Mechanical Problem
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    # Kinematics
    E = df.Constant(50e3)
    nu = df.Constant(0.2)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    alpha = df.Constant(1e-5)

    f = df.Constant((0, 0))

    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(3) + 2.0*mu*eps(v)

    Vu = df.VectorFunctionSpace(mesh, 'CG', 2)
    du = df.TrialFunction(Vu)
    u_ = df.TestFunction(Vu)
    Wint = inner(sigma(du, Delta_T), eps(u_))*dx(301)
    aM = df.lhs(Wint)
    LM = df.rhs(Wint) + inner(f, u_)*dx

    bcu = DirichletBC(Vu, Constant((0., 0.)), lateral_sides)

    u = Function(Vu, name="Displacement")

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    bcs = [outer_bc]

    # Create linear variational problem and solve
    problem = LinearVariationalProblem(a, L, u, bcs=bcs)
    solver = LinearVariationalSolver(problem)

    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    # Projections
    F = Identity(3) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    with df.XDMFFile(output_folder+"out.xdmf") as outfile:
        outfile.write(mesh)
        outfile.write_checkpoint(u, "u", 0, append=True)
        outfile.write_checkpoint(F, "F", 0, append=True)

if __name__ == "__main__":
    main()
