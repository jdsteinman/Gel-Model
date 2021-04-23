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
Simulation of hyperelastic thin plate with circular hole. 
Inner boundary is prescribed a displacement and outer boundary is fixed.

Test Cases:
    a. Uniform radial contraction
    b. Contraction in y only
    c. Contraction in y, expansion in x

Outputs:
    - Displacement (u) : xdmf
    - Deformation Gradient (F) : xdmf
"""

def main():

    params = {}

    params['case'] = 'a'

    params['output_folder'] = './output/' + params['case'] + '/'

    params['mesh'] = df.Mesh("./meshes/plate_with_hole.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, det, tr, grad, Identity, inner

    # Mesh
    mesh = params["mesh"]

    domains = params["physical_region"]
    boundaries = params["facet_region"]

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    ## Kinematics
    B = df.Constant((0, 0))  # Body force per unit volume
    T = df.Constant((0, 0))  # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor

    # Material parameters
    nu = 0.49                        # Poisson's ratio
    mu = df.Constant(325 * 10**12)   # Bulk Modulus
    lmbda = 2*nu*mu/ (1-2*nu)        # 1st Lame Parameter

    # symmetric strain-rate tensor
    def eps(v):
        return 0.5*(grad(v) + grad(v).T)

    # Stress Tensor
    def sigma(v):
        return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

    # Liner Variational Problem
    a = inner(sigma(du), eps(w))*dx
    L = dot(B, w)*dx + dot(T, w)*ds

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))
    u_a = df.Expression(["-x[0]/10","-x[1]/10"], degree=2)
    u_b = df.Expression("-x[1]/10", degree=2)
    u_c = df.Expression(["x[0]/100","-x[1]/100"], degree=2)

    case = params["case"]
    if case=="a":
        inner_bc = df.DirichletBC(V, u_a, boundaries, 102)
    elif case=="b":
        inner_bc = df.DirichletBC(V.sub(1), u_b, boundaries, 102)
    elif case=="c":
        inner_bc = df.DirichletBC(V, u_c, boundaries, 102)
    else:
        print("Invalid case: ", case)
        sys.exit()

    outer_bc = df.DirichletBC(V, zero, boundaries, 101)
    bcs = [inner_bc, outer_bc]

    # Create nonlinear variational problem and solve
    problem = df.LinearVariationalProblem(a, L, u, bcs=bcs)
    solver = df.LinearVariationalSolver(problem)

    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    # Projections
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

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
