import dolfin as df
import numpy as np
import sys
import os
import time

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3

"""
Written by: John Steinman
Simulation of hyperelastic thin plate with circular hole. 

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

    params['case'] = 'c'

    params['output_folder'] = './output/single_field/' + params['case'] + '/'

    params['mesh'] = df.Mesh("./meshes/plate_with_hole.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, det, tr, grad, Identity

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

    # Kinematics
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Constant((0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Material parameters
    nu = 0.35                       # Poisson's ratio
    mu = df.Constant(325e12)       # Bulk Modulus
    lmbda = 2*nu*mu/ (1-2*nu)      # 1st Lame Parameter

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx(201) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, u, w)
    Jac = df.derivative(F, u, du)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))
    u_a = df.Expression(["-x[0]/10","-x[1]/10"], degree=2)
    u_b = df.Expression("-x[1]/10", degree=2)
    u_c = df.Expression(["x[0]/10","-x[1]/10"], degree=2)

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
    problem = df.NonlinearVariationalProblem(F, u, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    
    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    # Projections
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    psi = df.project(psi, df.FunctionSpace(mesh, "CG", 1))

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

    psi_file = df.XDMFFile(output_folder + "Psi.xdmf")
    psi.rename("psi","energy density")
    psi_file.write(psi)

if __name__ == "__main__":
    main()
