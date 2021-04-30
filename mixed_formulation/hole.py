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

def main():

    params = {}

    params['output_folder'] = './output/'

    params['mesh'] = df.Mesh("./meshes/hole.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/hole_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/hole_facet_region.xml")

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
    V = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    W = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    R = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)   
    M = df.FunctionSpace(mesh, df.MixedElement([V,W,R]))
    V, W, R = df.split(M)

    # Trial and Test functions
    dxi = df.TrialFunction(M)
    m = df.TestFunction(M)

    # Functions from most recent iteration
    xi = df.Function(M) 
    u, p, Jt = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Material parameters
    nu = 0.30                      # Poisson's ratio
    mu = df.Constant(325e12)       # Shear Modulus (2nd Lame)
    lmbda = 2*nu*mu/ (1-2*nu)      # 1st Lame Parameter
    kappa = lmbda+2/3*mu           # bulk modulus

    # Stored strain energy density (compressible neo-Hookean model)
    # psi_1 = mu/2*(Ic-3-2*ln(J))
    # psi_2 = (kappa/2)*(J-1)**2
    # psi_3   = p*(J-Jt)
    # psi = psi_1 + psi_2 + psi_3
    psi = (mu/2)*(Ic - d) - mu*ln(J) + (lmbda/2)*(ln(J))**2 + p*(J-Jt)

    # Total potential energy
    Pi = psi*dx(301) - dot(B, u)*dx(301) - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, xi, m)
    Jac = df.derivative(F, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_a = df.Expression(["x[0]/1000","x[1]/1000","-2*x[2]/1000"], degree=2)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, u_a, boundaries, 202)

    bcs = [inner_bc]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    
    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    u, Jt, p = xi.split()    

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
