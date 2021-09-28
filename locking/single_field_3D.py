import dolfin as df
from dolfin import dot, ln, det, sym, grad, div, inner, tr, Identity
import matplotlib.pyplot as plt
import numpy as np
import time
import os

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 100000


def single_field():

    # Geometry
    N = 15
    mesh = df.UnitCubeMesh(N, N, N)

    # Subdomains
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = df.CompiledSubDomain("on_boundary && near(x[2], 1)")
    top.mark(boundaries, 1)

    # Measures
    dx = df.Measure("dx", domain=mesh)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), degree=1)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    # Material Properties
    nu = 0.4999     # Poissons ratio
    mu = 1
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter
    kappa = lmbda+2*mu/3

    # Forces
    tf = -1e-1                     # load
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Expression(("0", "0", "t*g"), t=0, g=tf, degree=0)

    # Kinematics
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = df.tr(C)
    J  = df.det(F)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - d) - mu*df.ln(J) + (lmbda/2)*(df.ln(J))**2

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds(1)

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, u, w)
    Dres = df.derivative(res, u, du)

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[2], 0.0))

    bc = df.DirichletBC(V, df.Constant((0.0, 0.0, 0.0)), bottom)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(res, u, bcs=bc, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'

    chunks = 5
    total_start = time.time()
    for i in range(chunks):
        iter_start = time.time()
        print("Solver Call: ", i)
        print("----------------")

        # Increment traction force
        T.t = (i+1)/chunks

        ## Solver
        solver.solve()
        print("Time: ", time.time() - iter_start)

    print("Total time: ", time.time() - total_start)

    # Outputs
    output_folder = "output/3D/single_field/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)


if __name__=="__main__":
    single_field()