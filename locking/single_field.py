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
    l_x, l_y = 5, 5  # Domain dimensions
    n_x, n_y = 20, 20    # Number of elements
    mesh = df.RectangleMesh(df.Point(0.0,0.0), df.Point(l_x, l_y), n_x, n_y)    

    # Subdomains
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = df.AutoSubDomain(lambda x: df.near(x[1], 5.0))
    top.mark(boundaries, 1)

    # Measures
    dx = df.Measure("dx", domain=mesh)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    deg = 2
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), degree=deg)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    # Parameters
    nu = 0.4999  # Poissons ratio
    mu = 80.194
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter

    g_int = -100                # load
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Expression(("0", "t*g"), t=0, g=g_int, degree=1)

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds(1)

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, u, w)
    Jac = df.derivative(F, u, du)

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[1], 0.0))

    bcs = df.DirichletBC(V, df.Constant((0.0, 0.0)), bottom)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, u, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    # solver.parameters['newton_solver']['relative_tolerance'] = 1E-4
    # solver.parameters['newton_solver']['maximum_iterations'] = 10
    # solver.parameters['newton_solver']['linear_solver'] = 'lu'
    # solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    # solver.parameters['newton_solver']['linear_solver'] = 'minres'
    # solver.parameters['newton_solver']['preconditioner'] = 'jacobi'

    chunks = 100
    total_start = time.time()
    for i in range(chunks):
        iter_start = time.time()
        print("Solver Call: ", i)
        print("----------------")

        # Increment eigenstrain
        T.t = (i+1)/chunks

        ## Solver
        solver.solve()
        print("Time: ", time.time() - iter_start)

    print("Total time: ", time.time() - total_start)

    # Post-process
    plot = df.plot(u, mode="displacement")
    plt.colorbar(plot)
    plt.show()
"""
    # Outputs
    output_folder = "output/single_field/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)
"""

if __name__=="__main__":
    single_field()