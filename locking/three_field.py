import dolfin as df
from dolfin import dot, ln, det, grad, tr, Identity
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


def three_field():
    # Geometry
    l_x, l_y = 5.0, 5.0  # Domain dimensions
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
    V = df.VectorElement('Lagrange', mesh.ufl_cell(), 3)
    W = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    R = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)   
    M = df.FunctionSpace(mesh, df.MixedElement([V,W,R]))
    V, W, R = M.split()

    # Trial and Test functions
    dxi = df.TrialFunction(M)
    xi_ = df.TestFunction(M)

    # Functions from most recent iteration
    xi = df.Function(M) 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V.collapse())
    p_0 = df.interpolate(df.Constant((0.0)), W.collapse())
    J_0 = df.interpolate(df.Constant((1.0)), R.collapse())
    df.assign(xi, [u_0, p_0, J_0])

    # Variational forms
    u, p, Jt = df.split(xi)

    # Parameters
    nu = 0.4999  # Poissons ratio
    mu = 1.45e7
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter
    kappa = lmbda+2*mu/3
    c1 = kappa/4
    c2=mu/2

    g_int = -1e7                # load
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Expression(("0", "t*g"), t=0, g=g_int, degree=1)

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Greem tensor

    # Invariants of deformation tensors
    J = det(F)
    I = tr(b) 
    Ibar = I * J**-1

    # Stored strain energy density
    psi_1 = c1*(Jt**2-1-2*ln(Jt))
    psi_2 = c2*(Ibar-2)
    psi_3 = p*(J-Jt)
    psi = psi_1 + psi_2 + psi_3

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds(1)

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, xi, xi_)
    Jac = df.derivative(F, xi, dxi)

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[1], 0.0))

    bcs = df.DirichletBC(V, df.Constant((0.0, 0.0)), bottom)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['maximum_iterations'] = 10
    solver.parameters['newton_solver']['linear_solver'] = 'minres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'

    chunks = 250
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
    u, p, Jt = xi.split() 

    # Post-process
    plot = df.plot(u, mode="displacement")
    plt.colorbar(plot)
    plt.show()

    # Outputs
    output_folder = "output/three_field/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)


if __name__=="__main__":
    three_field()    