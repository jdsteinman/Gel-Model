import dolfin as df
import matplotlib.pyplot as plt
import time
import os

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 100000


def three_field():
    # Geometry
    N = 50
    mesh = df.UnitSquareMesh(N, N)

    # Subdomains
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = df.AutoSubDomain(lambda x: df.near(x[1], 1))
    top.mark(boundaries, 1)

    # Measures
    dx = df.Measure("dx", domain=mesh)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    V = df.VectorElement('CG', mesh.ufl_cell(), degree=1)
    W = df.FiniteElement('DG', mesh.ufl_cell(), degree=0)
    R = df.FiniteElement('DG', mesh.ufl_cell(), degree=0)   
    M = df.FunctionSpace(mesh, df.MixedElement([V,W,R]))
    V, W, R = M.split()

    # Functions from most recent iteration
    xi = df.Function(M) 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V.collapse())
    p_0 = df.interpolate(df.Constant((0.0)), W.collapse())
    J_0 = df.interpolate(df.Constant((1.0)), R.collapse())
    df.assign(xi, [u_0, p_0, J_0])

    # Variational forms
    u, p, J = df.split(xi)

    # Parameters
    nu = 0.4999     # Poissons ratio
    mu = 1
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter
    kappa = lmbda+2*mu/3
    c1 = df.Constant(kappa)
    c2 = df.Constant(mu/2)

    g_int = 1e-1
    B = df.Constant((0., 0.))     # Body force per unit volume
    T = df.Expression(("0", "t*g"), t=0, g=g_int, degree=0)

    # Kinematics
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition
    IC_bar = df.tr(C_bar)          # Invariant of isochoric C

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds(1)

    # Gateaux Derivative
    res = df.derivative(Pi, xi)
    Dres = df.derivative(res, xi)

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[1], 0.0))

    bcs = df.DirichletBC(V, df.Constant((0.0, 0.0)), bottom)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
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
    u, p, J = xi.split() 

    # Post-process
    plot = df.plot(u, mode="displacement")
    plt.colorbar(plot)
    plt.show()

    plot2 = df.plot(J)
    plt.colorbar(plot2)
    plt.show()

    # Outputs
    output_folder = "output/three_field/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

if __name__=="__main__":
    three_field()    