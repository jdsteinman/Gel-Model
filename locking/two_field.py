import dolfin as df
from dolfin import dot, ln, det, grad, tr, Identity
import matplotlib.pyplot as plt
import numpy as np
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


def two_field():
    # Geometry
    l_x, l_y = 5.0, 5.0  # Domain dimensions
    n_x, n_y = 20, 120    # Number of elements
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
    V = df.VectorElement('P', mesh.ufl_cell(), 1)
    S = df.FiniteElement('P', mesh.ufl_cell(), 1)
    M = df.FunctionSpace(mesh, df.MixedElement([V,S]))
    V, S = M.split()

    # Functions from most recent iteration
    dup = df.TrialFunction(M) 
    _u, _p = df.TestFunctions(M)

    _u_p = df.Function(M)
    u, p = df.split(_u_p)

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V.collapse())
    p_0 = df.interpolate(df.Constant((0.0)), S.collapse())
    df.assign(_u_p, [u_0, p_0])

    # Paramters
    E = 1.0e3   # Youngs modulus
    nu = 0.4999  # Poissons ratio

    g_int = -2.5e2
    B = df.Constant((0., 0.))     # Body force per unit volume
    T = df.Expression(("0", "t*g"), t=0, g=g_int, degree=0)

    # Kinematics
    def pk1Stress(u,pressure,E,nu):
        G = E/(2*(1+nu))
        c1 = G/2.0
        
        I = df.Identity(V.mesh().geometry().dim())  # Identity tensor
        F = I + df.grad(u)          # Deformation gradient
        C = F.T*F                # Right Cauchy-Green tensor
        Ic = df.tr(C)               # Invariants of deformation tensors
        J = det(F)
        pk2 = 2*c1*I-pressure*df.inv(C) # second PK stress
        return pk2, (J-1)    

    pkstrs, hydpress =  pk1Stress(u,p,E,nu)
    I = df.Identity(M.mesh().geometry().dim())
    dgF = I + df.grad(u)
    F1 = df.inner(df.dot(dgF,pkstrs), df.grad(_u))*dx - df.dot(B, _u)*dx - dot(T, _u)*ds
    F2 = hydpress*_p*dx 
    F = F1+F2
    J = df.derivative(F, _u_p,dup)

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[1], 0.0))

    bcs = df.DirichletBC(V, df.Constant((0.0, 0.0)), bottom)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, _u_p, bcs=bcs, J=J)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1E-6
    solver.parameters['newton_solver']['linear_solver'] = 'lu'
    # solver.parameters['newton_solver']['linear_solver'] = 'minres'
    # solver.parameters['newton_solver']['preconditioner'] = 'jacobi'

    chunks = 50
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
    u, p = _u_p.split() 

    # Post-process
    plot = df.plot(u, mode="displacement")
    plt.colorbar(plot)
    plt.show()

    plot2 = df.plot(p)
    plt.colorbar(plot2)
    plt.show()

    # Outputs
    output_folder = "output/two_field/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p.rename("p","pressure")
    p_file.write(p)

if __name__=="__main__":
    two_field()    