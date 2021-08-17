import dolfin as df
import os
import sys
import time
import numpy as np
from mpi4py import MPI

"""
Written by: John Steinman
Simulation of three-field hyperelastic Gel using eigenstrain.
""" 

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 2
# df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
# df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 50000

df.set_log_level(40)

def main():

    params = {}

    params['output_folder'] = './output/eigenstrain/'

    params['mesh'] = "../cell_meshes/bird/inclusion.xdmf"
    params['domains'] = "../cell_meshes/bird/inclusion_domains.xdmf"
    params['boundaries'] = "../cell_meshes/bird/inclusion_boundaries.xdmf"

    solver_call(params)

def solver_call(params):

    # Mesh
    mesh = df.Mesh()
    with df.XDMFFile(params["mesh"]) as infile:
        infile.read(mesh)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(params["domains"]) as infile:
        infile.read(mvc, "domains") 
    domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(params["boundaries"]) as infile:
        infile.read(mvc, "boundaries") 
    boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    element_u = df.VectorElement("CG", mesh.ufl_cell(), 2)
    element_p = df.FiniteElement("DG", mesh.ufl_cell(), 0)
    element_J = df.FiniteElement("DG", mesh.ufl_cell(), 0)
  
    V = df.FunctionSpace(mesh,df.MixedElement([element_u,element_p,element_J]))
    xi = df.Function(V)
    xi.rename('xi','mixed solution')
    xi_ = df.TestFunction(V)
    dxi = df.TrialFunction(V)

    # Set initial values
    V_u = V.sub(0)
    V_p = V.sub(1)
    V_J = V.sub(2)
    u_0 = df.interpolate(df.Constant((0.0, 0.0, 0.0)), V_u.collapse())
    p_0 = df.interpolate(df.Constant(0.0), V_p.collapse())
    J_0 = df.interpolate(df.Constant(1.), V_J.collapse())

    df.assign(xi,[u_0,p_0,J_0])
    u,p,J = df.split(xi)

    # Material parameters
    c1 = df.Constant(1.0)
    c2 = df.Constant(1.0)

    mu = c1*2
    kappa = c2
    lmbda = kappa-2*mu/3

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition

    # Invariants of deformation tensors
    IC_bar = df.tr(C_bar)

    # Eigenstrain
    eigenstrain = df.Expression((("t*k", "0.0", "0.0"),
                                 ("0.0", "t*k", "0.0"), 
                                 ("0.0", "0.0", "t*k")), t=0, k=-.001, degree=1)

    # Linear model
    def eps(v):
        return 0.5*(df.grad(u)+df.grad(u).T)
    def sigma(v):
        return lmbda*df.tr(eps(v))*df.Identity(3) + 2.0*mu*eps(v)

    # Stored strain energy density
    psi_outer = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)
    psi_inner = df.inner(sigma(u), eps(u)-eigenstrain)

    # Total potential energy
    Pi = psi_outer*dx(301) + psi_inner*dx(302) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    bcs = [df.DirichletBC(V_u, zero, boundaries, 201)]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    # solver.parameters['newton_solver']['linear_solver'] = 'superlu_dist'
    # solver.parameters['newton_solver']['linear_solver'] = 'superlu_dist'
    # solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    # solver.parameters['newton_solver']['preconditioner'] = 'jacobi'

    solver.parameters['nonlinear_solver'] = 'snes'
    solver.parameters["snes_solver"]["maximum_iterations"] = 50000

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    val = np.array(len(mesh.cells()),'d')
    val_sum = np.array(0.,'d')
    comm.Reduce(val, val_sum, op=MPI.SUM, root=0)

    if rank==0:
        print("Nonlinear problem created")
        print("Solving")

    chunks = 10
    total_start = time.time()
    sys.stdout.flush() 
    for i in range(chunks):
        iter_start = time.time()
        if rank==0:
            print("     Solver Call: ", i)
            print("     ----------------")

        # Increment eigenstrain
        eigenstrain.t = (i+1)/chunks

        # Solver
        solver.solve()

        if rank == 0:
            print('    Time elapsed = {:2.1f}s'.format(time.time()-iter_start))
        sys.stdout.flush()  

    u, p, J = xi.split(True)

    if rank==0:
        print("Total Time = {:2.1f}s".format(time.time() - total_start))

    # Projections (submesh not yet supported in parallel)
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    energy = df.project(psi_outer, df.FunctionSpace(mesh, "DG", 0))

    # Outputs
    output_folder = params["output_folder"]
    if rank==0:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p.rename("p","pressure")
    p_file.write(p)

    energy_file = df.XDMFFile(output_folder + "psi.xdmf")
    energy.rename("psi", "strain energy density")
    energy_file.write(energy)

    if rank == 0:
        print("Results in: ", output_folder)
        print("Done")
        print("========================================")

if __name__ == "__main__":
    main()