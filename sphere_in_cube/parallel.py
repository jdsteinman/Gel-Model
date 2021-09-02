import dolfin as df
import os
import shutil
import sys
import time
import numpy as np
from mpi4py import MPI

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 10000

"""
Written by: John Steinman
"""

def main():

    params = {}

    params['mesh'] = "./meshes/hole.xdmf"
    params['domains'] = "./meshes/hole_domains.xdmf"
    params['boundaries'] = "./meshes/hole_boundaries.xdmf"

    params['mu_ff'] = 100e12
    params['c'] = 0.1
    params['near_field'] = 50

    params['output_folder'] = './output/hole/C=' + str(params['c']) + '/'

    solver_call(params)

class ShearModulus(df.UserExpression):
    def __init__(self, mu_ff, c, L, *args, **kwargs):
        self.mu_ff = mu_ff
        self.c = c
        self.L = L
        super().__init__(*args, **kwargs)

    def value_shape(self):
        return()

    def eval(self, value, x):
        if abs(x[0])<=self.L/2 and abs(x[1])<=self.L/2 and abs(x[2])<=self.L/2:
            value[0]=self.mu_ff*self.c
        else:   
            value[0]=self.mu_ff

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
    element_u = df.VectorElement("CG",mesh.ufl_cell(),2)
    element_p = df.FiniteElement("DG",mesh.ufl_cell(),0)
    element_J = df.FiniteElement("DG",mesh.ufl_cell(),0)
  
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

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition
    IC_bar = df.tr(C_bar)          # Invariant of isochoric C

    # Material parameters
    mu_ff = params["mu_ff"]
    c = params["c"]
    near_field = params["near_field"]

    mu = ShearModulus(mu_ff, c, near_field)
    nu = 0.499 
    kappa = 2*mu_ff*(1+nu)/3/(1-2*nu)

    c1 = mu/2
    c2 = df.Constant(kappa)

    # Stored strain energy densit    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size()>1:
        df.set_log_level(40)  # Mute output
    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_inner = df.Expression(["x[0]/15*c*t","x[1]/15*c*t","x[2]/15*c*t"], c=5, t=1, degree=1)

    outer_bc = df.DirichletBC(V_u, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 202)
    bcs = [outer_bc, inner_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size()>1:
        df.set_log_level(40)  # Mute output

    val = np.array(len(mesh.cells()),'d')
    val_sum = np.array(0.,'d')
    comm.Reduce(val, val_sum, op=MPI.SUM, root=0)

    if rank == 0:
        print("Mesh: ", params["mesh"])
        print('Total number of elements = {:d}'.format(int(val_sum)))
        print("mu_ff = {:5.3e}".format(mu_ff))
        print("c     = {:f}".format(c))
        print("kappa = {:5.3e}".format(kappa))
        print("Solving =========================")

    # Solve
    chunks = 5
    start = time.time()
    sys.stdout.flush() 
    for i in range(chunks):
        start = time.time()
        if rank == 0:
            print("Iteration: ", i)

        u_inner.t = (i+1)/chunks
        solver.solve()

        end = time.time()
        time_elapsed = end - start

        if rank == 0:
            print('Time elapsed = {:2.1f}s\n'.format(time_elapsed))
        sys.stdout.flush()  

    u, p, J = xi.split(True)

    # Projections
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    mu = df.project(mu, df.FunctionSpace(mesh, "DG", 1))

    # Outputs
    output_folder = params["output_folder"]    
    if rank==0:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    mu_file = df.XDMFFile(output_folder + "mu.xdmf")
    mu.rename("mu", "Shear Modulus")
    mu_file.write(mu)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    # F_file = df.XDMFFile(output_folder + "F.xdmf")
    # F.rename("F","deformation gradient")
    # F_file.write(F)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    # p_file = df.XDMFFile(output_folder + "p.xdmf")
    # p.rename("p","pressure")
    # p_file.write(p)

    if rank == 0:
        with open(output_folder+"log_params.txt", "w+") as f:
            f.write("Mesh: {:s}\n".format(params["mesh"]))
            f.write("No. Elements: {:d}\n".format(mesh.num_cells()))
            f.write("mu_ff = {:e}\n".format(mu_ff))
            f.write("kappa = {:e}\n".format(kappa))
            f.write("c =     {:f}\n".format(c))

        shutil.copyfile("log", output_folder+"log")
        shutil.copyfile("parallel.py", output_folder+"parallel.py")

if __name__ == "__main__":
    main()