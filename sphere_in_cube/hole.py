import dolfin as df
import numpy as np 
import sys
import os
import time
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

    params['mesh'] = "./meshes/hole_fgm.xdmf"
    params['domains'] = "./meshes/hole_fgm_domains.xdmf"
    params['boundaries'] = "./meshes/hole_fgm_boundaries.xdmf"

    params['mu'] = 100e-6
    params['nu'] = 0.499
    params['c'] = 0.5

    params['output_folder'] = './output/hole/C=' + str(params['c']) + '/'

    solver_call(params)

class ElasticModulus(df.UserExpression):
    """
    Custom Expression for spatially varying elastic modulus, delta
        - Currently a step function
    """
    def __init__(self, delta, c, mf, tag, *args, **kwargs):
        self.delta = delta
        self.c = c
        self.mf = mf
        self.tag = tag
        super().__init__(*args, **kwargs)

    def value_shape(self):
        return()   # Scalar

    def eval_cell(self, value, x, cell):
        if self.mf.array()[cell.index] == self.tag:
            value[0]=self.delta*self.c
        else:   
            value[0]=self.delta

def solver_call(params):

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size()>1:
        df.set_log_level(40)  # Mute output

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
    u, p, J = df.split(xi)

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
    mu_ff = params["mu"]
    nu_ff = params["nu"]
    c = params["c"]

    mu = ElasticModulus(mu_ff, c, domains, 302)
    nu = ElasticModulus(nu_ff, c, domains, 302)
    kappa = 2*mu*(1+nu)/3/(1-2*nu)
    kappa_ff = 2*mu_ff*(1+nu_ff)/3/(1-2*nu_ff)

    c1 = mu/2
    c2 = kappa

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Gateaux Derivative
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    u_inner = df.Expression(["x[0]/25*c*t","x[1]/25*c*t","x[2]/25*c*t"], c=2, t=0, degree=1)
    inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 202)
    outer_bc = df.DirichletBC(V_u, df.Constant((0.,0.,0)), boundaries, 201)
    bcs = [inner_bc, outer_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'

    # Solve
    chunks = 5
    total_start = time.time()
    for i in range(chunks):
        if rank==0: print("\nIteration: ", i)

        start = time.time()
        u_inner.t = (i+1)/chunks
        solver.solve()

        if rank==0: print("Time: ", time.time()-start) 
    if rank==0: 
        print("Total Time: ", time.time() - total_start, "s")

    # Split solution
    u, p, J = xi.split(True)

    # Projections
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    mu = df.project(mu, df.FunctionSpace(mesh, "DG", 1))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder) and rank==0:
        os.makedirs(output_folder)

    mu_file = df.XDMFFile(output_folder + "mu.xdmf")
    mu.rename("mu", "Shear Modulus")
    mu_file.write(mu)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F", "Deformation Gradient")
    F_file.write(F)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    with open(output_folder+"log_params", "w+") as f:
        f.write("Mesh: {:s}\n".format(params["mesh"]))
        f.write("mu =    {:e}\n".format(mu_ff))
        f.write("nu =    {:e}\n".format(nu_ff))
        f.write("kappa = {:e}\n".format(kappa_ff))
        f.write("c =     {:f}\n".format(c))

if __name__ == "__main__":
    main()
