import dolfin as df
from mpi4py import MPI
from shutil import copyfile 
import numpy as np
import sys
import os
import time

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

    params['mesh'] = "./meshes/hole_with_inner_cube.xdmf"
    params['domains'] = "./meshes/hole_with_inner_cube_domains.xdmf"
    params['boundaries'] = "./meshes/hole_with_inner_cube_boundaries.xdmf"
    
    params['L']=400
    params['D']=50

    params['mu_ff'] = 100e-6
    params['d'] = d 
    params['u_inner'] = -2

    params['output_folder'] = './output/d_mu'

    solver_call(params)

class ShearModulus(df.UserExpression):
    def __init__(self, mu_ff, c, mf, *args, **kwargs):
        self.mu_ff = mu_ff
        self.c = c
        self.mf = mf
        super().__init__(*args, **kwargs)

    def value_shape(self):
        return()

    def eval_cell(self, value, x, cell):
        if self.mf.array()[cell.index]==302:
            value[0]=self.mu_ff*self.c
        elif self.mf.array()[cell.index]==301: 
            value[0]=self.mu_ff
        else:
            print("Unknown cell index")
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
    dim = u.geometric_dimension()
    I = df.Identity(dim)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/dim)            # Isochoric decomposition
    IC_bar = df.tr(C_bar)          # Invariant of isochoric C

    # Material parameters
    mu_ff = params["mu_ff"]
    c = params["c"]

    mu = ShearModulus(mu_ff, c, domains)
    nu = 0.499 
    kappa = 2*mu_ff*(1+nu)/3/(1-2*nu)

    c1 = mu/2
    c2 = df.Constant(kappa)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-dim) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Gateaux Derivative
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Subdomains
    length = params["L"]
    xboundary = df.CompiledSubDomain("near(abs(x[0]), R) && abs(x[1])<1 && abs(x[2])<1", R=length/2)
    yboundary = df.CompiledSubDomain("abs(x[0])<1 &&near(abs(x[1]), R)&& abs(x[2])<1", R=length/2)
    zboundary = df.CompiledSubDomain("abs(x[0])<1 && abs(x[1])<1 && near(abs(x[2]), R) ", R=length/2)
    corners = df.CompiledSubDomain("near(abs(x[0]), R) && near(abs(x[1]), R) && near(abs(x[2]), R)", R=length/2)
    
    # Boundary Conditions
    D = params["D"]
    u_mag = params['u_inner']
    u_inner = df.Expression(["x[0]/r*c*t","x[1]/r*c*t","x[2]/r*c*t"], r=D/2, c=u_mag, t=0, degree=1)

    inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 202)
    fixed_outer = df.DirichletBC(V_u, df.Constant((0.,0.,0.)), boundaries, 201)
    #corners_bc = df.DirichletBC(V_u, df.Constant((0.,0.,0.)), corners, method="pointwise")
    bc_x_1 = df.DirichletBC(V_u.sub(1), df.Constant(0), xboundary, method="pointwise")
    bc_x_2 = df.DirichletBC(V_u.sub(2), df.Constant(0), xboundary, method="pointwise")
    #bc_y_1 = df.DirichletBC(V_u.sub(0), df.Constant(0), yboundary, method="pointwise")
    #bc_y_2 = df.DirichletBC(V_u.sub(2), df.Constant(0), yboundary, method="pointwise")
    #bc_z_1 = df.DirichletBC(V_u.sub(0), df.Constant(0), zboundary, method="pointwise")
    #bc_z_2 = df.DirichletBC(V_u.sub(1), df.Constant(0), zboundary, method="pointwise")
    bcs = [inner_bc, bc_x_1, bc_x_2]
    bcs = [inner_bc, fixed_outer]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'lu'

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size()>1:
        df.set_log_level(40)  # Mute output

    ele = np.array(len(mesh.cells()),'d') # Number of elements
    ele_sum = np.array(0.,'d')
    comm.Reduce(ele, ele_sum, op=MPI.SUM, root=0)

    output_folder = params["output_folder"]
    if rank==0:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder) 

    if rank == 0:
        print("Mesh: ", params["mesh"])
        print('Total number of elements = {:d}'.format(int(ele_sum)))
        print("Solving =========================")

    # Solve
    chunks = 1
    total_start = time.time()
    for i in range(chunks):
        if rank==0: print("Iteration: ", i)
        start = time.time()
        u_inner.t = (i+1)/chunks
        solver.solve()
        if rank==0: print("Time: ", time.time()-start) 
    if rank==0: print("Total Time: ", time.time() - total_start, "s")

    u, p, J = xi.split(True)

    # Projections
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    mu = df.project(mu, df.FunctionSpace(mesh, "DG", 0))

    # Outputs
    mu_file = df.XDMFFile(os.path.join(output_folder, "mu.xdmf"))
    mu.rename("mu", "Shear Modulus")
    mu_file.write(mu)

    disp_file = df.XDMFFile(os.path.join(output_folder, "U.xdmf"))
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(os.path.join(output_folder, "F.xdmf"))
    F.rename("F", "Deformation Gradient")
    F_file.write(F)

    J_file = df.XDMFFile(os.path.join(output_folder, "J.xdmf"))
    J.rename("J","Jacobian")
    J_file.write(J)

    if rank==0:
        fname = os.path.basename(__file__)
        dir = os.path.dirname(os.path.abspath(__file__))
        copyfile(os.path.join(dir,fname), os.path.join(output_folder,fname))

    with open(os.path.join(output_folder,"log_params.txt"), "w+") as f:
        f.write("Mesh: {:s}\n".format(params["mesh"]))
        f.write("No. Elements: {:d}\n".format(int(ele_sum)))
        f.write("No. Processors: {:d}\n".format(int(comm.Get_size())))
        f.write("mu_ff = {:e}\n".format(mu_ff))
        f.write("kappa = {:e}\n".format(kappa))
        f.write("c =     {:f}\n".format(c))
        f.write("Total Time = {:f}s\n".format(time.time()-total_start))
    
    if rank==0: print("Done")

if __name__ == "__main__":
    main()
