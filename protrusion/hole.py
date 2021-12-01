import dolfin as df
import numpy as np
import sys
import os
import time
from shutil import copyfile 

"""
Written by: John Steinman
"""

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

def main():

    params = {}

    params['mesh'] = "./meshes/hole.xdmf"
    params['domains'] = "./meshes/hole_domains.xdmf"
    params['boundaries'] = "./meshes/hole_boundaries.xdmf"
    params['cell_vertices'] = np.loadtxt("./meshes/hole_surface.txt")

    params['u_init'] = "./output/FBC/homogeneous/u_out.xdmf"

    params['mu']  = 100e-6
    params['nu'] = 0.49

    params["chunks"] = 3

    # params['output_folder'] = './output/DBC/homogeneous'
    # params['output_folder'] = './output/DBC/FGM'
    params['output_folder'] = './output/FBC/homogeneous'
    # params['output_folder'] = './output/FBC/FGM'

    solver_call(params)

class Modulus(df.UserExpression):
    def __init__(self, modulus_gel, **kwargs):
        self.lmbda_gel = modulus_gel
        super().__init__(**kwargs)

    def value_shape(self):
        return ()

    def eval(self, value, x):
            value[0] = self.lmbda_gel*1.01 

class DegradedModulus(df.UserExpression):
    def __init__(self, modulus_gel, surface_vert, **kwargs):
        self.lmbda_gel = modulus_gel
        self.vert = surface_vert
        super().__init__(**kwargs)

    def value_shape(self):
        return ()

    def eval(self, value, x):
        px = np.array([x[0], x[1], x[2]], dtype="float64")
        r = px - self.vert
        r = np.sum(np.abs(r)**2, axis=-1)**(1./2)
        r = np.amin(r)

        if r < 10:
            value[0]=self.lmbda_gel*((r/10)**0.5 + 0.01)
        else:
            value[0] =  self.lmbda_gel*1.01 

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

    surface_vert = params['cell_vertices']

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    # Initialize  
    V_init = df.VectorFunctionSpace(mesh, "CG", 2)
    u_init = df.Function(V_init)  
    u_init_file = df.XDMFFile(params["u_init"])
    u_init_file.read_checkpoint(u_init, "u", 0)
    u_init.set_allow_extrapolation(True)

    u_0 = df.interpolate(u_init, V)
    # df.assign(u, u_0)

    # Kinematics
    B = df.Constant((0, 0, 0))     # Traction force on the boundary
    # T = df.Constant((-5e-5, 0, 0))     # Traction force on the boundary
    T = df.Expression(["-7e-5*t*x[0]/10", "0", "0"], t=0, degree=0)    # Traction force on the boundary
    dim = u.geometric_dimension()
    I = df.Identity(dim)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    C = F.T*F                      # Right Cauchy-Green tensor
    J  = df.det(F)

    # Invariants of deformation tensors
    C_bar = C/J**(2/dim)            # Isochoric decomposition
    IC_bar = df.tr(C_bar)          # Invariant

    # Material parameters
    nu = params["nu"]          
    mu_ff = params["mu"]
    mu = Modulus(mu_ff)
    # mu = DegradedModulus(mu_ff, surface_vert)  
    lmbda = 2*nu*mu/(1-2*nu)      
    kappa = lmbda+2*mu/3

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(IC_bar - dim) + (kappa/4)*(J**2-1-2*df.ln(J))

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx(301) - df.dot(T, u)*ds(202)

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    dPi = df.derivative(Pi, u, w)
    ddPi = df.derivative(dPi, u, du)
 
    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_d = df.Expression(["-x[0]/10*2","0","0"], degree=1)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, u_d, boundaries, 202)

    # bcs = [inner_bc, outer_bc]
    bcs = [outer_bc]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(dPi, u, bcs=bcs, J=ddPi)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-9
    # solver.parameters['newton_solver']['relative_tolerance'] = 1e-4
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    # solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    # solver.parameters['newton_solver']['preconditioner'] = 'hypre_amg'
    
    # Solve
    chunks = params["chunks"]
    u.vector()[:]*=1/chunks

    total_start = time.time() 
    for i in range(chunks):
        start = time.time()
 
        T.t = (i+1)/chunks
        solver.solve()

        end = time.time()
        time_elapsed = end - start
        print(time_elapsed)
    print("Total runtime: ", time.time()-total_start)

    # Projections
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    mu = df.project(mu, df.FunctionSpace(mesh, "CG", 1))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(os.path.join(output_folder, "u.xdmf"))
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(os.path.join(output_folder, "F.xdmf"))
    F.rename("F","deformation gradient")
    F_file.write(F)

    mu_file = df.XDMFFile(os.path.join(output_folder, "mu.xdmf"))
    mu.rename("mu","shear modulus")
    mu_file.write(mu)

    out_file = df.XDMFFile((os.path.join(output_folder, "u_out.xdmf")))
    out_file.write_checkpoint(u, "u", 0)   #Not appending

    python_file = os.path.basename(__file__)
    copyfile(python_file, os.path.join(output_folder, python_file))

if __name__ == "__main__":
    main()
