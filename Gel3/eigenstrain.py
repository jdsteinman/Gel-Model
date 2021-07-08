import dolfin as df
import os
import time

"""
Written by: John Steinman
Fenics simulation of single-field hyperelastic Gel.
- outputs:
    - displacement and deformation gradient fields (XDMF)
""" 

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

def main():

    params = {}

    params['output_folder'] = './output/single_field/'

    params['mesh'] = df.Mesh("./meshes/cytod_uncentered_unpca.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/cytod_uncentered_unpca_physical_region.xml")

    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/cytod_uncentered_unpca_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, inner, sym, det, tr, grad, Identity

    # Mesh
    mesh = params["mesh"]

    domains = params["physical_region"]
    boundaries = params["facet_region"]

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    U = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, U)

    du, w = df.TrialFunction(V), df.TestFunction(V) 
    u = df.Function(V)
    u.vector()[:] = 0

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Material parameters
    nu = 0.49                      # Poisson's ratio
    mu = df.Constant(108.4e12)     # Shear Modulus
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter

    # Eigenstrain
    eigenstrain = df.Expression((("t*k", "0.0", "0.0"),
                                 ("0.0", "t*k", "0.0"), 
                                 ("0.0", "0.0", "t*k")), t=0, k=-0.001, degree=1)

    # Linear model
    def eps(v):
        return 0.5*(grad(u)+grad(u).T)
    def sigma(v):
        return lmbda*tr(eps(v))*Identity(3) + 2.0*mu*eps(v)

    # Stored strain energy density (compressible neo-Hookean model)
    psi_outer = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
    psi_inner = inner(sigma(u), eps(u)-eigenstrain)

    # Total potential energy
    Pi = psi_outer*dx(301) + psi_inner*dx(302) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, u, w)
    Jac = df.derivative(F, u, du)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    bcs = [df.DirichletBC(V, zero, boundaries, 201)]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, u, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'minres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    
    chunks = 10
    total_start = time.time()
    for i in range(chunks):
        iter_start = time.time()
        print("Solver Call: ", i)
        print("----------------")

        # Increment eigenstrain
        eigenstrain.t = (i+1)/chunks

        ## Solver
        start = time.time()
        solver.solve()
        print("Time: ", time.time() - iter_start)
    
    print("Total Time: ", time.time() - start, "s")

    # Projections
    F = Identity(3) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

    p1 = df.project(psi_outer, df.FunctionSpace(mesh, "CG", 2))

    # Extract values in outer region
    submesh = df.SubMesh(mesh, domains, 301)
    u_outer = df.interpolate(u, df.FunctionSpace(submesh, U))
    F_outer = df.interpolate(F, df.TensorFunctionSpace(submesh, "CG", 1, shape=(3, 3)))
    p_outer = df.interpolate(p1, df.FunctionSpace(submesh, U))

    energy = df.assemble(psi_outer*dx(301) + psi_inner*dx(302))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u_outer.rename("U","displacement")
    disp_file.write(u_outer)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F_outer.rename("F","deformation gradient")
    F_file.write(F_outer)

    p_file = df.XDMFFile(output_folder + "F.xdmf")
    p_outer.rename("p","deformation gradient")
    p_file.write(p_outer)


if __name__ == "__main__":
    main()