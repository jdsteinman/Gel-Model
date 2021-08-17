import dolfin as df
import os
import time

"""
Written by: John Steinman
Simulation of three-field hyperelastic Gel using eigenstrain.
""" 

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 2
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6

def main():

    params = {}

    params['output_folder'] = './output/eigenstrain/'

    params['mesh'] = df.Mesh("../cell_meshes/bird/inclusion.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], \
        "../cell_meshes/bird/inclusion_physical_region.xml")

    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], \
        "../cell_meshes/bird/inclusion_facet_region.xml")

    solver_call(params)

def solver_call(params):

    # Mesh
    mesh = params["mesh"]
    domains = params["physical_region"]
    boundaries = params["facet_region"]

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
    c2 = df.Constant(1000.0)

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
                                 ("0.0", "0.0", "t*k")), t=0, k=-1., degree=1)

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
    solver.parameters['newton_solver']['linear_solver'] = 'umfpack'
    # solver.parameters['newton_solver']['linear_solver'] = 'minres'
    # solver.parameters['newton_solver']['preconditioner'] = 'sor'
    
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
    F = df.Identity(3) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

    # Extract values in outer region
    submesh = df.SubMesh(mesh, domains, 301)
    u_outer = df.interpolate(u, df.FunctionSpace(submesh, element_u))
    F_outer = df.interpolate(F, df.TensorFunctionSpace(submesh, "CG", 1, shape=(3, 3)))

    # energy = df.assemble(psi_outer*dx(301) + psi_inner*dx(302))
    energy = df.project(psi_outer, df.FunctionSpace(submesh, "DG", 0))

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

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p.rename("p","pressure")
    p_file.write(p)

    energy_file = df.XDMFFile(output_folder + "psi.xdmf")
    energy.rename("psi", "strain energy density")
    energy_file.write(energy)

if __name__ == "__main__":
    main()