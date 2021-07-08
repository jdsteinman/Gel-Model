import dolfin as df
import os
import time

"""
Written by: John Steinman
Fenics simulation of three-field hyperelastic plate with eigenstrain.
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

    params['output_folder'] = './output/three_field/'

    params['mesh'] = df.Mesh("./meshes/plate_with_inclusion.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/plate_with_inclusion_physical_region.xml")

    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], \
        "./meshes/plate_with_inclusion_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, inner, det, tr, grad, Identity

    # Mesh
    mesh = params["mesh"]

    domains = params["physical_region"]
    boundaries = params["facet_region"]

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    V_ele = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    W_ele = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    R_ele = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)   
    M = df.FunctionSpace(mesh, df.MixedElement([V_ele,W_ele,R_ele]))
    V, W, R = M.split()

    # Trial and Test functions
    dxi = df.TrialFunction(M)
    xi_ = df.TestFunction(M)

    # Functions from most recent iteration
    xi = df.Function(M) 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V.collapse())
    p_0 = df.interpolate(df.Constant(0.0), W.collapse())
    J_0 = df.interpolate(df.Constant(1.0), R.collapse())
    df.assign(xi, [u_0, p_0, J_0])

    # Variational forms
    u, p, Jt = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Constant((0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Greem tensor

    # Invariants of deformation tensors
    J = det(F)
    I = tr(b) 
    Ibar = I * J**-1

    # Inclusion parameters
    nu = 0.49                      # Poisson's ratio
    mu = df.Constant(1.)           # Bulk Modulus
    lmbda = 2*nu*mu/(1-2*nu)       # 1st Lame Parameter
    kappa = lmbda+2*mu/3

    # Gel parameters
    c1 = df.Constant(50.0)
    c2 = df.Constant(0.5)

    # Eigenstrain
    eigenstrain = df.Expression((("t*0.1", "0.0"),
                                 ("0.0", "t*0.1")), t=0, degree=1)

    # Linear model
    def eps(v):
        return 0.5*(grad(u)+grad(u).T)
    def sigma(v):
        return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

    # Stored strain energy density (compressible neo-Hookean model)
    psi_outer = c1*(Jt**2-1-2*ln(Jt))+c2*(Ibar-2)+p*(J-Jt)
    psi_inner = inner(sigma(u), eps(u)-eigenstrain)

    # Total potential energy
    Pi = psi_outer*dx(201) + psi_inner*dx(202) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, xi, xi_)
    Jac = df.derivative(F, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))
    bcs = [df.DirichletBC(V, zero, boundaries, 101)]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
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
        solver.solve()
        print("Time: ", time.time() - iter_start)
    
    print("Total Time: ", time.time() - total_start, "s")
    u, p, Jt = xi.split() 

    # Projections
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    # Extract values in outer region
    submesh = df.SubMesh(mesh, domains, 201)
    u_outer = df.interpolate(u, df.FunctionSpace(submesh, V_ele))
    F_outer = df.interpolate(F, df.TensorFunctionSpace(submesh, "CG", 1, shape=(2, 2)))
    p_outer = df.interpolate(p, df.FunctionSpace(submesh, W_ele))
    J_outer = df.interpolate(Jt, df.FunctionSpace(submesh, R_ele))

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
    J_outer.rename("J","Jacobian")
    J_file.write(J_outer)

    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p_outer.rename("p","pressure")
    p_file.write(p_outer)


if __name__ == "__main__":
    main()