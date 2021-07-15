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

    params['output_folder'] = './output/eigenstrain/'

    params['mesh'] = df.Mesh("../cell_meshes/bird/inclusion.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], \
        "../cell_meshes/bird/inclusion_physical_region.xml")

    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], \
        "../cell_meshes/bird/inclusion_facet_region.xml")

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

    # Material parameters
    c1 = df.Constant(1.0)
    c2 = df.Constant(1000.0)

    mu = c1*2
    lmbda = 

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds


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
    psi_outer = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)
    psi_inner = inner(sigma(u), eps(u)-eigenstrain)

    # Total potential energy
    Pi = psi_outer*dx(301) + psi_inner*dx(302) - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    bcs = [df.DirichletBC(V, zero, boundaries, 201)]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
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