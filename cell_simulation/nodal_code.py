import dolfin as df
import numpy as np
import nodal_tools as nt
import time
import os

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 10000

def main():
    params = {}
    NN = 500
    mu = 108
    kappa = 7500

    params["mu"] = mu
    params["kappa"] = kappa

    params['output_folder'] = './output/graded/' 

    params['mesh'] = "../cell_meshes/bird/hole.xdmf"
    params['domains'] = "../cell_meshes/bird/hole_domains.xdmf"
    params['boundaries'] = "../cell_meshes/bird/hole_boundaries.xdmf"

    params['surface_nodes'] = np.loadtxt('../cell_meshes/bird/surface_vertices_' + str(NN) + '.txt')
    params['surface_faces'] = np.loadtxt('../cell_meshes/bird/surface_faces_' + str(NN) + '.txt', dtype=int)
    params['displacements'] = np.loadtxt('../cell_data/bird/surface_displacements_' + str(NN) + '.txt')
    params['beads_init'] = np.loadtxt('../cell_data/bird/beads_init_filtered.txt')
    params['beads_final'] = np.loadtxt('../cell_data/bird/beads_final_filtered.txt')

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

    surface_nodes = params['surface_nodes']
    surface_faces = params['surface_faces']
    displacements = params['displacements']

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
    mu = params["mu"]
    mu = nt.shear_modulus(surface_nodes, mu, 40, 1, method="power")
    kappa = params["kappa"]
    # kappa = 2*mu*(1+nu)/(3*(1-2*nu))

    c1 = mu/2
    c2 = df.Constant(kappa)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    midpoints = nt.get_midpoints(surface_nodes, surface_faces)
    midpoint_disp = nt.get_midpoint_disp(displacements, surface_faces)
    face_map = nt.get_face_mapping(midpoints, mesh, boundaries, 202)
    face2disp = dict(zip(face_map, midpoint_disp))

    zero = df.Constant((0.0, 0.0, 0.0))
    bf = nt.BoundaryFunc(mesh, face2disp, 1)

    outer_bc = df.DirichletBC(V_u, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V_u, bf, boundaries, 202)
    bcs = [outer_bc, inner_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['relative_tolerance'] = 1E-2

    # Solve
    chunks = 20
    total_start = time.time()
    for i in range(chunks):
        print("Solver call: ", i)
        start = time.time()

        bf.scalar = (i+1)/chunks

        solver.solve()
        print("Time: ", time.time()-start) 
    print("Total Time: ", time.time() - total_start, "s")

    u, p, J = xi.split(True)

    # Projections
    F_proj = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    psi_proj = df.project(psi, df.FunctionSpace(mesh, "DG", 0))

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Interpolate at bead locations
    beads_init = params['beads_init']
    beads_final = params['beads_final']
    points = []
    u_sim = []
    u_data = []

    for i, (init, final) in enumerate(zip(beads_init, beads_final)):
        try:
            u_sim.append(u(init))
            u_data.append(final-init)
            points.append(init)
        except:
            print(i)

    u_sim = np.array(u_sim)
    u_data = np.array(u_data)
    points = np.array(points)
    nt.write_vtk(output_folder+"bead_displacements", points, u_sim, u_data)
    nt.write_txt(output_folder+"bead_displacements_sim.txt", points, u_sim)
    nt.write_txt(output_folder+"bead_displacements_data.txt", points, u_data)
    
    u.rename("U","displacement")
    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    disp_file.write(u)

    F_proj.rename("F","deformation gradient")
    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F_file.write(F_proj)

    J.rename("J","Jacobian")
    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J_file.write(J)

    p.rename("p","pressure")
    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p_file.write(p)

    psi_proj.rename("psi", "strain energy density")
    psi_file = df.XDMFFile(output_folder + "psi.xdmf")
    psi_file.write(psi_proj)

if __name__=="__main__":
    main()