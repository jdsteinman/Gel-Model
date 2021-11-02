import dolfin as df
import numpy as np
import time
import os
import sys
from fenics_tools import *
from mpi4py import MPI

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 3
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-4
df.parameters['krylov_solver']['maximum_iterations'] = 100000

def main():
    params = {}

    params['mesh'] = "./meshes/hole.xdmf"
    params['domains'] = "./meshes/hole_domains.xdmf"
    params['boundaries'] = "./meshes/hole_boundaries.xdmf"

    params['mu_ff'] = 100e-6
    params['nu'] = 0.49

    params['u_init'] = "./data/u_init.xdmf"
    params['mesh_init'] = "./meshes/hole_coarse.xdmf"

    params['surface_nodes'] = np.loadtxt('./data/cell_surface_500_vertices.txt')
    params['surface_faces'] = np.loadtxt('./data/cell_surface_500_faces.txt', int)
    params['displacements'] = np.loadtxt('./data/surface_displacements_500.txt')

    params['output_folder'] = './output'

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
   
    mesh_init = df.Mesh()
    with df.XDMFFile(params["mesh_init"]) as infile:
        infile.read(mesh_init)
 
    V_init = df.VectorFunctionSpace(mesh_init, "CG", 2)
    u_init = df.Function(V_init)
    
    init_file = df.XDMFFile(params["u_init"])
    init_file.read_checkpoint(u_init, "u", 0)

    u_init.set_allow_extrapolation(True)
    u_0 = df.interpolate(u_init, V_u.collapse())
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
    mu = params["mu_ff"]
    nu = params["nu"]
    kappa = 2*mu*(1+nu)/3/(1-2*nu)

    c1 = df.Constant(mu/2)
    c2 = df.Constant(kappa)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi 
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    midpoints = get_midpoints(surface_nodes, surface_faces)
    midpoint_disp = get_midpoint_disp(displacements, surface_faces)
    face_map = get_face_mapping(midpoints, mesh, boundaries, 202)
    face2disp = dict(zip(face_map, midpoint_disp))

    zero = df.Constant((0.0, 0.0, 0.0))
    bf = BoundaryFunc(mesh, face2disp, 0)

    outer_bc = df.DirichletBC(V_u, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V_u, bf, boundaries, 202)
    bcs = [inner_bc, outer_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver']  = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'hypre_amg'

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

        print("Mesh: ", params["mesh"])
        print('Total number of elements = {:d}'.format(int(ele_sum)))
        print("Solving =========================")

    # Solve
    chunks = 10
    sys.stdout.flush()
    total_start = time.time() 
    for i in range(chunks):
        start = time.time()
        if rank==0:    
            print("Time: ", time.time()-start)
            print("Iteration: ", i)

        # Increment
        bf.scalar = (i+1)/chunks
        solver.solve()

        end = time.time()
        time_elapsed = end - start
        if rank == 0:
            print('    Time elapsed = {:2.1f}s'.format(time_elapsed))
        sys.stdout.flush()  

    u, p, J = xi.split(True)

    # Projections
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

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

    with open(os.path.join(output_folder,"log_params.txt"), "w+") as f:
        f.write("Mesh: {:s}\n".format(params["mesh"]))
        f.write("No. Elements: {:d}\n".format(int(ele_sum)))
        f.write("No. Processors: {:d}\n".format(int(comm.Get_size())))
        f.write("mu_ff = {:e}\n".format(mu))
        f.write("kappa = {:e}\n".format(kappa))
        f.write("Total Time = {:f}s\n".format(time.time()-total_start))

    if rank==0: print("Done")

if __name__=="__main__":
    main()
