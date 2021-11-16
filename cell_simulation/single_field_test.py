import dolfin as df
import numpy as np
import nodal_tools as nt
import time
import os
import sys
from mpi4py import MPI
from shutil import copyfile 

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

    params['mesh'] = "../cell_meshes/bird/hole_coarse.xdmf"
    params['domains'] = "../cell_meshes/bird/hole_coarse_domains.xdmf"
    params['boundaries'] = "../cell_meshes/bird/hole_coarse_boundaries.xdmf"

    params['mesh_init'] = "../cell_meshes/bird/hole_coarse.xdmf"
    params['u_init'] = "./output/single_field/coarse/homogeneous/u_out.xdmf"

    params['mu'] = 100e-6
    params['nu'] = 0.49
    # params['degradation'] = np.loadtxt("./output/single_field/coarse/homogeneous/degradation.txt")
    params['degradation'] = np.loadtxt("./output/single_field/iteration_2/degradation.txt")

    params['surface_nodes'] = np.loadtxt('../cell_meshes/bird/cell_surface_500_vertices.txt')
    params['surface_faces'] = np.loadtxt('../cell_meshes/bird/cell_surface_500_faces.txt', int)
    params['displacements'] = np.loadtxt('../cell_data/bird/surface_displacements_500.csv')

    params['chunks'] = 5

    params['output_folder'] = "./output/single_field/iteration_3"
    # params['output_folder'] = './output/single_field/coarse/homogeneous'
    # params['output_folder'] = './output/single_field/coarse/FGM-mu'
    # params['output_folder'] = './output/single_field/coarse/FGM-psi'
    # params['output_folder'] = './output/single_field/test/psi'

    solver_call(params)

def solver_call(params):
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size()>1:
        df.set_log_level(40)  # Mute output

    # Gel Volume Mesh
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

    # Initialization Mesh
    mesh_init = df.Mesh()
    with df.XDMFFile(params["mesh_init"]) as infile:
        infile.read(mesh_init)

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    element_u = df.VectorElement("CG", mesh.ufl_cell(), 2)
    V = df.FunctionSpace(mesh, element_u)
    u = df.Function(V)
    u.rename('u','displacement')
    u_ = df.TestFunction(V)
    du = df.TrialFunction(V)

    # Initialize  
    V_init = df.VectorFunctionSpace(mesh_init, "CG", 2)
    u_init = df.Function(V_init)  
    u_init_file = df.XDMFFile(params["u_init"])
    u_init_file.read_checkpoint(u_init, "u", 0)
    u_init.set_allow_extrapolation(True)

    u_0 = df.interpolate(u_init, V)
    df.assign(u, u_0)

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)                 # Jacobian
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition
    IC_bar = df.tr(C_bar)          # Invariant

    # Material parameters
    mu_ff = params["mu"]
    nu_ff = params["nu"]
    kappa_ff = 2*mu_ff*(1+nu_ff)/3/(1-2*nu_ff)
    
    # Degradation
    V_1 = df.FunctionSpace(mesh, "CG", 1)
    degradation = df.Function(V_1)
    v2d = df.vertex_to_dof_map(V_1)
    degradation.vector()[v2d] = params["degradation"]
    degradation.set_allow_extrapolation(True)

    mu = nt.ElasticModulus(mu_ff, degradation)
    nu = nt.ElasticModulus(nu_ff, degradation) 
    kappa = 2*mu*(1+nu)/3/(1-2*nu)

    c1 = mu/2
    c2 = kappa_ff

    # Stored strain energy density (Neo-Hookean formulation)
    psi = c1*(IC_bar-d) + c2*(Ju**2-1-2*df.ln(Ju))/4 

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, u, u_)
    Dres = df.derivative(res, u, du)

    # Boundary Conditions
    surface_nodes = params['surface_nodes']
    surface_faces = params['surface_faces']
    displacements = params['displacements']

    midpoints = nt.get_midpoints(surface_nodes, surface_faces)
    midpoint_disp = nt.get_midpoint_disp(displacements, surface_faces)
    face_map = nt.get_face_mapping(midpoints, mesh, boundaries, 202)
    face2disp = dict(zip(face_map, midpoint_disp))

    zero = df.Constant((0.0, 0.0, 0.0))
    bf = nt.BoundaryFunc(mesh, face2disp, 0)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, bf, boundaries, 202)
    bcs = [inner_bc, outer_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, u, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver']  = 'gmres'
    solver.parameters['newton_solver']['preconditioner']  = 'hypre_amg'

    # MPI
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
    chunks = params["chunks"]
    u.vector()[:]*=1/chunks

    sys.stdout.flush()
    total_start = time.time() 
    for i in range(chunks):
        start = time.time()
        if rank == 0: print("    Iter: ", i)
        sys.stdout.flush()  

        bf.scalar = (i+1)/chunks
        solver.solve()

        end = time.time()
        time_elapsed = end - start
        if rank == 0: print('    Time elapsed = {:2.1f}s'.format(time_elapsed))
        sys.stdout.flush()  

    # Projections
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    J = df.project(Ju, V=df.FunctionSpace(mesh, "DG", 0))
    mu = df.project(mu, V=df.FunctionSpace(mesh, "CG", 1))

    # Outputs
    output_folder = params["output_folder"]
    if rank==0:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    u_file = df.XDMFFile(os.path.join(output_folder, "u.xdmf"))
    u.rename("u","displacement")
    u_file.write(u)

    F_file = df.XDMFFile(os.path.join(output_folder, "F.xdmf"))
    F.rename("F","deformation gradient")
    F_file.write(F)

    J_file = df.XDMFFile(os.path.join(output_folder, "J.xdmf"))
    J.rename("J","Jacobian")
    J_file.write(J)

    mu_file = df.XDMFFile(os.path.join(output_folder, "mu.xdmf"))
    mu.rename("mu","shear modulus")
    mu_file.write(mu)

    out_file = df.XDMFFile((os.path.join(output_folder, "u_out.xdmf")))
    out_file.write_checkpoint(u, "u", 0)   #Not appending

    if rank==0:
        python_file = os.path.basename(__file__)
        copyfile(python_file, os.path.join(output_folder, python_file))

        with open(os.path.join(output_folder,"log_params.txt"), "w+") as f:
            f.write("Mesh: {:s}\n".format(params["mesh"]))
            f.write("No. Elements: {:d}\n".format(int(ele_sum)))
            f.write("No. Processors: {:d}\n".format(int(comm.Get_size())))
            f.write("mu_ff = {:e}\n".format(mu_ff))
            f.write("nu_ff = {:e}\n".format(nu_ff))
            f.write("Total Time = {:f}s\n".format(time.time()-total_start))

        print("Done")

if __name__=="__main__":
    main()
