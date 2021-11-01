import dolfin as df
import numpy as np
import nodal_tools as nt
import time
import os
import sys
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

    params['mesh'] = "../cell_meshes/bird/hole.xdmf"
    params['domains'] = "../cell_meshes/bird/hole_domains.xdmf"
    params['boundaries'] = "../cell_meshes/bird/hole_boundaries.xdmf"

    params['L'] = 250
    params['mu_ff'] = 100e-6
    params['nu'] = 0.49

    params['surface_nodes'] = np.loadtxt('../cell_meshes/bird/cell_surface_1000_vertices.txt')
    params['surface_faces'] = np.loadtxt('../cell_meshes/bird/cell_surface_1000_faces.txt', int)
    params['displacements'] = np.loadtxt('../cell_data/bird/surface_displacements_1000.csv')
    params['beads_init'] = np.loadtxt('../cell_data/bird/beads_init_filtered.txt')
    params['beads_final'] = np.loadtxt('../cell_data/bird/beads_final_filtered.txt')

    params['output_folder'] = './output/bird/single_field/fixed'

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
    L = params["L"]

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

    # Set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0,0.0)), V)
    df.assign(u, u_0)

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
    psi = c1*(IC_bar-d) + c2*(Ju**2-1-2*df.ln(Ju))/4 

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, u, u_)
    Dres = df.derivative(res, u, du)

    # Subdomains
    length = params["L"]
    centroid = [72.20045715, 72.90093189, 47.46392168]
    cx, cy, cz = centroid

    xboundary = df.CompiledSubDomain("near(abs(x[0]-cx), R, 1) && near(abs(x[1]-cy), 0, 1) && near(abs(x[2]-cz), 0, 1)", R=length/2, cx=cx, cy=cy, cz=cz)
    yboundary = df.CompiledSubDomain("near(abs(x[0]-cx), 0, 1) && near(abs(x[1]-cy), R, 1) && near(abs(x[2]-cz), 0, 1)", R=length/2, cx=cx, cy=cy, cz=cz)
    zboundary = df.CompiledSubDomain("near(abs(x[0]-cx), 0, 1) && near(abs(x[1]-cy), 0, 1) && near(abs(x[2]-cz), R, 1)", R=length/2, cx=cx, cy=cy, cz=cz)
    corners   = df.CompiledSubDomain("near(abs(x[0]-cx), R, 1) && near(abs(x[1]-cy), R, 1) && near(abs(x[2]-cz), R, 1)", R=length/2, cx=cx, cy=cy, cz=cz)

    # Boundary Conditions
    midpoints = nt.get_midpoints(surface_nodes, surface_faces)
    midpoint_disp = nt.get_midpoint_disp(displacements, surface_faces)
    face_map = nt.get_face_mapping(midpoints, mesh, boundaries, 202)
    face2disp = dict(zip(face_map, midpoint_disp))

    zero = df.Constant((0.0, 0.0, 0.0))
    bf = nt.BoundaryFunc(mesh, face2disp, 0)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, bf, boundaries, 202)
    corners_bc =  df.DirichletBC(V, zero, corners, method="pointwise")
    bc_x_1 = df.DirichletBC(V.sub(1), df.Constant(0), xboundary, method="pointwise")
    bc_x_2 = df.DirichletBC(V.sub(2), df.Constant(0), xboundary, method="pointwise")
    bc_y_1 = df.DirichletBC(V.sub(0), df.Constant(0), yboundary, method="pointwise")
    bc_y_2 = df.DirichletBC(V.sub(2), df.Constant(0), yboundary, method="pointwise")
    bc_z_1 = df.DirichletBC(V.sub(0), df.Constant(0), zboundary, method="pointwise")
    bc_z_2 = df.DirichletBC(V.sub(1), df.Constant(0), zboundary, method="pointwise")
    #bcs = [inner_bc, bc_x_1, bc_x_2, bc_y_1, bc_y_2, bc_z_1, bc_z_2]
    bcs = [inner_bc, outer_bc]
    #bcs = [inner_bc, corners_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, u, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver']  = 'mumps'

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
    chunks = 5
    sys.stdout.flush()
    total_start = time.time() 
    for i in range(chunks):
        start = time.time()
        if rank==0:    
            print("Time: ", time.time()-start)
            print("Iteration: ", i)

        bf.scalar = (i+1)/chunks
        solver.solve()

        end = time.time()
        time_elapsed = end - start
        if rank == 0:
            print("    Iter: ", i)
            print('    Time elapsed = {:2.1f}s'.format(time_elapsed))
        sys.stdout.flush()  

    # Projections
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')
    J = df.project(Ju, V=df.FunctionSpace(mesh, "DG", 0))

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

    out_file = df.XDMFFile((os.path.join(output_folder, "u_out.xdmf")))
    out_file.write_checkpoint(u, "u", 0)   #Not appending

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
