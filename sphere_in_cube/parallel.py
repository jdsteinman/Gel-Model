import dolfin as df
import os
import sys
import time
import numpy as np
from mpi4py import MPI
from shutil import copyfile

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False
df.parameters['form_compiler']['quadrature_degree'] = 3
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 10000
df.set_log_level(40)

"""
Written by: John Steinman
"""

def main():

    params = {}

    params['output_folder'] = './output/constant/1000x/'

    params['mesh'] = "hole_3"

    params['c1'] = df.Constant(1.0)
    params['c2'] = df.Constant(1000.0)

    solver_call(params)

def solver_call(params):

    # Mesh
    mesh = df.Mesh()
    with df.XDMFFile("meshes/" + params["mesh"] + ".xdmf") as infile:
        infile.read(mesh)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile("meshes/" + params["mesh"] + "_domains.xdmf") as infile:
        infile.read(mvc, "domains") 
    domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile("meshes/" + params["mesh"] + "_boundaries.xdmf") as infile:
        infile.read(mvc, "boundaries") 
    boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    element_u = df.VectorElement("CG",mesh.ufl_cell(),2)
    element_p = df.FiniteElement("DG",mesh.ufl_cell(),0)
    element_J = df.FiniteElement("DG",mesh.ufl_cell(),0)
  
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
    c1 = params["c1"]
    c2 = params["c2"]

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(301) - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_inner = df.Expression(["t*x[0]/10","t*x[1]/10","-t*2*x[2]/10"], t=0, degree=2)

    outer_bc = df.DirichletBC(V_u, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 202)

    bcs = [outer_bc, inner_bc]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    val = np.array(len(mesh.cells()),'d')
    val_sum = np.array(0.,'d')
    comm.Reduce(val, val_sum, op=MPI.SUM, root=0)

    if rank == 0:
        print("Mesh: ", params["mesh"])
        print('Total number of elements = {:d}'.format(int(val_sum)))

    # Solve
    chunks = 1
    start = time.time()
    sys.stdout.flush() 
    for i in range(chunks):
        start = time.time()

        u_inner.t = (i+1)/chunks
        solver.solve()

        end = time.time()
        time_elapsed = end - start

        if rank == 0:
            print("    Iter: ", i)
            print('    Time elapsed = {:2.1f}s'.format(time_elapsed))
        sys.stdout.flush()  

    u, p, J = xi.split(True)

    # Projections
    F = df.Identity(3) + df.grad(u)
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

    if rank == 0:
        print("Results in: ", output_folder)
        print("Done")
        print("========================================")

if __name__ == "__main__":
    main()
