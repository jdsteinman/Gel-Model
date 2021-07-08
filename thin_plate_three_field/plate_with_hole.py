import dolfin as df
import numpy as np
import sys
import os
import time

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
# df.parameters['form_compiler']['optimize'] = True
# df.parameters['form_compiler']['cpp_optimize'] = False 
df.parameters['form_compiler']['quadrature_degree'] = 4
# df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
# df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
# df.parameters['krylov_solver']['maximum_iterations'] = 10000

"""
Written by: John Steinman
Simulation of hyperelastic thin plate with circular hole. 

Test Cases:
    a. Isotropic contraction
    b. Uniaxial contraction
    c. Contraction in y, expansion in x

Outputs:
    - Displacement (u) : xdmf
    - Deformation Gradient (F) : xdmf
"""

def main():

    params = {}

    params['case'] = 'c'

    mesh = "coarse"    
    # params['output_folder'] = './output/plate_with_hole/' + params['case'] + '/'
    params['output_folder'] = './output/plate_with_hole_coarse/'

    params['mesh'] = df.Mesh("./meshes/plate_with_hole_coarse.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_coarse_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_coarse_facet_region.xml")

    solver_call(params)

def solver_call(params):
    from dolfin import ln, dot, det, tr, grad, Identity

    # Mesh
    mesh = params["mesh"]

    domains = params["physical_region"]
    boundaries = params["facet_region"]

    # Measures
    dx = df.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Space
    element_u = df.VectorElement('CG', mesh.ufl_cell(), 2)
    element_p = df.FiniteElement('DG', mesh.ufl_cell(), 0)
    element_J = df.FiniteElement('DG', mesh.ufl_cell(), 0)   
    W = df.FunctionSpace(mesh, df.MixedElement([element_u,element_p,element_J]))
    V_u, V_p, V_J = W.split()

    # Trial and Test functions
    dxi = df.TrialFunction(W)
    xi_ = df.TestFunction(W)

    # Functions from most recent iteration
    xi = df.Function(W)
    xi.rename('xi','mixed solution') 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V_u.collapse())
    p_0 = df.interpolate(df.Constant((0.0)), V_p.collapse())
    J_0 = df.interpolate(df.Constant((1.0)), V_J.collapse())
    df.assign(xi, [u_0, p_0, J_0])

    # Variational forms
    u, p, J = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Constant((0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Greem tensor

    # Invariants of deformation tensors
    Ju  = det(F)
    I = tr(b)
    Ibar = I*Ju**-1

    # Material parameters
    c1 = df.Constant(1.0)
    c2 = df.Constant(1000.0)

    # Stored strain energy density (three-field formulation)
    psi = c1*(Ibar-2) + c2*(J**2-1-2*ln(J)) + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx(201) - dot(B, u)*dx - dot(T, u)*ds

    # Compute stationary of Pi 
    R = df.derivative(Pi, xi, xi_)
    Jac = df.derivative(R, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))

    case = params["case"]
    if case=="a":
        u_inner = df.Expression(["-t*x[0]/2","-t*x[1]/2"], t=0, degree=2)
        inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 102)
    elif case=="b":
        u_inner = df.Expression("-t*x[1]/2", t=0, degree=2)
        inner_bc = df.DirichletBC(V_u.sub(1), u_inner, boundaries, 102)
    elif case=="c":
        u_inner = df.Expression(["t*x[0]/5","-t*x[1]/2"], t=0, degree=2)
        inner_bc = df.DirichletBC(V_u, u_inner, boundaries, 102)
    else:
        print("Invalid case: ", case)
        sys.exit()

    outer_bc = df.DirichletBC(V_u, zero, boundaries, 101)
    bcs = [inner_bc, outer_bc]
    # bcs=[inner_bc]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(R, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'

    chunks = 2
    total_start = time.time()
    for i in range(chunks):
        print("i = ", i)
        u_inner.t = (i+1)/chunks

        start = time.time()
        solver.solve()
        print("Time: ", time.time() - start, "s")

    print("Total Time: ", time.time() - total_start, "s")

    # Projections
    u, p, J = xi.split()
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("u", "displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    J_file = df.XDMFFile(output_folder + "F.xdmf")
    J.rename("J","Jacobian")
    J_file.write(J)

    p_file = df.XDMFFile(output_folder + "F.xdmf")
    p.rename("p","pressure")
    p_file.write(p)


if __name__ == "__main__":
    main()
