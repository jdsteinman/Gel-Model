import dolfin as df
import sys
import os
import time

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

"""
Written by: John Steinman
Simulation of three-field, hyperelastic thin plate with circular hole. 

Outputs:
    - Displacement (u) : xdmf
    - Deformation Gradient (F) : xdmf
    - Pressure (p) : xdmf
    - Jacobian (J) : xdmf
    - Green-Lagrange Strain (E) - xdmf
"""

def main():

    params = {}

    params['case'] = 'dirichlet'
    # params['case'] = 'neumann'

    params['output_folder'] = './output/quarter/' + params['case'] + '/'

    params['mesh'] = df.Mesh("./meshes/quarter_plate_with_hole.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/quarter_plate_with_hole_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/quarter_plate_with_hole_facet_region.xml")

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
    V = df.VectorElement('Lagrange', mesh.ufl_cell(), 2)
    W = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    R = df.FiniteElement('Lagrange', mesh.ufl_cell(), 1)   
    M = df.FunctionSpace(mesh, df.MixedElement([V,W,R]))
    V, W, R = M.split()

    # Trial and Test functions
    dxi = df.TrialFunction(M)
    m = df.TestFunction(M)

    # Functions from most recent iteration
    xi = df.Function(M) 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V.collapse())
    p_0 = df.interpolate(df.Constant((0.0)), W.collapse())
    J_0 = df.interpolate(df.Constant((1.0)), R.collapse())
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

    # Material parameters
    c1 = df.Constant(50.0)
    c2 = df.Constant(0.5)

    # Boundary Conditions
    u_d = df.Expression(["x[0]/100","-x[1]/100"], degree=2)
    traction_term = 0
    body_term = 0

    bc_bottom = df.DirichletBC(V.sub(1), df.Constant(0.0), boundaries, 102)
    bc_left= df.DirichletBC(V.sub(0), df.Constant(0.0), boundaries, 103)
    bc_top = df.DirichletBC(V, df.Constant((0.0, 0.0)), boundaries, 104)
    bc_right = df.DirichletBC(V, df.Constant((0.0, 0.0)), boundaries, 105)
    bcs = [bc_left, bc_bottom, bc_top, bc_right]

    case = params["case"]
    if case=="dirichlet":
        inner_bc = df.DirichletBC(V, u_d, boundaries, 101)
        bcs.append(inner_bc)
    elif case=="neumann":
        traction = df.Expression(["x[0]/c","-x[1]/c"], c=395, degree=2)
        traction_term = dot(traction, u)*ds(101)
    else:
        print("Invalid case: ", case)
        sys.exit()

    # Stored strain energy density (three-field formulation)
    psi_1 = c1*(Jt**2-1-2*ln(Jt))
    psi_2 = c2*(Ibar-2)
    psi_3 = p*(J-Jt)
    psi = psi_1 + psi_2 + psi_3

    p1 = df.project(psi_1, df.FunctionSpace(mesh, "CG", 1))
    p2 = df.project(psi_2, df.FunctionSpace(mesh, "CG", 1))
    p3 = df.project(psi_3, df.FunctionSpace(mesh, "CG", 1))
    print(df.norm(p1))
    print(df.norm(p2))
    print(df.norm(p3))

    # Total potential energy
    Pi = psi*dx(201) - body_term - traction_term

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, xi, m)
    Jac = df.derivative(F, xi, dxi)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    
    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    u, p, Jt = xi.split() 

    # Projections
    F = Identity(2) + grad(u)    # Deformation Gradient Tensor
    E = 0.5*(F.T*F-Identity(2))  # Green-Lagrange Strain Tensor

    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')
    E = df.project(E, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    # Outputs
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    E_file = df.XDMFFile(output_folder + "E.xdmf")
    E.rename("E","Green-Lagrange")
    E_file.write(E)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    Jt.rename("J","Jacobian")
    J_file.write(Jt)

    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p.rename("p","pressure")
    p_file.write(p)

if __name__ == "__main__":
    main()
