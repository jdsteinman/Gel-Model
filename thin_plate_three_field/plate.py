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
Simulation of three-field, hyperelastic thin plate.

Outputs:
    - Displacement (u) : xdmf
    - Deformation Gradient (F) : xdmf
    - Pressure (p) : xdmf
    - Jacobian (J) : xdmf
"""

def main():

    params = {}

    params['case'] = 'axial'
    # params['case'] = 'biaxial'
    # params['case'] = 'shear'

    params['output_folder'] = './output/plate/' + params['case'] + '/'

    params['mesh'] = df.Mesh("./meshes/plate.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_facet_region.xml")

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
    V = df.VectorElement('Lagrange', mesh.ufl_cell(), 1)
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
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Green tensor

    # Invariants of deformation tensors
    J = det(F)
    I = tr(b) 
    Ibar = I * J**-1

    # Material parameters
    c1 = df.Constant(50.0)
    c2 = df.Constant(0.5)

    # Boundary Conditions
    bcs = []
    body_term=0
    traction_term=0

    case = params["case"]
    if case == "axial":
        Ty = df.Constant((0.0, 1.0e-1))
        traction_term = dot(Ty, u)*ds(103)
        bc_bottom = df.DirichletBC(V, df.Constant((0.0, 0.0)), boundaries, 101)  
        bcs = [bc_bottom]
    elif case == "biaxial":
        Tx = df.Constant((1.0e1, 0.0))
        Ty = df.Constant((0.0,  1.0e1))
        traction_term = dot(Tx, u)*ds(102) + dot(Ty, u)*ds(103)

        bc_bottom = df.DirichletBC(V.sub(1), df.Constant(0.0), boundaries, 101) 
        bc_left = df.DirichletBC(V.sub(0), df.Constant(0.0), boundaries, 104) 
        bcs.append(bc_bottom)
        bcs.append(bc_left)
    elif case == "shear":
        Tx = df.Constant((1.0e-2, 0.0))
        traction_term = dot(Tx, u)*ds(103)

        bc_bottom = df.DirichletBC(V, df.Constant((0.0,0.0)), boundaries, 101) 
        bcs.append(bc_bottom)
    else:
        print("Invalid case: ", case)
        sys.exit()

    # Stored strain energy density (three-field formulation)
    psi_1 = c1*(Jt**2-1-2*ln(Jt))
    psi_2 = c2*(Ibar-2)
    psi_3 = p*(J-Jt)
    psi = psi_1 + psi_2 + psi_3

    # Total potential energy
    Pi = psi*dx - body_term - traction_term

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, xi)
    Jac = df.derivative(F, xi)

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    
    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    u, p, Jt = xi.split(True)    

    # Projections
    F = Identity(2) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

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

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    Jt.rename("J","Jacobian")
    J_file.write(Jt)

    p_file = df.XDMFFile(output_folder + "p.xdmf")
    p.rename("p","pressure")
    p_file.write(p)

if __name__ == "__main__":
    main()
