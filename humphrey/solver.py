import dolfin as df
import numpy as np
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
"""

def main():

    params = {}

    params['output_folder'] = './output/'

    params['mesh'] = df.Mesh("./meshes/hole.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/hole_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/hole_facet_region.xml")

    solver_call(params)

# Kinematics
def pk1_stress(u, pressure, mu, nu):
    from dolfin import grad, Identity, inv, det
    c1 = mu/2.0

    d = u.geometric_dimension()
    I = Identity(d)
    F = I + grad(u)             # Deformation gradient
    J = det(F)
    C = F.T*F                   # Right Cauchy-Green tensor

    pk2 = 2*c1*I - pressure*inv(C)  # Second PK stress
    return pk2, (J-1)

def solver_call(params):
    from dolfin import ln, dot, det, tr, grad, Identity, inner

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
    M = df.FunctionSpace(mesh, df.MixedElement([V,W]))
    V, W = M.split()

    # Trial and Test functions
    dxi = df.TrialFunction(M)
    du, dp = df.TestFunctions(M)

    # Functions from most recent iteration
    xi = df.Function(M) 
    u, p = df.split(xi)

    # Material parameters
    nu = 0.49                      # Poisson's ratio
    mu = df.Constant(325e12)       # Shear Modulus (2nd Lame)
    lmbda = 2*nu*mu/ (1-2*nu)      # 1st Lame Parameter
    kappa = lmbda+2/3*mu           # bulk modulus
    E = 2*mu*(1+nu)                # Young's modulus

    # Potential Energy
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    
    d = u.geometric_dimension()
    pkstrs, hydpress = pk1_stress(u,p,mu,nu)
    I = Identity(d)
    dgF = I+grad(u)

    F1 = inner(dot(dgF,pkstrs), grad(du))*dx - dot(B, du)*dx - dot(T, du)*ds
    F2 = hydpress*dp*dx 
    F = F1+F2
    Jac = df.derivative(F, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_inner = df.Expression(["t*x[0]/10","t*x[1]/10","t*-2*x[2]/10"], t=0, degree=2)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, u_inner, boundaries, 202)

    bcs = [inner_bc]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(F, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    # solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    # solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    
    ## Run Sim ==================================================================================
    chunks = 1

    total_start = time.time()
    for i in range(chunks):
        iter_start = time.time()
        print("Solver Call: ", i)
        print("----------------")

        # Increment Boundary Conditions
        u_inner.t = (i+1)/chunks
        bcs[-1] = inner_bc

        ## Solver
        solver.solve()

        print("Time: ", time.time() - iter_start)
        print()

    print("Total Time: ", time.time() - total_start)

    u, p = xi.split()    

    # Projections
    F = Identity(3) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

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

if __name__ == "__main__":
    main()
