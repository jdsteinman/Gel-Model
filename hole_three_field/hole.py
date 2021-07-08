import dolfin as df
import numpy as np
import sys
import os
import time

from ufl.operators import outer

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

    params['mesh'] = df.Mesh("./meshes/hole50.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/hole50_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/hole50_facet_region.xml")

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
    u, p, Jt = xi.split()
    u.vector()[:] = 0.
    p.vector()[:] = 0.
    Jt.vector()[:] = 1.

    # Variational forms
    u, p, Jt = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0, 0))     # Body force per unit volume
    T = df.Constant((0, 0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Greem tensor

    # Invariants of deformation tensors
    J  = det(F)
    I = tr(b)
    Ibar = I*J**(-2./3)

    # Material parameters
    c1 = df.Constant(50.0)
    c2 = df.Constant(0.5)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = c1*(Jt**2-1-2*ln(Jt))/4. + c2*(Ibar-3) + p*(J-Jt)

    # Total potential energy
    Pi = psi*dx(301) - dot(B, u)*dx(301) - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    F = df.derivative(Pi, xi, m)
    Jac = df.derivative(F, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0, 0.0))
    u_inner = df.Expression(["t*x[0]/100","t*x[1]/100","-t*2*x[2]/100"], t=0, degree=1)

    outer_bc = df.DirichletBC(V, zero, boundaries, 201)
    inner_bc = df.DirichletBC(V, u_inner, boundaries, 202)
    bcs = [inner_bc, outer_bc]

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
    u, p, Jt = xi.split()    

    # Projections
    F = Identity(3) + grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(3, 3)), solver_type = 'cg', preconditioner_type = 'amg')

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

if __name__ == "__main__":
    main()