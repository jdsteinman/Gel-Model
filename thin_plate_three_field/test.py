import dolfin as df
import numpy as np
import sys
import os
import time

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = False 
df.parameters['form_compiler']['quadrature_degree'] = 2
df.parameters['krylov_solver']['absolute_tolerance' ]= 1E-8
df.parameters['krylov_solver']['relative_tolerance'] = 1E-6
df.parameters['krylov_solver']['maximum_iterations'] = 100000

"""
Written by: John Steinman
Simulation of hyperelastic thin plate with circular hole. 

Outputs:
    - Displacement (u) : xdmf
    - Deformation Gradient (F) : xdmf
"""

def main():

    params = {}

    params['output_folder'] = './output/test/'

    params['mesh'] = df.Mesh("./meshes/plate_with_hole.xml")

    params['physical_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_physical_region.xml")
    params['facet_region'] = df.MeshFunction("size_t", params["mesh"], "./meshes/plate_with_hole_facet_region.xml")

    solver_call(params)

def solver_call(params):
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

    V_u = W.sub(0).collapse()
    V_p = W.sub(1).collapse()
    V_J = W.sub(2).collapse()

    # Trial and Test functions
    dxi = df.TrialFunction(W)
    xi_ = df.TestFunction(W)

    # Functions from most recent iteration
    xi = df.Function(W)
    xi.rename('xi','mixed solution') 

    # set initial values
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V_u)
    p_0 = df.interpolate(df.Constant((0.0)), V_p)
    J_0 = df.interpolate(df.Constant((1.0)), V_J)
    df.assign(xi, [u_0, p_0, J_0])

    # Variational forms
    u, p, J = df.split(xi)

    # Kinematics
    d = mesh.geometry().dim()
    I = df.Identity(d)
    F = I+df.grad(u)
    C = F.T*F
    b = F*F.T
    E = (C-I)/2
    IC = df.tr(C)
    C_bar = C/J**(2/d)
    b_bar = b/J**(2/d)
    IC_bar = df.tr(C_bar)
    Ju = df.det(F)

    c1 = df.Constant(1.)
    c2 = df.Constant(100.)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx 

    # Compute stationary of Pi 
    R = df.derivative(Pi, xi, xi_)
    Jac = df.derivative(R, xi, dxi)

    # Boundary Conditions
    zero = df.Constant((0.0, 0.0))
    u_c = df.Expression(["x[0]/100","-x[1]/100"], degree=2)
    inner_bc = df.DirichletBC(W.sub(0), u_c, boundaries, 102)
    outer_bc = df.DirichletBC(W.sub(0), zero, boundaries, 101)
    bcs = [inner_bc]

    # Create nonlinear variational problem and solve
    problem = df.NonlinearVariationalProblem(R, xi, bcs=bcs, J=Jac)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    
    start = time.time()
    solver.solve()
    print("Total Time: ", time.time() - start, "s")

    # Projections
    u, p, J = xi.split()
    F = df.Identity(2) + df.grad(u)
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

    disp_File = df.File(os.path.join(output_folder,'u.pvd'))
    disp_File << (u,1)

if __name__ == "__main__":
    main()
