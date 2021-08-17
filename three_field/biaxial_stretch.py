import dolfin as df
import os
import time
import numpy as np
import numpy.linalg as LA
import pandas as pd
from matplotlib import pyplot as plt

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

def main():

    # Mesh
    mesh = df.UnitSquareMesh(100, 100)

    dx = df.Measure("dx")
    ds = df.Measure("ds")

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
    u_0 = df.interpolate(df.Constant((0.0, 0.0)), V_u.collapse())
    p_0 = df.interpolate(df.Constant(0.0), V_p.collapse())
    J_0 = df.interpolate(df.Constant(1.), V_J.collapse())

    df.assign(xi,[u_0,p_0,J_0])
    u,p,J = df.split(xi)

    # Kinematics
    B = df.Constant((0, 0))     # Body force per unit volume
    T = df.Constant((0, 0))     # Traction force on the boundary
    d = u.geometric_dimension()
    I = df.Identity(d)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/d)            # Isochoric decomposition

    # Invariants of deformation tensors
    IC_bar = df.tr(C_bar)

    # Material parameters
    mu = 2.0
    nu = 0.3
    kappa = 2*mu*(1+nu)/(3*(1-2*nu))

    print("c1 = ", mu/2)
    print("c2 = ", kappa/2)

    c1 = df.Constant(mu/2)
    c2 = df.Constant(kappa/2)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-d) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Boundary Conditions
    left =  df.CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
    bottom = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
    top = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)

    bc_left = df.DirichletBC(V_u.sub(0), df.Constant(0.0), left)
    bc_bottom =  df.DirichletBC(V_u.sub(1), df.Constant(0.0), bottom)
    bc_right = df.DirichletBC(V_u.sub(0), df.Constant(0.001), right)
    bc_top = df.DirichletBC(V_u.sub(1), df.Constant(0.001), top)
    
    bcs = [bc_left, bc_bottom, bc_right, bc_top]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.solve()
   
    u, p, J = xi.split(True)

    # Projections
    F = df.Identity(d) + df.grad(u)
    F = df.project(F, V=df.TensorFunctionSpace(mesh, "CG", 1, shape=(d, d)), solver_type = 'cg', preconditioner_type = 'amg')

    # Stresses
    if nu <= 0.45:
        sigma_11 = 4*c1/(3*J^(5/3))*()

    


    # Outputs
    output_folder = "./biaxial/"
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

