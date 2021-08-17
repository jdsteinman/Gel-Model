import dolfin as df
import os
import time
import numpy as np
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt

df.parameters['linear_algebra_backend'] = 'PETSc'
df.parameters['form_compiler']['representation'] = 'uflacs'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2

def solve(mu, kappa, stretch):
    # Mesh
    mesh = df.UnitCubeMesh(5, 5, 5)
    mesh.coordinates()[:] = mesh.coordinates()[:]-0.5
    dim = mesh.geometry().dim()

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
    zero = df.Constant(np.zeros(dim))
    V_u = V.sub(0)
    V_p = V.sub(1)
    V_J = V.sub(2)
    u_0 = df.interpolate(zero, V_u.collapse())
    p_0 = df.interpolate(df.Constant(0.0), V_p.collapse())
    J_0 = df.interpolate(df.Constant(1.), V_J.collapse())

    df.assign(xi,[u_0,p_0,J_0])
    u,p,J = df.split(xi)

    # Kinematics
    B = zero                       # Body force per unit volume
    T = zero                       # Traction force on the boundary
    I = df.Identity(dim)             # Identity tensor
    F = I + df.grad(u)             # Deformation gradient
    Ju = df.det(F)
    C = F.T*F                      # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/dim)            # Isochoric decomposition

    # Invariants of deformation tensors
    IC_bar = df.tr(C_bar)

    # Material parameters
    c1 = df.Constant(mu/2)
    c2 = df.Constant(kappa)

    # Stored strain energy density (mixed formulation)
    psi = c1*(IC_bar-dim) + c2*(J**2-1-2*df.ln(J))/4 + p*(Ju-J)

    # Total potential energy
    Pi = psi*dx - df.dot(B, u)*dx - df.dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of w)
    res = df.derivative(Pi, xi, xi_)
    Dres = df.derivative(res, xi, dxi)

    # Solve for analytical Jacobian and stress
    def func(J):
        return kappa/2*J**(8/3)-kappa/2*J**(5/3)+mu/2/3/stretch*J-mu/2/3*stretch**2
    
    J_analytical = root_scalar(func, bracket=[0,5], method="brentq")
    J_analytical = J_analytical.root

    sigma_analytical = mu/(J_analytical**(5/3))*(stretch**2-J_analytical/stretch)

    # Boundary Conditions
    stretch_z_pos  = df.Expression("a*t",  a=(stretch-1)/2, t=0, degree=0)
    stretch_z_neg  = df.Expression("a*t",  a=-(stretch-1)/2, t=0, degree=0)
    stretch_xy_pos  = df.Expression("a*t",  a=np.sqrt(J_analytical/stretch), t=0, degree=0)
    stretch_xy_neg  = df.Expression("a*t",  a=-np.sqrt(J_analytical/stretch), t=0, degree=0)

    front  = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = -0.5)
    back   = df.CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.5)
    right  = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.5)
    left   = df.CompiledSubDomain("near(x[1], side) && on_boundary", side = -0.5)
    top    = df.CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.5)
    bottom = df.CompiledSubDomain("near(x[2], side) && on_boundary", side = -0.5)
    
    bc_front  = df.DirichletBC(V_u.sub(0), stretch_xy_pos, front)
    bc_back   = df.DirichletBC(V_u.sub(0), stretch_xy_neg, back)
    bc_right  = df.DirichletBC(V_u.sub(1), stretch_xy_neg, right)
    bc_left   = df.DirichletBC(V_u.sub(1), stretch_xy_pos, left)
    bc_top    = df.DirichletBC(V_u.sub(2), stretch_z_pos, top)
    bc_bottom = df.DirichletBC(V_u.sub(2), stretch_z_neg, bottom)
    bcs = [bc_front, bc_back, bc_right, bc_left, bc_top, bc_bottom]

    # Create nonlinear variational problem
    problem = df.NonlinearVariationalProblem(res, xi, bcs=bcs, J=Dres)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'

    # Solve
    chunks = 2
    total_start = time.time()
    for i in range(chunks):
        print("Solver call: ", i)
        start = time.time()

        stretch_z_pos.t = (i+1)/chunks
        stretch_z_neg.t = (i+1)/chunks
        stretch_xy_pos.t = (i+1)/chunks
        stretch_xy_neg.t = (i+1)/chunks

        solver.solve()
        print("Time: ", time.time()-start) 
    print("Total Time: ", time.time() - total_start, "s")

    u, p, J = xi.split(True)

    # Calculated Stresses
    F = df.Identity(dim) + df.grad(u)
    F = df.variable(F)
    Ju = df.det(F)
    C = F.T*F                # Right Cauchy-Green tensor
    C_bar = C/Ju**(2/dim)    # Isochoric decomposition
    IC_bar = df.tr(C_bar)

    psi = c1*(IC_bar-dim) + c2*(Ju**2-1-2*df.ln(Ju))/4 + p*(Ju-J)
    sigma = 1/df.det(F)*df.diff(psi,F)*F.T
    
    # Projections
    F = df.Identity(dim) + df.grad(u)
    F = df.project(F, df.TensorFunctionSpace(mesh, "CG", 1), solver_type = 'cg', preconditioner_type = 'amg')
    sigma = df.project(sigma, df.TensorFunctionSpace(mesh, "CG", 1))
    psi = df.project(psi, df.FunctionSpace(mesh, "CG", 1))

    # Outputs
    output_folder = "./uniaxial/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = df.XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

    J_file = df.XDMFFile(output_folder + "J.xdmf")
    J.rename("J","")
    J_file.write(J)

    sigma_file = df.XDMFFile(output_folder + "sigma.xdmf")
    sigma.rename("Sigma", "True Stress")
    sigma_file.write(sigma)

    psi_file = df.XDMFFile(output_folder + "psi.xdmf")
    psi.rename("psi","")
    psi_file.write(psi)

    sigma_sim = sigma.sub(8)(0.5,0.5,0.5)

    return(sigma_sim, sigma_analytical)

def main():

    mu = 1.5e6
    kappa=0.5e9
    sigma_zz, sigma_true = solve(mu, kappa, stretch=2)
    print("Sigma ZZ: ", sigma_zz)
    print("Expected:", sigma_true)

if __name__ == "__main__":
    main()

