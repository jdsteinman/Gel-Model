from dolfin import *
import numpy as np
import sys
import os

parameters['linear_algebra_backend'] = 'PETSc'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 2

"""
Written by: John Steinman
Simulation of hyperelastic thin plate with circular hole. 
Inner boundary is prescribed a displacement and outer boundary is fixed.

Test Cases:
    a. Uniform radial contraction
    b. Contraction in y only
    c. Contraction in y, expansion in x

Outputs:
    - Displacement (u) : xdmf
    - Deformation Gradient (F) : xdmf
"""

def main():

    # Select case
    case = "a"

    # Output folder
    output_folder = "./output" + case + "/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Mesh
    mesh = Mesh("./meshes/plate_with_hole.xml")

    # Function Space
    U = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, U)

    du, w = TrialFunction(V), TestFunction(V)    # Incremental displacement
    u = Function(V)
    u.vector()[:] = 0

    # BCs
    mf = cpp.mesh.MeshFunctionSizet(mesh, mesh.topology().dim()-1)
    mf.set_all(0)
    inner = inner_boundary()
    inner.mark(mf, 1)
    outer = outer_boundary()
    outer.mark(mf, 2)

    zero = Constant((0.0, 0.0))
    u_a = Expression(["-x[0]/10","-x[1]/10"], degree=2)
    u_b = Expression("-x[1]/10", degree=2)
    u_c = Expression(["x[0]/10","-x[1]/10"], degree=2)

    outer_bc = DirichletBC(V, zero, mf, 2)

    if case=="a":
        inner_bc = DirichletBC(V, u_a, mf, 1)
    elif case=="b":
        inner_bc = DirichletBC(V.sub(1), u_b, mf, 1)
    elif case=="c":
        inner_bc = DirichletBC(V, u_c, mf, 1)
    else:
        print("Invalid case: ", case)
        sys.exit()

    bcs = [inner_bc, outer_bc]

    # Solver
    u, du = solver_call(u, du, w, bcs)

    F = Identity(2) + grad(u)
    F = project(F, V=TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2)), solver_type = 'cg', preconditioner_type = 'amg')

    # Outputs
    disp_file = XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

    F_file = XDMFFile(output_folder + "F.xdmf")
    F.rename("F","deformation gradient")
    F_file.write(F)

class outer_boundary(SubDomain):
    def inside(self, x, on_boundary):
        cond = abs(x[0])<20 and abs(x[1])<20
        return on_boundary and not cond

class inner_boundary(SubDomain):
    def inside(self, x, on_boundary):
        cond = abs(x[0])<20 and abs(x[1])<20 
        return on_boundary and cond

def solver_call(u, du, w, bcs):
    ## Kinematics
    B = Constant((0, 0))  # Body force per unit volume
    T = Constant((0, 0))  # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    # Material parameters
    nu = 0.49                        # Poisson's ratio
    mu = Constant(325 * 10**12)      # Bulk Modulus
    lmbda = 2*nu*mu/ (1-2*nu)        # 1st Lame Parameter

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, w)
    Jac = derivative(F, u, du)

    # Create nonlinear variational problem and solve
    problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=Jac)

    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-2
    solver.parameters['newton_solver']['linear_solver'] = 'gmres'
    solver.parameters['newton_solver']['preconditioner'] = 'jacobi'
    solver.solve()

    return u, du

if __name__ == "__main__":
    main()
