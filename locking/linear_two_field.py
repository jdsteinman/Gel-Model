import dolfin as df
from dolfin import sym, grad, div, inner, Identity
import matplotlib.pyplot as plt
import os

def main():
    # Parameters
    E = 70.0e6  # Youngs modulus
    nu = 0.4999  # Poissons ratio
    lmbda, mu = E*nu/(1 + nu)/(1 - 2*nu), E/2/(1 + nu)  # Lame's constant

    # Strain and Stress
    def eps(v):
        return sym(grad(v))

    def sigma(v, p):
        return p*Identity(2) + 2.0*mu*eps(v)

    # Load
    g_int = -1e7

    # Geometry
    l_x, l_y = 5.0, 5.0  # Domain dimensions
    n_x, n_y = 20, 20  # Number of elements

    mesh = df.RectangleMesh(df.Point(0.0,0.0), df.Point(l_x, l_y), n_x, n_y)

    # Definition of Neumann condition domain
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = df.AutoSubDomain(lambda x: df.near(x[1], 5.0))
    top.mark(boundaries, 1)

    # Measures
    dx = df.Measure("dx", domain=mesh)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Function Spaces
    P1 = df.VectorElement('P', mesh.ufl_cell(), 2)
    P2 = df.FiniteElement('P', mesh.ufl_cell(), 1)
    element = df.MixedElement([P1, P2])
    V = df.FunctionSpace(mesh, element)
    g = df.Constant((0.0, g_int))

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[1], 0.0))

    bc = df.DirichletBC(V.sub(0), df.Constant((0.0, 0.0)), bottom)

    # Weak Form
    u_test, p_test = df.TestFunctions(V)
    u_tr, p_tr = df.TrialFunctions(V)

    a = inner(eps(u_test), sigma(u_tr, p_tr))*dx
    a += -p_test*(div(u_tr) + p_tr/lmbda)*dx
    L = inner(g, u_test)*ds(1)

    # Solver
    sol = df.Function(V)
    df.solve(a == L, sol, bc)

    # Post-process
    plot1 = df.plot(sol.sub(0), mode="displacement")
    plt.colorbar(plot1)
    plt.show()

    plot2 = df.plot(sol.sub(1))
    plt.colorbar(plot2)
    plt.show()

    # Outputs
    output_folder = "output/linear_two_field/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    u = sol.sub(0)
    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

if __name__=="__main__":
    main()