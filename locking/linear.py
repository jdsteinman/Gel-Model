import dolfin as df
from dolfin import sym, grad, inner, tr, Identity
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Parameters
    E = 70.0e6  # Youngs modulus
    nu = 0.4999  # Poissons ratio
    lmbda, mu = E*nu/(1 + nu)/(1 - 2*nu), E/2/(1 + nu)  # Lame's constant

    # Linear elastic model
    def eps(v):
        return sym(grad(v))

    def sigma(v):
        return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

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
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    u_tr = df.TrialFunction(V)
    u_test = df.TestFunction(V)
    g = df.Constant((0.0, g_int))

    # Boundary Conditions
    def bottom(x, on_boundary):
        return (on_boundary and df.near(x[1], 0.0))

    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), bottom)

    # Weak Form
    a = inner(sigma(u_tr), eps(u_test))*dx
    l = inner(g, u_test)*ds(1)

    # Solver
    u = df.Function(V)
    df.solve(a == l, u, bc)

    # Post-process
    plot = df.plot(u, mode="displacement")
    plt.colorbar(plot)
    plt.show()

    # Outputs
    output_folder = "output/linear/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disp_file = df.XDMFFile(output_folder + "U.xdmf")
    u.rename("U","displacement")
    disp_file.write(u)

if __name__=="__main__":
    main()