import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dolfin as df

# Read Inputs
path = "../sphere_in_cube2/output/inclusion/"
mesh = df.Mesh()

with df.XDMFFile(path+"out.xdmf") as infile:
    infile.read(mesh)
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    T = df.TensorFunctionSpace(mesh, "CG", 1, shape=(3,3))

    U = df.Function(V)
    F = df.Function(T)

    infile.read_checkpoint(U, "u")
    infile.read_checkpoint(F, "F")

# Get data
npoints = 1000
line = np.hstack((np.zeros((npoints,1)), np.zeros((npoints,1)), np.linspace(0, 25, npoints).reshape(-1,1)))

Uarr = np.array([U(point) for point in line])
Farr = np.array([F(point) for point in line])

dudz = np.gradient(Uarr[:,2], line[:,2])
F_approx = dudz + 1

# Plots
sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(line[:,2], Uarr[:,2], c='b')
ax[0].set_title('0.2 Micron Compression in Z')
ax[0].set_ylabel('Uz')

ax[1].plot(line[:,2], Farr[:,8], c='g', label="Projected")
ax[1].plot(line[:,2], F_approx, c='r', label="Numerical")
ax[1].set_ylabel('F33')
ax[1].set_xlabel('Distance from surface')
ax[1].legend()

plt.tight_layout()
plt.show()
