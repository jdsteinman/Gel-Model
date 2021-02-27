import numpy as np
import  matplotlib.pyplot as plt

## Data
path = "../../gel_model/output/"
out_path = "/home/john/Pictures/2-5-21/"
axis = "z"

u_unif = np.loadtxt(path + "uniform/disp_" + axis + "_uniform.txt")
mu_unif = np.loadtxt(path + "uniform/mu_" + axis + "_uniform.txt")
u_step = np.loadtxt(path + "step/disp_" + axis + "_step.txt")[:, 3:]
mu_step = np.loadtxt(path + "step/mu_" + axis + "_step.txt")[:,-1]

u_unif = np.sum(u_unif**2, axis=1)**0.5
u_step = np.sum(u_step**2, axis=1)**0.5

u_rat = u_step / u_unif
mu_rat = mu_step / mu_unif

npoints = 1000
points = np.linspace(20, 100, npoints) - 20

## Plots
fig, ax = plt.subplots(1,1)
ax.plot(points, u_rat, label="u ratio")
ax.plot(points, mu_rat, label="mu ratio")

ax.set_title("Ratios along " + axis + "-axis")
ax.set_xlabel(r'Distance from surface ($\mu$m)')
ax.set_ylabel("Step/Uniform")
ax.set_ylim(0, 1.25)
ax.grid()
ax.legend()

fig2, ax = plt.subplots(1,1)
ax.plot(points, u_step, label="step", c='r')
ax.plot(points, u_unif, label="uniform", c='b')

ax.set_title("Displacement Magnitude along " + axis + "-axis")
ax.set_xlabel(r'Distance from surface ($\mu$m)')
ax.set_ylabel("Displacement ($\mu$m)")
ax.grid()
ax.legend()

fig3, ax = plt.subplots(1,1)
ax.plot(points, mu_step, label="step", c='r')
ax.plot(points, mu_unif, label="uniform", c='b')

ax.set_title("Shear modulus along " + axis + "-axis")
ax.set_xlabel(r'Distance from surface ($\mu$m)')
ax.set_ylabel("Pascals (Pa)")
ax.set_ylim(0, 350)
ax.grid()
ax.legend()

fig.savefig(out_path + "ratios_" + axis + ".png")
fig2.savefig(out_path + "disp_" + axis + ".png")
fig3.savefig(out_path + "shear_" + axis + ".png")
plt.show()