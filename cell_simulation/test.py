import nodal_tools as nt
import numpy as np

u_sim  = np.loadtxt("output/500/bead_displacements_sim.txt", skiprows=1)[:,4:]
u_data = np.loadtxt("output/500/bead_displacements_data.txt", skiprows=1)[:,4:]

# Normalize displacements
row_mag = np.sum(u_sim*u_sim, axis=1)**0.5
u_sim_norm = u_sim/row_mag[:,np.newaxis]
print(u_sim_norm)

row_mag = np.sum(u_data*u_data, axis=1)**0.5
u_data_norm = u_data/row_mag[:,np.newaxis]
print(u_data_norm)

# Dot products
dots = np.sum(u_sim_norm*u_data_norm, axis=1)

print(np.sort(dots))