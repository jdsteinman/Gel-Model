import numpy as np

vert_init = np.genfromtxt("CytoD_vertices.txt", delimiter=" ")
vert_final = np.genfromtxt("predicted_normal_vertices_from_cytod.txt", delimiter=" ")
disp = vert_final - vert_init

np.savetxt("./displacements_cytod_to_normal_uncentered_unpca2.csv", disp, delimiter = ",")