import numpy as np

data = np.loadtxt("/home/john/research/Gel-Model/cell_data/bird/surface_displacements_1000_vertices.csv",
            dtype = float, skiprows=1, delimiter=",")

data = data[:,3:]
np.savetxt("/home/john/research/Gel-Model/cell_data/bird/surface_displacements_1000.csv", data)