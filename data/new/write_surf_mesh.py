import meshio
import numpy as np

points = np.genfromtxt("CytoD_vertices.txt", delimiter=" ")
faces = np.genfromtxt("CytoD_faces.txt", delimiter=" ") 
faces = faces.astype("int")
faces = faces - 1

cells = [("triangle", faces)]

meshio.write_points_cells(
    "cytod_uncentered_unpca_surface.xml",
    points,
    cells,   
)
