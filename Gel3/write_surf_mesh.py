import meshio
import numpy as np

path = "../data/new/"
points = np.genfromtxt(path+"CytoD_vertices.txt", delimiter=" ")
faces = np.genfromtxt(path+"CytoD_faces.txt", delimiter=" ") 
faces = faces.astype("int")
faces = faces - 1

cells = [("triangle", faces)]

meshio.write_points_cells(
    "cytod_uncentered_unpca_surface.xdmf",
    points,
    cells,   
)
