import meshio
import numpy as np

inpath = "../cell_data/brid/"
outpath = "bird/"

points = np.genfromtxt(inpath + "CytoD_vertices.txt", delimiter=" ")
faces = np.genfromtxt(inpath + "CytoD_faces.txt", delimiter=" ") 
faces = faces.astype("int")
faces = faces - 1

cells = [("triangle", faces)]

meshio.write_points_cells(
    outpath+"cytod_uncentered_unpca_surface.xml",
    points,
    cells,   
)
