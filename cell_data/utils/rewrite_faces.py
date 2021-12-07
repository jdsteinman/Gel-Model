import numpy as np
import os

directory = "../triangle"
filename = os.path.join(directory, "CytoD_faces.txt")
faces = np.loadtxt(filename, dtype=int)
if faces[0,0] >= 1:
    faces = faces - 1
else:
    pass
np.savetxt(filename, faces, fmt="%d")
