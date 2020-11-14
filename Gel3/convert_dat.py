import pandas as pd
import numpy as np
import csv

node_file = "CytoD_vertices.txt"
face_file = "CytoD_faces.txt"
out_file  = "cytod_mesh.msh"

out = open(out_file, 'w+')
out.write("$MeshFormat \n2.2 0 8 \n$EndMeshFormat \n$Nodes\n")

with open(node_file) as nf:
    # Get number of nodes
    num_lines = sum(1 for line in nf)
    out.write(str(num_lines) + "\n")

    nf.seek(0,0)  # set pointer to beginning
    idx = 1
    for line in nf:
        ostr = str(idx) + " " + str(line)
        out.write(ostr)
        idx += 1

out.write("$EndNodes") # \n$Elements\n")
"""
with open(face_file) as ff:
    # Get number of nodes
    num_lines = sum(1 for line in ff)
    out.write(str(num_lines) + "\n")

    ff.seek(0,0)  # set pointer to beginning
    idx = 1
    for line in ff:
        ostr = str(idx) + " " + str(line)
        out.write(ostr)
        idx += 1    

out.write("$EndElements")
"""
out.close()



