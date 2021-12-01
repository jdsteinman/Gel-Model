import meshio
import os
import numpy as np
def main():
    """
    Use to extract nodes from 3D mesh from a certain physical volume
    """
  
    phy_num = 202
  
    directory = "new_cell"
    file = "cell_surface_1000.stl"
    input_filename = os.path.join(directory, file)
    
    print("Converting %s" % input_filename)
    output_filename = os.path.splitext(input_filename)[0]
    
    # Read Mesh
    mesh = meshio.read(input_filename)
    
    # Vertices
    points = mesh.points 
    np.savetxt(output_filename + "_vertices.txt", points)

    # Elements
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
            np.savetxt(output_filename + "_faces.txt", triangle_cells, fmt="%d")
        elif  cell.type == "tetra":
            tetra_cells = cell.data
            
    

if __name__ == "__main__":
    main()