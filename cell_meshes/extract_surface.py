import meshio
import os
import numpy as np
def main():
    """
    Use to extract nodes from 3D mesh from a certain physical volume
    """
  
    phy_num = 202
  
    directory = "new_cell"
    file = "hole.msh"
    input_filename = os.path.join(directory, file)
    
    print("Converting %s" % input_filename)
    output_filename = os.path.splitext(input_filename)[0]
    
    # Read Mesh
    mesh = meshio.read(input_filename)
    
    # Vertices
    points = mesh.points 
    # Elements
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif  cell.type == "tetra":
            tetra_cells = cell.data
    # Get physical labels
    for key in mesh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            triangle_data = mesh.cell_data_dict["gmsh:physical"][key]
        elif key == "tetra":
            tetra_data = mesh.cell_data_dict["gmsh:physical"][key]
    
    # Select points
    select_cells = triangle_cells[triangle_data==phy_num]
    vertex_ids = np.unique(select_cells)
    select_points = points[vertex_ids]
            
    np.savetxt(output_filename + "_extracted_vertices.txt", select_points)

if __name__ == "__main__":
    main()