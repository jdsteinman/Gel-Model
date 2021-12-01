import meshio
import os
import numpy as np

def main():
    directories = ['./meshes']
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".msh"):
                input_filename = os.path.join(directory, file)
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0]

                mesh = meshio.read(input_filename)
                points = mesh.points

                for cell in mesh.cells:
                    if cell.type == "triangle":
                        triangle_cells = cell.data

                # Get physical labels
                for key in mesh.cell_data_dict["gmsh:physical"].keys():
                    if key == "triangle":
                        triangle_data = mesh.cell_data_dict["gmsh:physical"][key]

                # Get surface nodes
                surface_cells = triangle_cells[triangle_data==202]
                surface_nodes = np.unique(surface_cells)
                surface_vert  = points[surface_nodes]

                # Separate meshes
                try:
                    print("\tWriting vertices")
                    np.savetxt(output_filename + "_surface.txt", surface_vert)
                    surface_mesh = meshio.Mesh(points=points,
                                        cells=[("triangle", surface_cells)])
                    meshio.write(output_filename + "_surface.xdmf", surface_mesh)
                except Exception as e: 
                    print("\t", e)
                    quit()

if __name__ == "__main__":
    main()
