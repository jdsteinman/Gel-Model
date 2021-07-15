import meshio
import os

def main():
    directories = ['./meshes']
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".msh"):
                input_filename = os.path.join(directory, file)
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0]

                mesh = meshio.read(input_filename)
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

                # Separate meshes
                try:
                    meshio.write(output_filename + ".xdmf", meshio.Mesh(points=mesh.points, cells={"tetra": tetra_cells}))
                    meshio.write(output_filename + "_domains.xdmf", meshio.Mesh(points=mesh.points, 
                                cells={"tetra": tetra_cells},
                                cell_data={"domains":[tetra_data]}))
                except:
                    print("Unable to convert", output_filename)

                try:
                    triangle_mesh = meshio.Mesh(points=mesh.points,
                                        cells=[("triangle", triangle_cells)],
                                        cell_data={"boundaries":[triangle_data]})
                    meshio.write(output_filename +  "_boundaries.xdmf", triangle_mesh)
                except:
                    print("Unable to convert", output_filename)


if __name__ == "__main__":
    main()
