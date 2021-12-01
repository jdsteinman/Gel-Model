import meshio
import os
import numpy as np

def main():
    directories = ['./bird']
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".msh"):
                input_filename = os.path.join(directory, file)
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0]

                mesh = meshio.read(input_filename)
                points = mesh.points
                points[:,0] = points[:,0] + 72.20045715
                points[:,1] = points[:,1] + 72.90093189
                points[:,2] = points[:,2] + 47.46392168

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
                        print(tetra_data)
                        print(tetra_cells[tetra_data==301])
                        print(points[np.unique(tetra_cells[tetra_data==301])])
                # Separate meshes
                try:
                    print("\tWriting mesh")
                    meshio.write(output_filename + ".xdmf", meshio.Mesh(points=points, cells={"tetra": tetra_cells}))
                    print("\tWriting domains")
                    meshio.write(output_filename + "_domains.xdmf", meshio.Mesh(points=points, 
                                cells={"tetra": tetra_cells},
                                cell_data={"domains":[tetra_data]}))
                except Exception as e: 
                    print("\t", e)
                    quit()

                try:
                    print("\tWriting boundaries")
                    triangle_mesh = meshio.Mesh(points=points,
                                        cells=[("triangle", triangle_cells)],
                                        cell_data={"boundaries":[triangle_data]})
                    meshio.write(output_filename +  "_boundaries.xdmf", triangle_mesh)
                except Exception as e: 
                    print("\t",e)
                    quit()

if __name__ == "__main__":
    main()
