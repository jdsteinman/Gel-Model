import meshio
import os
import numpy as np

def main():
    directories = ["bird"]

    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".stl"):
                input_filename = os.path.join(directory, file)
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0]

                # Read Mesh
                mesh = meshio.read(input_filename)

                # Vertices
                points = mesh.points 
                points[:,0] = points[:,0] + 72.20045715
                points[:,1] = points[:,1] + 72.90093189
                points[:,2] = points[:,2] + 47.46392168

                np.savetxt(output_filename + "_vertices.txt", points)

                # Faces
                for cell in mesh.cells:
                    if cell.type == "triangle":
                        faces = cell.data
                        np.savetxt(output_filename + "_faces.txt", faces, fmt='%d')


if __name__ == "__main__":
    main()




