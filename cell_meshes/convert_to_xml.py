import os
from dolfin_utils.meshconvert import meshconvert

def main():
    directories = ['./bird']
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".msh"):
                input_filename = os.path.join(directory, file)
                print("Converting %s" % input_filename)
                output_filename = os.path.splitext(input_filename)[0] + '.xml'
                print(output_filename)
                meshconvert.convert2xml(input_filename,
                                        output_filename, iformat="gmsh")

if __name__ == "__main__":
    main()
