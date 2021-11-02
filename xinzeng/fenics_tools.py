import dolfin as df
import numpy as np
from scipy.spatial import distance_matrix

class BoundaryFunc(df.UserExpression):
    def __init__(self, mesh, face2disp_dict, scalar, **kwargs):
        self.mesh = mesh 
        self.disp = face2disp_dict
        self.scalar = scalar
        super().__init__(**kwargs)

    def value_shape(self):
        return (3,)

    def eval_cell(self, value, x, cell):
        try:
            value[0], value[1], value[2] = self.scalar*self.disp[cell.index]
        except KeyError:
            value[0], value[1], value[2] = (0, 0, 0)

def get_midpoints(nodes, faces):
    _faces = faces.astype(int)
    midpoints = np.zeros((_faces.shape[0], 3))
    for i, triangle in enumerate(_faces):
        triangle = triangle.astype(int)
        midpoints[i] = nodes[triangle].mean(0)
    return midpoints

def get_midpoint_disp(disp, faces):
    _faces = faces.astype(int)
    midpoint_disp = np.zeros((_faces.shape[0], 3))
    for i, triangle in enumerate(_faces):
        midpoint_disp[i, :] = np.mean((disp[triangle[0]],
                                        disp[triangle[1]],
                                        disp[triangle[2]]), axis=0)
    return midpoint_disp

def get_face_mapping(midpoints, mesh, mf, inner_number):
    face_map = np.zeros(midpoints.shape[0])

    for i, face in enumerate(df.faces(mesh)):
        if mf.array()[i] == inner_number:
            mesh_midpoint = face.midpoint().array().reshape((1,3)) 
            dist_mat = distance_matrix(mesh_midpoint, midpoints)
            face_map[np.argmin(dist_mat)] = face.entities(3)[0]
    return face_map.astype(int)
