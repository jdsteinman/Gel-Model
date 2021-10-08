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

class shear_modulus(df.UserExpression):
    def __init__ (self, vert, mu, rmax, p=1, method="power", **kwargs):
        super().__init__(**kwargs)
        self._vert = np.asarray(vert, dtype="float64")  # surface vertices
        self.mu = mu        # Shear modulus 
        self.p = p          # Shape parameter
        self.rmax = rmax    # Max distance from surface
        self.method=method  # power or step

    def eval(self, value, x):
        px = np.array([x[0], x[1], x[2]], dtype="float64")

        # Distance to surface
        r = px - self._vert
        r = np.sum(np.abs(r)**2, axis=-1)**(1./2)
        r = np.amin(r)

        if r < self.rmax:
            if self.method=="power":
                value[0] = self.mu*(r/self.rmax)**self.p + self.mu*.01  # Power Model
            elif self.method=="step":
                value[0] = self.mu*0.5 + self.k*.01   # Step function
            else:
                print("Unknown method")
                quit()
        else:
            value[0] =  self.mu*1.01 

    def value_shape(self):
        return ()

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
