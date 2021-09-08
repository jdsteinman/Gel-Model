import numpy as np
import pandas as pd
import pyvtk
from numpy.linalg import eig
from scipy.linalg import polar
# from dolfin import Function as fenicsFunc

"""
Useful functions
"""
## Normalize an array of 3-component vectors
def normalize(a):

    arr = np.asarray(a)
    assert(np.ndim(arr) <= 2)

    if arr.ndim == 1:
        ss = np.sum(arr**2)**0.5
        arr = arr/ss
    elif arr.ndim == 2:
        ss = np.sum(arr**2, axis=1)**0.5
        arr = arr / ss[:, np.newaxis]

    return arr

def arr_to_tensor(arr):
    return tuple([tuple(map(tuple, A)) for A in arr])

## Coordinate conversions
def cart2pol(x, y, z):
    r = np.sqrt(x**2 + y**2)
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(r, z)
    theta = np.arctan2(y, x)
    return(rho, phi, theta)

def pol2cart(rho, phi, theta):
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return(x, y, z)

## Convert to polar, add r + k, convert back to cartesian
def add_to_radius(vertices, k):
    polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
    polar_vert[:,0] += k
    cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
    
    return cart_vert

## Convert to polar, multiply r*k, convert back to cartesian
def scale_radius(vertices, k):
    polar_vert = np.array(cart2pol(vertices[:,0], vertices[:,1], vertices[:,2] ) ).T
    polar_vert[:,0] *= k
    cart_vert = np.array(pol2cart(polar_vert[:,0], polar_vert[:,1], polar_vert[:,2] ) ).T
    
    return cart_vert

## Surface Normals
def get_surface_normals(vert, conn):
     # Check type
    vert = np.asarray(vert, dtype="float64")
    conn = np.asarray(conn, dtype="int64")

    # Face coordinates
    tris = vert[conn]

    # Face normals
    fn = np.cross(tris[:,1,:] - tris[:,0,:], tris[:,2,:] - tris[:,0,:] )

    # Normalize face normals
    fn = normalize(fn)

    # Vertex normal = sum of adjacent face normals
    n = np.zeros(vert.shape, dtype = vert.dtype)
    n[ conn[:,0] ] += fn
    n[ conn[:,1] ] += fn
    n[ conn[:,2] ] += fn

    # Normalize vertex normals
    # Mult by -1 to make them point outward??
    n = normalize(n) * -1

    return n

## Returns dot product of displacement vectors with surface outward normal
def dots(disp, norms):

    # normalize row-wise
    disp = normalize(disp)

    # dot products
    dp = np.sum(disp*norms, axis=1)

    return dp

"""
Point Data
"""
class PointData:
    def __init__(self, points, conn, u, F, mu=None):
        """
        Base class to calculate data at points and store in dataframe.

        Parameters
        ----------
        points  : array-like, float
        u       : function
        F       : function
        mu      : function
        """

        # Set inputs
        self.points = np.array(points, dtype=float)
        self.conn = np.array(conn, dtype=int)
        npoints = self.points.shape[0]

        self.u = np.asfarray(u)
        self.F = np.asfarray(F)
        self.mu = np.asfarray(mu)

        # Preallocate
        self.C = np.zeros((npoints, 3, 3))
        self.R = np.zeros((npoints, 3, 3))
        self.U = np.zeros((npoints, 3, 3))
        self.eigval = np.zeros((npoints, 3))
        self.eigvec = np.zeros((npoints, 3, 3))
        self.mu = np.zeros((npoints))
        self.df = pd.DataFrame()

    # Top level functions
    def save_to_csv(self, fname, sep=","):

        # Deformation
        self._calc_deformation()

        # Store Data in df
        self._assemble_df()

        # Save to csv
        self.df.to_csv(fname, sep=sep)

    def save_to_vtk(self, fname):

        # Deformation
        self._calc_deformation()

        # structure = pyvtk.PolyData(points = self.points, polygons=self.conn)

        point_data = pyvtk.PointData(\
            pyvtk.Vectors(self.u, name="u"),
            pyvtk.Tensors(arr_to_tensor(self.F), name="F"),
            pyvtk.Tensors(arr_to_tensor(self.C), name="C"),
            pyvtk.Tensors(arr_to_tensor(self.R), name="R"),
            pyvtk.Tensors(arr_to_tensor(self.U), name="U"),
            pyvtk.Vectors(self.eigval, name="w"),
            pyvtk.Tensors(arr_to_tensor(self.eigvec), name="v"),
            pyvtk.Scalars(self.mu, name="mu")
        )

        # print(arr_to_tensor(self.F))
        # vtk = pyvtk.VtkData(structure, point_data)
        vtk = pyvtk.VtkData(\
            pyvtk.UnstructuredGrid(self.points, 
                tetra=self.conn),
                point_data)
        vtk.tofile(fname)

    #  Low Level Functions
    def _calc_deformation(self):

        # Polar Decomposition
        R, U = [], []
        for i, F in enumerate(self.F):
            R, U = polar(F)
            self.R[i] = R
            self.U[i] = U

        # Right Cauchy-Green Tensor
        self.C = np.matmul(self.F.transpose(0,2,1), self.F)

        # Eigenvalues/eigenvectors
        self.eigval, self.eigvec = eig(self.C)

        # Order by decreasing eigenvalue
        sort_ind = np.argsort(self.eigval, axis=-1)
        sort_ind = np.flip(sort_ind, axis=-1)
        self.eigval = np.take_along_axis(self.eigval, sort_ind, -1)
        self.eigval = np.sqrt(self.eigval)   

        for i, v in enumerate(self.eigvec):
            v = v[:, sort_ind[i]]

    def _assemble_df(self):
        # Clear df
        self.df = pd.DataFrame()

        # Points
        x, y, z = np.hsplit(self.points, 3)
        r = np.sum((self.points-self.points[0])**2, axis=1)**0.5
        self.df["x"] = x.flatten()
        self.df["y"] = y.flatten()
        self.df["z"] = z.flatten()
        self.df["r"] = r.flatten()

        # Shear Modulus
        self.df["mu"] = self.mu.flatten()

        # Displacement
        ux, uy, uz = np.hsplit(self.u, 3)
        umag = (ux**2 + uy**2 + uz**2)**.5
        self.df["Ux"] = ux.flatten()
        self.df["Uy"] = uy.flatten()
        self.df["Uz"] = uz.flatten()
        self.df["Umag"] = umag.flatten()

        # Deformation Tensor        
        columns = ['F11','F12','F13','F21','F22','F23','F31','F32','F33']
        for col, dat in zip(columns, self.F.reshape((-1,9)).T):
            self.df[col] = dat

        # Rotation Tensor      
        columns = ['R11','R12','R13','R21','R22','R23','R31','R32','R33']
        for col, dat in zip(columns, self.R.reshape((-1,9)).T):
            self.df[col] = dat

        # Stretch Tensor        
        columns = ['U11','U12','U13','U21','U22','U23','U31','U32','U33']
        for col, dat in zip(columns, self.U.reshape((-1,9)).T):
            self.df[col] = dat

        # Right Cauchy-Green Tensor
        columns = ['C11','C12','C13','C21','C22','C23','C31','C32','C33']    
        for col, dat in zip(columns, self.C.reshape((-1,9)).T):
            self.df[col] = dat

        # Eigenvalues, Eigenvectors
        columns = ['w1', 'w2', 'w3']
        for col, dat in zip(columns, self.eigval.transpose()):
            self.df[col] = dat

        columns = ["v11","v12", "v13", "v21", "v22", "v23", "v31", "v32", "v33"]
        for col, dat in zip(columns, self.eigvec.reshape((-1,9), order="F").T):
            self.df[col] = dat