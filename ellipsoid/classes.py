import meshio
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from numpy.linalg import eig
from scipy.linalg import polar, inv
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle
from dolfin import Function as fenicsFunc
from sim_tools import normalize, pol2cart, cart2pol, scale_radius, get_surface_normals, dots

"""
Classes for Isosurface and Line outputs
"""

class PointData:
    def __init__(self, points, u, F, mu):
        """
        Class to calculate data at points and store in dataframe.

        Parameters
        ----------

        points  : array-like, float
        u       : function
        F       : function
        mu      : function
        """

        # Set inputs
        self._points = np.asarray(points)
        npoints = self._points.shape[0]

        self._uFunc = u
        self._defFunc = F
        self._muFunc = mu

        # Preallocate
        self._u = np.zeros((npoints, 3))
        self._F = np.zeros((npoints, 3, 3))
        self._C = np.zeros((npoints, 3, 3))
        self._R = np.zeros((npoints, 3, 3))
        self._U = np.zeros((npoints, 3, 3))
        self._w = np.zeros((npoints, 3))
        self._v = np.zeros((npoints, 3, 3))
        self._mu = np.zeros((npoints))

        self._df = pd.DataFrame()

    # Top level functions (for user)
    def save_to_csv(self, fname, sep=","):
        
        # Get Arrays
        self._calc_disp()
        self._calc_deformation()
        self._calc_mu()

        # Store Data in df
        self._to_df()

        # Save to csv
        self._df.to_csv(fname, sep=sep)

    def save_to_vtk(self, fname, sep=","):
        # Not sure if we need this
        pass

    #  Low Level Functions
    def _calc_disp(self):
        self._u = np.array([self._uFunc(p) for p in self._points])

    def _calc_deformation(self):

        # Deformation Gradient
        self._F = np.array([self._defFunc(p) for p in self._points])
        self._F = self._F.reshape((-1,3,3))

        # Polar Decomposition
        R, U = [], []
        for i, F in enumerate(self._F):
            R, U = polar(F)
            self._R[i] = R
            self._U[i] = U

        # Right Cauchy-Green Tensor
        self._C = np.matmul(self._F.transpose(0,2,1), self._F)

        # Eigenvalues/eigenvectors
        self._w, self._v = eig(self._C)

        # Order by decreasing eigenvalue
        sort_ind = np.argsort(self._w, axis=-1)
        sort_ind = np.flip(sort_ind, axis=-1)
        self._w = np.take_along_axis(self._w, sort_ind,-1)
        self._w = np.sqrt(self._w)   

        for i, v in enumerate(self._v):
            v = v[:, sort_ind[i]]

    def _calc_mu(self):
        self._mu = np.array([self._muFunc(p) for p in self._points])

    def _to_df(self):
        # Clear df
        self._df = pd.DataFrame()

        # Points
        x, y, z = np.hsplit(self._points, 3)
        r = np.sum((self._points-self._points[0])**2, axis=1)**0.5
        self._df["x"] = x.flatten()
        self._df["y"] = y.flatten()
        self._df["z"] = z.flatten()
        self._df["r"] = r.flatten()

        # Shear Modulus
        self._df["mu"] = self._mu.flatten()

        # Displacement
        ux, uy, uz = np.hsplit(self._u, 3)
        umag = (ux**2 + uy**2 + uz**2)**.5
        self._df["Ux"] = ux.flatten()
        self._df["Uy"] = uy.flatten()
        self._df["Uz"] = uz.flatten()
        self._df["Umag"] = umag.flatten()

        # Deformation Tensor        
        columns = ['F11','F12','F13','F21','F22','F23','F31','F32','F33']
        for col, dat in zip(columns, self._F.reshape((-1,9)).T):
            self._df[col] = dat

        # Rotation Tensor      
        columns = ['R11','R12','R13','R21','R22','R23','R31','R32','R33']
        for col, dat in zip(columns, self._R.reshape((-1,9)).T):
            self._df[col] = dat

        # Stretch Tensor        
        columns = ['U11','U12','U13','U21','U22','U23','U31','U32','U33']
        for col, dat in zip(columns, self._U.reshape((-1,9)).T):
            self._df[col] = dat

        # Right Cauchy-Green Tensor
        columns = ['C11','C12','C13','C21','C22','C23','C31','C32','C33']    
        for col, dat in zip(columns, self._C.reshape((-1,9)).T):
            self._df[col] = dat

        # Eigenvalues, Eigenvectors
        columns = ['w1', 'w2', 'w3']
        for col, dat in zip(columns, self._w.transpose()):
            self._df[col] = dat

        columns = ["v11","v12", "v13", "v21", "v22", "v23", "v31", "v32", "v33"]
        for col, dat in zip(columns, self._v.reshape((-1,9), order="F").T):
            self._df[col] = dat


class Isosurfaces(PointData):

    def __init__(self, sets, points, conn, u, F, mu):
        """
        Creates Isosurface outputs. Calculates and formats data one set at a time
        using PointData's routines.

        Parameters
        ----------

        sets    : list of scalars
        points  : array-like, float
        conn    : array-like, int
        u       : function
        F       : function
        mu      : function
        """

        # Set inputs
        self._sets = sets
        self._conn = np.asarray(conn, dtype=np.int64)
        self._surf_points = points

        # Preallocate
        self._data_dict = {}
        self._set_dict = {}
        self._Uavg = []

        # Init super
        super().__init__(points, u, F, mu)

    # Top Level function
    def save_sets(self, output_folder="./"):

        # Loop through isosurfaces
        for scalar in self._sets:

            # reset df
            self._df = pd.DataFrame()

            # Scale points
            self._points = scale_radius(self._surf_points, scalar)

            # Caclulate data
            self._calc_disp()
            self._calc_deformation()
            self._calc_mu()
            self._to_df()
            
            # Normals
            n = get_surface_normals(self._points, self._conn)
            nx, ny, nz = np.hsplit(n, 3)
            self._df["nx"] = nx.flatten()
            self._df["ny"] = ny.flatten()
            self._df["nz"] = nz.flatten()

            # Normal Stretches
            n = n.reshape((-1,3,1))
            stretches = np.matmul(n.transpose(0, 2, 1), np.matmul(self._C, n)) ** 0.5
            stretches = stretches.flatten()
            self._df["nstretch"] = stretches

            # Format Data
            self._df_to_dict()
            self._data_dict[str(scalar)] = self._df

            # Write to VTK
            x = np.ascontiguousarray(self._points[:,0], dtype="float64")
            y = np.ascontiguousarray(self._points[:,1], dtype="float64")
            z = np.ascontiguousarray(self._points[:,2], dtype="float64")

            # cell types
            ncells = np.size(self._conn, 0)
            ctype = np.zeros(ncells)
            ctype[:] = VtkTriangle.tid

            # flatten conenctivity
            conn = self._conn.flatten().astype("int64")

            # offset begins with 1
            offset = 3 * (np.arange(ncells, dtype='int64') + 1)

            # save to vtk
            unstructuredGridToVTK(output_folder + "set_" + str(scalar), x, y, z, connectivity=conn, offsets=offset, cell_types = ctype, pointData=self._set_dict)

    def _df_to_dict(self):
        skips = ['x', 'y', 'z', 'r']
        for column in self._df:
            if column in skips: continue
            dat = np.ascontiguousarray(self._df[column],  dtype=np.float64)
            self._set_dict[column] = dat

class LineData(PointData):
    def __init__(self, u, F, mu, point, direction, bound, step=0.1, max_points=1000):

        # Set inputs
        self._point = np.asarray(point, dtype=np.float64)
        self._direction = np.asarray(direction, dtype=np.float64)
        self._bound = bound
        self._step = step
        self._max_points = max_points
        self._set_points()

        # Super Init
        super().__init__(self._points, u, F, mu)

        # Preallocate
        self._stretches = np.zeros(self._points.shape[0])

    def _set_points(self):

        self._points = np.asarray(self._point).reshape((1,3))

        for i in range(self._max_points):
            nextpoint = self._points[-1] + self._direction * self._step
            cond_x = abs(nextpoint[0])>=self._bound
            cond_y = abs(nextpoint[1])>=self._bound
            cond_z = abs(nextpoint[2])>=self._bound

            if cond_x or cond_y or cond_z:
                break

            self._points = np.vstack((self._points, nextpoint)) 
        
    def _calc_deformation(self):
        
        # Deformations
        super()._calc_deformation()

        # Stretches
        npoints = self._points.shape[0]
        v = np.broadcast_to(self._direction, (npoints, 3))
        v = np.ascontiguousarray(v)
        v = v.reshape((-1, 3, 1))

        self._stretches = np.matmul(v.transpose(0, 2, 1), np.matmul(self._C, v)) ** 0.5
        self._stretches = self._stretches.flatten()

    def _to_df(self):
        super()._to_df()

        self._df["stretch"] = self._stretches


