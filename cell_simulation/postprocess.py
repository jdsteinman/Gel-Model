import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones
from numpy.core.records import array
import pandas as pd
import meshio
import h5py
import os
from scipy.spatial import distance_matrix
from numpy.linalg import eig, norm, det
from scipy.linalg import polar

class postprocess():
    def __init__(self, filename, points, conn, surface_points, u_sim, u_data, F, mu):
        """
        Class to export point data to vtk.
        """

        # Set inputs
        self.points = np.array(points, dtype=float)
        self.conn = np.array(conn, dtype=int)
        self.surface_points = np.array(surface_points, dtype=float)

        self.npoints = self.points.shape[0]
        self.ncells = self.conn.shape[0]

        self.u_sim = np.array(u_sim, dtype=float)
        self.u_data = np.asfarray(u_data, dtype=float)

        self.F = F

        self.mu=mu

        # Preallocate
        self.r = np.zeros((self.npoints))               # Distance to cell surface
        self.discrepancy = np.zeros((self.npoints))     # |U_data| - |u_sim|
        self.dots = np.zeros((self.npoints))            # dot(u_data, u_sim)
        self.J = np.zeros((self.npoints))               # Jacobian     
        self.C = np.zeros((self.npoints, 3, 3))         # Right Cauchy-Green Tensor
        self.R = np.zeros((self.npoints, 3, 3))         # Rotation Tensor   
        self.U = np.zeros((self.npoints, 3, 3))         # Right Stretch Tensor
        self.eigval = np.zeros((self.npoints, 3))       # Principle stretches
        self.eigvec = np.zeros((self.npoints, 3, 3))    # Principle directions
        self.theta = np.zeros((self.npoints))           # Rotation Angle                  

        self.calculate()
        # self.filter_displacements()
        self.write_to_vtk(filename)

    def calculate(self):
        # Normal Distance
        for i, point in enumerate(self.points):
            r = point - self.surface_points
            r = np.sum(np.abs(r)**2, axis=-1)**(0.5)
            self.r[i] = np.amin(r)

        # Discrepancy
        u_data_mag = norm(self.u_data, axis=1).reshape(-1,1)
        u_sim_mag =  norm(self.u_sim, axis=1).reshape(-1,1)
        self.discrepancy =  u_data_mag - u_sim_mag

        # Dot products
        self.dots  = np.sum(self.u_sim/u_sim_mag*self.u_data/u_data_mag, axis=1)

        # Jacobain
        self.J = det(self.F)

        # Polar Decomposition
        R, U = [], []
        for i, F in enumerate(self.F):
            R, U = polar(F)
            self.R[i] = R
            self.U[i] = U

        # Rotation angle
        tr_R = np.trace(self.R, axis1=1, axis2=2)
        self.theta = np.arccos((tr_R-1)/2) * 180 / np.pi

        # Right Cauchy-Green Tensor
        self.C = np.matmul(self.F.transpose(0,2,1), self.F)

        # Eigenvalues/eigenvectors
        self.eigval, self.eigvec = eig(self.U)

        # Order by decreasing eigenvalue
        sort_ind = np.argsort(self.eigval, axis=-1)
        sort_ind = np.flip(sort_ind, axis=-1)
        self.eigval = np.take_along_axis(self.eigval, sort_ind, -1)
        self.eigval = np.sqrt(self.eigval)

        for i, v in enumerate(self.eigvec):
            v = v[:, sort_ind[i]]
            v = norm(v, axis=1)

    def filter_displacements(self):
        # Normalize displacements
        norms = np.apply_along_axis(np.linalg.norm, 1, self.u_data)
        u_norm = self.u_data/norms.reshape(-1,1)

        for i in range(self.npoints):

            # N closest neighbors
            N=10
            x = self.points[i].reshape((1,-1))
            dist = distance_matrix(x, self.points)
            ind = np.argsort(dist[0])[1:N+1]  

            # Dot Products
            dots = [np.dot(u_norm[i], u_norm[j,:]) for j in ind]
            avg_dot = np.mean(dots)

            # Magnitudes
            u_neighbor_mag = np.mean(norm(self.u_data[ind], axis=1))
            u_mag = norm(self.u_data[i])

            pct_diff = abs((u_mag-u_neighbor_mag)/u_neighbor_mag)*100
            pct_diff_avg = np.mean(pct_diff)

            if self.r[i]>10 and pct_diff_avg > 200:
                self.u_data[i,:] = np.nan
                self.u_sim[i,:] = np.nan
                self.discrepancy[i] = np.nan    

    def write_to_vtk(self, filename):
        # Write to mesh
        point_data = {
            "u_sim":self.u_sim,
            "u_data":self.u_data,
            "discrepancy":self.discrepancy,
            "dots":self.dots,
            "r":self.r,
            "J":self.J,
            "F":self.F.reshape(-1,9),
            "C":self.C.reshape(-1,9),
            "U":self.U.reshape(-1,9),
            "R":self.R.reshape(-1,9),
            "theta":self.theta,
            "eigenval":self.eigval,
            "eigenvec":self.eigvec.reshape(-1,9),
            "mu":self.mu
        }

        mesh = meshio.Mesh(self.points,
            cells=[("tetra", self.conn)],
            point_data=point_data 
        )

        mesh.write(filename)


if __name__=="__main__":
    cell = "finger"
    # directory = "./output/" + cell + "/homogeneous"
    directory = "./output/" + cell + "/degraded/iteration_2"

    # Surface points
    surface_points = np.loadtxt("../cell_data/"+cell+"/NI/meshes/cell_surface_coarse_vertices.txt")

    # Mesh points, connectivity
    f = h5py.File(os.path.join(directory, "u.h5"), 'r')
    points = f['Mesh']['mesh']['geometry'][()]
    points = np.array(points)

    conn = f['Mesh']['mesh']['topology'][()]

    # u sim
    u_sim = f['VisualisationVector']['0'][()]
    u_sim = np.array(u_sim)

    # u data
    u_data = pd.read_csv("../cell_data/"+cell+"/NI/displacements/interpolated_data_coarse.csv", index_col=False)
    u_data = u_data.loc[:,'u':'w'].to_numpy()

    # F
    f = h5py.File(os.path.join(directory, "F.h5"), 'r')
    F = f['VisualisationVector']['0'][()]
    F = np.array(F).reshape((-1,3,3))

    # mu
    if os.path.exists(os.path.join(directory, "mu.h5")):
        f = h5py.File(os.path.join(directory, "mu.h5"), 'r')
        mu = f['VisualisationVector']['0'][()]
        mu = np.array(mu) * 10**6
    else:
        mu = np.ones(u_sim.shape[0])*100

    # Postprocess
    output_filename = os.path.join(directory, "simulation_post.vtk")
    postprocess(output_filename, points, conn, surface_points, u_sim, u_data, F, mu)