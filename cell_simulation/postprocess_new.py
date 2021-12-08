import numpy as np
import pandas as pd
import meshio
import h5py
import os
from scipy.spatial import distance_matrix
from numpy.linalg import norm

class postprocess():
    def __init__(self, filename, points, conn, surface_points, u_sim, u_data):
        """
        Class to export point data to vtk.
        Usage:
            point_data = PointData(points, conn, u, F, mu)
            point_data.save_to_vtk("out.vtk")

        Parameters
        ----------
        points  : array-like, float
        conn    : array-like, int
        u       : array-like, float
        F       : array-like, float
        mu      : array-like, float
        """

        # Set inputs
        self.points = np.array(points, dtype=float)
        self.conn = np.array(conn, dtype=int)
        self.surface_points = np.array(surface_points, dtype=float)

        self.npoints = self.points.shape[0]
        self.ncells = self.conn.shape[0]

        self.u_sim = np.array(u_sim, dtype=float)
        self.u_data = np.asfarray(u_data, dtype=float)

        # Initialize
        self.r = np.zeros((self.npoints))
        self.discrepancy = np.zeros((self.npoints)) 

        self.calculate()
        # self.filter_displacements()
        self.write_to_vtk(filename)

    def calculate(self):
        # Normal Distance
        for i, point in enumerate(self.points):
            r = point - self.surface_points
            r = np.sum(np.abs(r)**2, axis=-1)**(0.5)
            self.r[i] = np.amin(r)

        self.discrepancy = norm(self.u_data, axis=1) - norm(self.u_sim, axis=1)

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

                print(u_neighbor_mag)
                print(u_mag)

    def write_to_vtk(self, filename):
        # Write to mesh
        point_data = {
            "u_sim":self.u_sim,
            "u_data":self.u_data,
            "discrepancy":self.discrepancy,
            "r":self.r
        }

        mesh = meshio.Mesh(self.points,
            cells=[("tetra", self.conn)],
            point_data=point_data 
        )
        print(mesh)

        mesh.write(filename)


if __name__=="__main__":
    cell = "claw"
    directory = "./output/" + cell + "/homogeneous"

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

    # Postprocess
    output_filename = os.path.join(directory, "displacement_compare.vtk")
    postprocess(output_filename, points, conn, surface_points, u_sim, u_data)