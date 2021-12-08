import numpy as np
from scipy.spatial import distance_matrix

def neighbors(points, disp, neighbors=10):
    points_ = []
    U_beads_ = []

    for i, (xyz, u_neighbor) in enumerate(zip(points, disp)):

        # closest neighbors
        xyz = xyz.reshape((1,-1))
        dist = distance_matrix(xyz, points)
        ind = np.argsort(dist[0])[1:neighbors+1]  

        # Normalize displacements
        norms = np.apply_along_axis(np.linalg.norm, 1, disp)
        u_norm = disp/norms.reshape(-1,1)

        # Dot product with neighoring beads
        dots = [np.dot(u_norm[i,:], u_neighbor) for i in ind]

        # Mean score  
        avg_dot = np.mean(dots)

        # Threshold
        if avg_dot > 0.8:
            points_.append(xyz.ravel())
            U_beads_.append(U_beads[i,:])

    return np.array(points_), np.array(U_beads_)

def main():
    cell = "bird"

    points = np.loadtxt("../"+cell+"/NI/meshes/hole_coarse_vertices.txt")
    disp = np.loadtxt("../"+cell+"/NI/displacements/surface_displacements_coarse.txt")


    (points, disp) = filter_displacements(points, disp, neighbors=10)

    np.savetxt(path+"beads_init_filtered.txt", points)
    np.savetxt(path+"bead_displacements_filtered.txt", disp)
    np.savetxt(path+"beads_final_filtered.txt", points+disp)

if __name__=="__main__":
    main()
