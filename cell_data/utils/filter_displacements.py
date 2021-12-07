import numpy as np
from scipy.spatial import distance_matrix

def neighbors(points, U_beads, N):
    points_ = []
    U_beads_ = []

    # Normalize displacements
    norms = np.apply_along_axis(np.linalg.norm, 1, U_beads)
    U_norm = U_beads/norms.reshape(-1,1)

    for i, (xyz, U) in enumerate(zip(points, U_norm)):

        # N closest neighbors
        xyz = xyz.reshape((1,-1))
        dist = distance_matrix(xyz, points)
        ind = np.argsort(dist[0])[1:N+1]  

        # Dot product with neighoring beads
        dots = [np.dot(U, U_norm[i,:]) for i in ind]

        # Mean score  
        avg_dot = np.mean(dots)

        # Threshold
        if avg_dot > 0.8:
            points_.append(xyz.ravel())
            U_beads_.append(U_beads[i,:])

    return np.array(points_), np.array(U_beads_)

def main():
    path = "bird/"
    beads_init  = np.loadtxt(path+"beads_init.txt")
    beads_final = np.loadtxt(path+"beads_final.txt")
    U_beads = beads_final - beads_init

    (points, disp) = neighbors(beads_init, U_beads, 10)

    np.savetxt(path+"beads_init_filtered.txt", points)
    np.savetxt(path+"bead_displacements_filtered.txt", disp)
    np.savetxt(path+"beads_final_filtered.txt", points+disp)

if __name__=="__main__":
    main()
