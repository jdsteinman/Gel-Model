import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load data
sim_path_1 = "../../gel_model/output/ellipsoid/"
sim_path_2 = "../../gel_model/output/func_grad/"
disp1 = np.loadtxt(sim_path_1 + "ellipsoid_disp.txt", delimiter=" ")
disp2 = np.loadtxt(sim_path_2 + "ellipsoid_disp.txt", delimiter=" ")

# Stats
mag1 = np.sum(np.abs(disp1)**2,axis=-1)**(1./2)
mag2 = np.sum(np.abs(disp2)**2,axis=-1)**(1./2)
corr = np.corrcoef(disp1, disp2, rowvar=False)[3:,3:]

# Plots
fig, axs = plt.subplots(1,4,figsize=(20,6))
titles = ['X disp', 'Y disp', 'Z disp', 'Magnitude']
for ax, title in zip(axs, titles):
    ax.grid()
    ax.set(adjustable='box', aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('Uniform',  fontsize=12)
    ax.set_ylabel('Gradient', fontsize=12)

axs[0].scatter(disp1[:,0], disp2[:,0], c='b', marker='o')
axs[1].scatter(disp1[:,1], disp2[:,1], c='b', marker='o')
axs[2].scatter(disp1[:,2], disp2[:,2], c='b', marker='o')
axs[3].scatter(mag1, mag2, c='r', marker='o')

fig.tight_layout()
plt.show()
# plt.savefig('./figures/scatter_1112.png', )
plt.close(fig)