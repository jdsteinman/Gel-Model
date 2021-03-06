import matplotlib.pyplot as plt
import numpy as np
import math

mu = 325 
rmax = 10
K = [0, -1]
x = np.linspace(0, 55, 100)
y = np.ones((len(K), len(x)))
labels = ["Uniform", "Step"]

fig, ax = plt.subplots(1,1)
c = ['r', 'b']
for i, k in enumerate(K):
    for j, r in enumerate(x):
        if r <= rmax:
            if k<0:
                y[i, j] = mu * 0.5
            else:
                y[i, j] = mu * (r/150) ** k
        else:
            y[i, j] = mu

    ax.plot(x, y[i,:], label=labels[i], color = c[i])

ax.set(xlabel=r"Distance from cell surface ($\mu$m)", ylabel="Shear Modulus (Pa)", title="Shear Modulus Profiles")
ax.set_ylim(0,mu+25)
ax.grid()
ax.legend(loc='right')

plt.show()