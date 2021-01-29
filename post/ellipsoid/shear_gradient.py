import matplotlib.pyplot as plt
import numpy as np
import math

mu = 325 
K = [0, 0.25, 1, 4]
x = np.linspace(-0, 200, 1000)
y = np.ones((len(K), len(x)))

fig, ax = plt.subplots(1,1)
c = ['g', 'gold', 'r', 'b']
for i, k in enumerate(K):
    for j, r in enumerate(x):
        if r <= 150:
            y[i, j] = mu * (r/150) ** k
        else:
            y[i, j] = mu

    ax.plot(x, y[i,:], label= "k = " + str(k), color = c[i])

ax.set(xlabel=r"Distance from cell surface ($\mu$m)", ylabel="Shear Modulus (Pa)", title="Shear Modulus Profiles")
ax.grid()
ax.legend()

plt.show()