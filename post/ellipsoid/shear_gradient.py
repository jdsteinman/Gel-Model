import matplotlib.pyplot as plt
import numpy as np
import math

mu = 325 * 10**12 
K = [0, 0.5, 1, 2]
x = np.linspace(-0, 250, 1000)
y = np.ones((len(K), len(x)))

fig, ax = plt.subplots(1,1)
c = ['g', 'gold', 'r', 'b']
for i, k in enumerate(K):
    for index, r in enumerate(x):
        y[i, index] = mu * (r/250) ** k

    y[i, :] = mu * y[i, :] 
    ax.plot(x, y[i,:], label= "k = " + str(k), color = c[i])

ax.set(xlabel=r"radius from cell surface ($\mu$m)", ylabel="Shear Modulus (Pa)", title="Shear Modulus Profiles\n y = $\mu$m * (r/r_max) ** k")
ax.grid()
ax.legend()

plt.show()