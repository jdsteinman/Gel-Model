import matplotlib.pyplot as plt
import numpy as np
import math

mu = 325 * 10**12 
k = [0.005, 0.015, 0.035, 0.055]
x = np.linspace(-250, 250, 1000)
y = np.ones((len(k), len(x)))

fig, ax = plt.subplots(1,1)
c = ['g', 'gold', 'r', 'b']
for i in range(len(k)):
    for index, p in enumerate(x):
        a = k[i]
        if p < 0:
            y[i, index] = 1 - math.exp(a*p)
        else:
            y[i, index] = 1 - math.exp(-a*p)
    y[i, :] = mu * y[i, :] 
    ax.plot(x, y[i,:], label= "k = " + str(k[i]), color = c[i])

# set units
ax.set(xlabel=r"radius from cell center ($\mu$m)", ylabel="Shear Modulus (Pa)", title="Shear Modulus Profiles\n y = $\mu$m * (1-exp(-kx))")
# ax.set_ylim(0, mu*1.1)
ax.grid()
ax.legend()

plt.show()