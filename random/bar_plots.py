import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Inputs
data = pd.read_csv("../gel_model/output/bar/bar.txt", index_col=0)

# Plots
sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(data.x, data.Ux, c='b')
ax[0].set_title('Bar Simulation with Left End Displaced')
ax[0].set_ylabel('Ux')

ax[1].plot(data.x, data.F11, c='g')
ax[1].set_ylabel('F11')
ax[1].set_xlabel('X Position')

# ax[2].plot(data.x, data.mu*1e-12, c='r')
# ax[2].set_xlabel('X Position')
# ax[2].set_ylabel('mu')
# ax[2].set_ylim(0, max(data.mu)*1.2e-12)

plt.tight_layout()
plt.show()