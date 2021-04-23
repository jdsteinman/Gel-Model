import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Inputs
data = pd.read_csv("../gel_model/output/bctest/CL1/Zdata.csv", index_col=0)

# Plots
sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(data.r, data.Uz, c='b')
ax[0].set_title('0.2 Micron Compression in Z')
ax[0].set_ylabel('Uz')

ax[1].plot(data.r, data.F33, c='g')
ax[1].set_ylabel('F33')
ax[1].set_xlabel('Distance from surface')

# ax[2].plot(data.r, data.mu, c='r')
# ax[2].set_xlabel('Distance from surface')
# ax[2].set_ylabel('mu')
# ax[2].set_ylim(0, max(data.mu)*1.2)

plt.tight_layout()
plt.show()