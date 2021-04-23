import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Inputs
data = pd.read_csv("../../ellipsoid/output/Zdata.csv", index_col=0)

params = ["Uz", "F33"]
title = "along Z-axis"
xlab = "Distance from cell surface"

output_folder = "/home/john/Pictures/2-12-21/"

# Plot setup
sns.set_theme(style="darkgrid")

# Displacement
fig, ax = [], []
for i, y in enumerate(params):
    f, a = plt.subplots(1,1)
    sns.lineplot(x='r', y=y, ci=None, data=data, ax = a)

    # if y == "mu": a.set_ylim(0, 350)
    # a.vlines(x=10, ymin=a.get_ylim()[0], ymax=a.get_ylim()[1], linestyles='dashed')

    a.set_title(y + " " + title)
    a.set_ylabel(y)
    a.set_xlabel(xlab)

    # f.savefig(output_folder + y + ".png")

    fig.append(f)
    ax.append(a)


plt.tight_layout()
plt.show()