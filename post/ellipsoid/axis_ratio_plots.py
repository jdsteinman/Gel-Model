import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Inputs
data1 = pd.read_csv("../../gel_model/output/uniform/data_z.csv", index_col=0)
data2 = pd.read_csv("../../gel_model/output/step/data_z.csv", index_col=0)

output_folder = "/home/john/Pictures/2-12-21/"

params = ["F33", "C33", ]
title = "Ratio along Z-axis"
xlab = "Distance from cell surface"
leg = ['uniform', 'step']

# Plot setup
sns.set_theme(style="darkgrid")

# Displacement
fig, ax = [], []

for i, y in enumerate(params):
    f, a = plt.subplots(1,1)

    ratio = data1.loc[:, y] / data2.loc[:, y]
    sns.lineplot(x=data1.r, y=ratio, ci=None, ax = a, label=leg[0] + " / " + leg[1])

    if y == "mu": a.set_ylim(0, 350)

    a.vlines(x=10, ymin=a.get_ylim()[0], ymax=a.get_ylim()[1], linestyles='dashed')

    a.set_title(y + " " + title)
    a.set_ylabel(y)
    a.set_xlabel(xlab)
    a.legend(loc='best')

    # f.savefig(output_folder + y + ".png")

    fig.append(f)
    ax.append(a)

plt.tight_layout()
plt.show()