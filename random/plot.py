import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Inputs
data = pd.read_csv("../gel_model/output/test/data.csv", index_col=0)
axis = "z"

# Plot setup
sns.set_theme(style="whitegrid")

# Displacement
# fig, ax = plt.subplots(1,1)
sns.lineplot(x='r', y='U_mag', ci=None, data=data)

# sns.lineplot(x='r', y='C11', ci=None, data=data, ax=ax[0])
# sns.lineplot(x='r', y='mu', ci=None, data=data, ax=ax[1])

plt.show()