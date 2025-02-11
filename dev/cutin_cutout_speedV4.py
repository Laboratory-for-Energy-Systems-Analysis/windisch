import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.f2py.crackfortran import skipfuncs

# Loading the data
file_path = "Power_curves.csv"
data = pd.read_csv(
    file_path,
    header=1,
)
data = data.iloc[:, :-2]

# for each row, find the maximum power output
# index of the first non-zero value
# and the column of the last non-zero value
# and fetch the corresponding wind speed value

max_power = data.iloc[:, 4:].max(axis=1).values
cut_in = data.iloc[:, 4:].apply(lambda row: row[row != 0].index[0], axis=1).values
cut_out = data.iloc[:, 4:].apply(lambda row: row[row != 0].index[-1], axis=1).values

# create a dataframe with the power of each turbine
# and their cut-in and cut-off speed

df = pd.DataFrame(data={"Power": max_power, "Cut-in": cut_in, "Cut-out": cut_out})

# All columns as numeric
df = df.apply(pd.to_numeric, errors="coerce")

# create a boxplot to show the distribution of cut-in and cut-out speeds
# with respect of the power output (bins of 500 kW)

bins = np.arange(0, 10000, 500)
labels = [f"{i}-{i+500}" for i in np.arange(0, 9500, 500)]

df["Power_bin"] = pd.cut(df["Power"], bins=bins, labels=labels)

fig, ax = plt.subplots(2, 1, figsize=(10, 12))
df.boxplot(column="Cut-in", by="Power_bin", ax=ax[0])
df.boxplot(column="Cut-out", by="Power_bin", ax=ax[1])

# add a label above each bin with the number of turbines
for i, a in enumerate(ax):
    for j, b in enumerate(a.get_xticklabels()):
        count = df[df["Power_bin"] == b.get_text()]["Power"].count()
        a.text(j + 1, 50, f"n={count}", ha="center", va="top")

# xticks labels should be in Mw, not kW
ax[0].set_xticklabels([f"{i}-{i+0.5}" for i in np.arange(0, 9.5, 0.5)])
ax[1].set_xticklabels([f"{i}-{i+0.5}" for i in np.arange(0, 9.5, 0.5)])

plt.show()

# create a dataframe giving the average cut-in,
# cut-off speeds per power bin
# with min-max values

df_grouped = df.groupby("Power_bin").agg(
    Cut_in_mean=("Cut-in", "mean"),
    Cut_in_min=("Cut-in", "min"),
    Cut_in_max=("Cut-in", "max"),
    Cut_out_mean=("Cut-out", "mean"),
    Cut_out_min=("Cut-out", "min"),
    Cut_out_max=("Cut-out", "max"),
)
df_grouped.to_excel("average_cutin_cutoff_speeds.xlsx")
