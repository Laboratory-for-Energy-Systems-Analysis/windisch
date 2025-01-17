import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Code/windisch_folder/dev/Power_Curves.csv"
data = pd.read_csv(file_path)

# Creating a DataFrame of size (881x71) containing the power outputs and corresponding wind speeds - the first row is the wind speed
dataframe = data.iloc[0:881, 4:75]

# Adding a new column containing the maximum power output of each wind turbine (starting from the second row)
dataframe['Max Power Output'] = dataframe.iloc[1:].max(axis=1)

# Adding new columns for "cut-in wind speed" and "cut-out wind speed"
dataframe["cut-in wind speed"] = None
dataframe["cut-out wind speed"] = None

# Iterating through each row starting from row 2 (index 1)
for index, row in dataframe.iloc[1:].iterrows():
    # Finding the first non-zero column and set the corresponding wind speed (cut-in wind speed)
    cut_in_col = (row.iloc[:71] != 0).idxmax()
    if cut_in_col in dataframe.columns:
        dataframe.at[index, "cut-in wind speed"] = dataframe.at[0, cut_in_col]

    # Finding the last non-zero column and set the corresponding wind speed (cut-out wind speed)
    cut_out_col = (row.iloc[:71] != 0)[::-1].idxmax()
    if cut_out_col in dataframe.columns:
        dataframe.at[index, "cut-out wind speed"] = dataframe.at[0, cut_out_col]

# Define bins and labels for both analyses
bin_edges = [35, 500, 1500, 3000, 6000, 10000]
bin_labels = ["35-500", "501-1500", "1501-3000", "3001-6000", "6001-10000"]

# Assign bins for both analyses
dataframe["Power Output Bin"] = pd.cut(dataframe["Max Power Output"], bins=bin_edges, labels=bin_labels, include_lowest=True)
dataframe["Cut-out Power Output Bin"] = pd.cut(dataframe["Max Power Output"], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Ensure "cut-in wind speed" and "cut-out wind speed" are numeric
dataframe["cut-in wind speed"] = pd.to_numeric(dataframe["cut-in wind speed"], errors="coerce")
dataframe["cut-out wind speed"] = pd.to_numeric(dataframe["cut-out wind speed"], errors="coerce")

# Create both plots in one go
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot for cut-in wind speed
dataframe.boxplot(column="cut-in wind speed", by="Power Output Bin", grid=False, patch_artist=True, ax=axs[0], showfliers=True)
axs[0].set_title("Distribution of Cut-in Wind Speed Across Custom Power Output Bins")
axs[0].set_xlabel("Max Power Output Bin")
axs[0].set_ylabel("Cut-in Wind Speed (m/s)")
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot for cut-out wind speed
dataframe.boxplot(column="cut-out wind speed", by="Cut-out Power Output Bin", grid=False, patch_artist=True, ax=axs[1], showfliers=True)
axs[1].set_title("Distribution of Cut-out Wind Speed Across Custom Power Output Bins")
axs[1].set_xlabel("Max Power Output Bin")
axs[1].set_ylabel("Cut-out Wind Speed (m/s)")
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and show the plots
plt.tight_layout()
plt.suptitle("")  # Remove the automatic subtitle
plt.show()