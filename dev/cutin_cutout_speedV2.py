import pandas as pd
import numpy as np

# Loading the data
file_path = 'Power_curves.csv'
data = pd.read_csv(file_path)

# Creating a DataFrame of size (881x71) containing the power outputs and corresponding wind speeds - the first row is the wind speed
dataframe = data.iloc[0:881, 4:75]

# Adding a new column named containing the maximum power output of each wind turbine (starting from the second row: the row after the wind speeds)
dataframe['Max Power Output'] = dataframe.iloc[1:].max(axis=1)

# Adding the new columns for "cut-in wind speed" and "cut-out wind speed"
dataframe["cut-in wind speed"] = None
dataframe["cut-out wind speed"] = None

# Iterating through each row starting from row 2 (index 1)
for index, row in dataframe.iloc[1:].iterrows():
    # Finding the first non-zero column and set the corresponding wind speed (cut-in wind speed)
    cut_in_col = (row.iloc[:71] != 0).idxmax()  # Finding the first column index where value isn't zero
    if cut_in_col in dataframe.columns:  # Ensuring the column is valid - maybe not useful finally
        dataframe.at[index, "cut-in wind speed"] = dataframe.at[0, cut_in_col]

    # Finding the last non-zero column and set the corresponding wind speed (cut-off wind speed)
    cut_out_col = (row.iloc[:71] != 0)[::-1].idxmax()  # Finding the last column index where value isn't zero
    if cut_out_col in dataframe.columns:  # Ensuring the column is valid - maybe not useful finally
        dataframe.at[index, "cut-out wind speed"] = dataframe.at[0, cut_out_col]

# Display the matrix
print(dataframe)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define bins for "Max Power Output"
bin_edges = np.linspace(dataframe["Max Power Output"].min(), dataframe["Max Power Output"].max(), 6)  # Create 5 equal bins
bin_labels = [f"Bin {i+1}" for i in range(len(bin_edges) - 1)]  # Label bins as Bin 1, Bin 2, etc.

# Assign bins to "Max Power Output"
dataframe["Power Output Bin"] = pd.cut(dataframe["Max Power Output"], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Create a boxplot for each bin showing "cut-in wind speed"
plt.figure(figsize=(10, 6))
dataframe.boxplot(column="cut-in wind speed", by="Power Output Bin", grid=False, patch_artist=True, showfliers=True)
plt.title("Distribution of Cut-in Wind Speed Across Max Power Output Bins")
plt.suptitle("")  # Remove the automatic subtitle
plt.xlabel("Max Power Output Bin")
plt.ylabel("Cut-in Wind Speed (m/s)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()