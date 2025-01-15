import pandas as pd
import numpy as np

# Load the data
file_path = 'Power_curves.csv'
data = pd.read_csv(file_path)

# Debugging: Check the structure of the data
#print("Loaded data preview:\n", data.head())
#print("Column names:\n", data.columns)

# Ensure nominal power column and wind speed columns are clean
nominal_power_col = 'Power (kW) at wind speed (m/s)'
data_filtered = data.dropna(subset=[nominal_power_col]) #I think that this row is useless!!

# Debugging: Check after dropping rows with missing nominal power
#print("\nData after filtering rows with missing nominal power:\n", data_filtered.head())

data_filtered = data_filtered.replace(np.nan, 0)

# Debugging: Check after replacing NaN with 0
#print("\nData after replacing NaN with 0:\n", data_filtered.head())

# Function to calculate cut-in and cut-out speeds
def calculate_cut_in_cut_out(power_curve):
    cut_in_speed = None
    cut_out_speed = None
    for speed, power in enumerate(power_curve):
        # Debugging: Log each speed and power value
        #print(f"Speed index: {speed}, Power: {power}")
        
        if float(power) > 0 and cut_in_speed is None:
            cut_in_speed = speed
            print(f"Cut-in speed identified: {cut_in_speed}")
        if float(power) == 0 and cut_in_speed is not None:
            cut_out_speed = speed
            #print(f"Cut-out speed identified: {cut_out_speed}")
            break
    return cut_in_speed, cut_out_speed

results = []
for _, row in data_filtered.iterrows():
    turbine_name = row['Turbine Name']  # Debugging: Check turbine name
    #print(f"\nProcessing turbine: {turbine_name}")
    
    power_curve = row.iloc[4:-2].values  # Debugging: Check power curve values
    #print(f"Power curve values: {power_curve}")
    
    nominal_power = power_curve.max()
    #print(f"Nominal power: {nominal_power}")  # Debugging: Check nominal power
    
    cut_in, cut_out = calculate_cut_in_cut_out(power_curve)
    
    results.append({
        'Turbine Name': turbine_name,
        'Nominal Power (kW)': nominal_power,
        'Cut-in Speed (m/s)': cut_in,
        'Cut-out Speed (m/s)': cut_out
    })

# Compile results
results_df = pd.DataFrame(results)

# Debugging: Check the results DataFrame
#print("\nResults DataFrame:\n", results_df)

# Calculate averages for each nominal power group
average_results = results_df.groupby('Nominal Power (kW)')[['Cut-in Speed (m/s)', 'Cut-out Speed (m/s)']].mean()

# Debugging: Check the averages
#print("\nAverage Results:\n", average_results)

# Save results
results_df.to_csv('results_df.csv')
average_results.to_csv('average_cut_in_cut_out_speeds.csv')