import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define the regression function
def func_rotor_diameter(power, coeff_a, coeff_b, coeff_c, coeff_d):
    return coeff_a - coeff_b * np.exp(-(power - coeff_d) / coeff_c)


# Previous coefficients for onshore and offshore turbines
coeff_onshore = [152.66, 136.57, 2478.03, 16.44]
coeff_offshore = [191.84, 147.37, 5101.29, 376.63]

# TO CHANGE : new coefficients after having found them in the script estimate_rotor_diameter.py
# coeff_onshore = [179.23, 144.91, 3061.77, 371.14]
# coeff_offshore = [16702.89, 10346.16, 2214138.00, -9896.84]

# Load the data
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"
data = pd.read_csv(file_path, encoding="latin-1")

# Remove the second row containing units
data = data.drop(index=1)

# Convert relevant columns to numeric
data["Rated power"] = pd.to_numeric(data["Rated power"], errors="coerce")
data["Rotor diameter"] = pd.to_numeric(data["Rotor diameter"], errors="coerce")

# Drop rows with missing values
data = data.dropna(subset=["Rated power", "Rotor diameter"])

# Filter data for offshore and onshore turbines
data_offshore = data.loc[data["Offshore"] == "Yes"]
data_onshore = data.loc[data["Offshore"] == "No"]


# Function to plot the data and fit
def plot_with_curve(data, coeffs, title, color):
    rated_power = data["Rated power"].values
    observed_diameter = data["Rotor diameter"].values

    # Generate predicted rotor diameters
    predicted_diameter = func_rotor_diameter(rated_power, *coeffs)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((observed_diameter - predicted_diameter) ** 2))

    # Generate curve points for smoother display
    power_range = np.linspace(rated_power.min(), rated_power.max(), 500)
    curve = func_rotor_diameter(power_range, *coeffs)

    # Plot observed data
    plt.scatter(rated_power, observed_diameter, color=color, label="Observed Data")

    # Plot the curve
    plt.plot(power_range, curve, color="red", label="Fitted Curve")

    # Add RMSE band
    plt.fill_between(
        power_range,
        curve - rmse,
        curve + rmse,
        color="gray",
        alpha=0.3,
        label=f"RMSE ± {rmse:.2f} m",
    )

    # Add labels and title
    plt.title(title)
    plt.xlabel("Rated Power (kW)")
    plt.ylabel("Rotor Diameter (m)")
    plt.legend()
    plt.grid()


# Create subplots
plt.figure(figsize=(14, 10))

# Plot for onshore turbines
plt.subplot(2, 1, 1)
plot_with_curve(data_onshore, coeff_onshore, "Onshore Turbines", "blue")

# Plot for offshore turbines
plt.subplot(2, 1, 2)
plot_with_curve(data_offshore, coeff_offshore, "Offshore Turbines", "green")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
