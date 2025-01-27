import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the regression function for rotor mass
def func_rotor_mass(diameter, coeff_a, coeff_b):
    return coeff_a * diameter**2 + coeff_b * diameter

# Coefficients for onshore and offshore turbines
coeff_onshore = [0.00460956, 0.11199577]  # Example coefficients
coeff_offshore = [0.0088365, -0.16435292]  # Example coefficients

# Load the data
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"
data = pd.read_csv(file_path, encoding="latin-1")

# Remove the second row containing units
data = data.drop(index=1)

# Convert relevant columns to numeric
data["Rotor diameter"] = pd.to_numeric(data["Rotor diameter"].str.replace("m", "", regex=False), errors="coerce")
data["Rotor weight"] = pd.to_numeric(data["Rotor weight"].str.replace("Tons", "", regex=False), errors="coerce")

# Drop rows with missing values
data = data.dropna(subset=["Rotor diameter", "Rotor weight", "Offshore"])

# Filter data for offshore and onshore turbines
data_offshore = data.loc[data["Offshore"] == "Yes"]
data_onshore = data.loc[data["Offshore"] == "No"]

# Function to plot the data and fit
def plot_with_curve(data, coeffs, title, color):
    rotor_diameter = data["Rotor diameter"].values
    observed_mass = data["Rotor weight"].values

    # Generate predicted rotor masses
    predicted_mass = func_rotor_mass(rotor_diameter, *coeffs)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((observed_mass - predicted_mass) ** 2))

    # Generate curve points for smoother display
    diameter_range = np.linspace(rotor_diameter.min(), rotor_diameter.max(), 500)
    curve = func_rotor_mass(diameter_range, *coeffs)

    # Plot observed data
    plt.scatter(rotor_diameter, observed_mass, color=color, label="Observed Data")

    # Plot the curve
    plt.plot(diameter_range, curve, color="red", label="Fitted Curve")

    # Add RMSE band
    plt.fill_between(
        diameter_range,
        curve - rmse,
        curve + rmse,
        color="gray",
        alpha=0.3,
        label=f"RMSE ± {rmse:.2f} tons",
    )

    # Add labels and title
    plt.title(title)
    plt.xlabel("Rotor Diameter (m)")
    plt.ylabel("Rotor Mass (tons)")
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