import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the regression function for nacelle mass calculation
def func_nacelle_mass(power, coeff_a, coeff_b):
    """
    Returns nacelle mass (kg) based on rated power (kW).
    :param power: Power output (kW)
    :param coeff_a: Coefficient a
    :param coeff_b: Coefficient b
    :return: Nacelle mass (kg)
    """
    nacelle_mass = coeff_a * power**2 + coeff_b * power
    return 1e3 * nacelle_mass  # Convert to kg

# Coefficients used in __set_nacelle_mass for onshore and offshore turbines
coeff_onshore = [1.66691134e-06, 3.20700974e-02]
coeff_offshore = [2.15668283e-06, 3.24712680e-02]

# Load the dataset
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"
data = pd.read_csv(file_path, encoding="latin-1")

# Remove the second row containing units
data = data.drop(index=1)

# Convert relevant columns to numeric
data["Rated power"] = pd.to_numeric(data["Rated power"], errors="coerce")
data["Nacelle weight"] = pd.to_numeric(data["Nacelle weight"], errors="coerce")

# Drop rows with missing values
data = data.dropna(subset=["Rated power", "Nacelle weight"])

# Filter data for offshore and onshore turbines
data_offshore = data.loc[data["Offshore"] == "Yes"]
data_onshore = data.loc[data["Offshore"] == "No"]

# Function to plot the observed and predicted nacelle mass
def plot_nacelle_mass(data, coeffs, title, color):
    rated_power = data["Rated power"].values
    observed_mass = data["Nacelle weight"].values

    # Generate predicted nacelle mass
    predicted_mass = func_nacelle_mass(rated_power, *coeffs)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((observed_mass - predicted_mass) ** 2))

    # Generate curve points for smoother display
    power_range = np.linspace(rated_power.min(), rated_power.max(), 500)
    curve = func_nacelle_mass(power_range, *coeffs)

    # Plot observed data
    plt.scatter(rated_power, observed_mass, color=color, label="Observed Data", alpha=0.7)

    # Plot the curve
    plt.plot(power_range, curve, color="red", label="Fitted Curve")

    # Add RMSE band
    plt.fill_between(
        power_range,
        curve - rmse,
        curve + rmse,
        color="gray",
        alpha=0.3,
        label=f"RMSE ± {rmse:.2f} kg",
    )

    # Add labels and title
    plt.title(title)
    plt.xlabel("Rated Power (kW)")
    plt.ylabel("Nacelle Mass (kg)")
    plt.legend()
    plt.grid()

# Create subplots
plt.figure(figsize=(14, 10))

# Plot for onshore turbines
plt.subplot(2, 1, 1)
plot_nacelle_mass(data_onshore, coeff_onshore, "Onshore Turbines - Nacelle Mass", "blue")

# Plot for offshore turbines
plt.subplot(2, 1, 2)
plot_nacelle_mass(data_offshore, coeff_offshore, "Offshore Turbines - Nacelle Mass