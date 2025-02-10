import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define functions for foundation mass calculations
def func_mass_foundation_onshore(height, diameter):
    """
    Returns mass of onshore turbine foundations (kg).
    :param height: Tower height (m)
    :param diameter: Rotor diameter (m)
    :return: Foundation mass (kg)
    """
    return 1696e3 * (height / 80) * (diameter**2 / (100**2))


def func_mass_reinf_steel_onshore(power):
    """
    Returns mass of reinforcing steel in onshore turbine foundations (kg).
    :param power: Rated power (kW)
    :return: Mass of reinforcing steel (kg)
    """
    return np.interp(power, [750, 2000, 4500], [10210, 27000, 51900])


# Load the dataset
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"
data = pd.read_csv(file_path, encoding="latin-1")

# Remove the second row containing units
data = data.drop(index=1)

# Convert relevant columns to numeric
data["Rated power"] = pd.to_numeric(data["Rated power"], errors="coerce")
data["Minimum hub height"] = pd.to_numeric(data["Minimum hub height"], errors="coerce")
data["Maximum hub height"] = pd.to_numeric(data["Maximum hub height"], errors="coerce")
data["Rotor diameter"] = pd.to_numeric(data["Rotor diameter"], errors="coerce")
data["Tower weight"] = pd.to_numeric(data["Tower weight"], errors="coerce")

# Drop rows with missing values
data = data.dropna(
    subset=[
        "Rated power",
        "Minimum hub height",
        "Maximum hub height",
        "Rotor diameter",
        "Tower weight",
    ]
)

# Filter data for offshore and onshore turbines
data_offshore = data.loc[data["Offshore"] == "Yes"]
data_onshore = data.loc[data["Offshore"] == "No"]


# Define function to plot foundation mass
def plot_foundation_mass(data, title, color):
    rated_power = data["Rated power"].values
    rotor_diameter = data["Rotor diameter"].values
    tower_height = (
        data["Minimum hub height"].values + data["Maximum hub height"].values
    ) / 2

    # Compute predicted foundation mass
    predicted_mass = func_mass_foundation_onshore(tower_height, rotor_diameter)

    # Compute RMSE
    observed_mass = data[
        "Tower weight"
    ].values  # Assuming 'Tower weight' as proxy for foundation mass
    rmse = np.sqrt(np.mean((observed_mass - predicted_mass) ** 2))

    # Generate smooth range for power-based foundation mass
    power_range = np.linspace(rated_power.min(), rated_power.max(), 500)
    tower_height_range = np.interp(power_range, rated_power, tower_height)
    rotor_diameter_range = np.interp(power_range, rated_power, rotor_diameter)
    curve = func_mass_foundation_onshore(tower_height_range, rotor_diameter_range)

    # Plot observed data
    plt.scatter(
        rated_power, observed_mass, color=color, label="Observed Data", alpha=0.7
    )

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
    plt.ylabel("Foundation Mass (kg)")
    plt.legend()
    plt.grid()


# Create subplots
plt.figure(figsize=(14, 10))

# Plot for onshore turbines
plt.subplot(2, 1, 1)
plot_foundation_mass(data_onshore, "Onshore Turbines - Foundation Mass", "blue")

# Plot for offshore turbines
plt.subplot(2, 1, 2)
plot_foundation_mass(data_offshore, "Offshore Turbines - Foundation Mass", "green")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
