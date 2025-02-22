import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Load dataset
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Code/windisch_folder/dev/extra data/Turbines_20230629.xlsx"
data = pd.read_excel(file_path, sheet_name="Turbines")

# Drop the first row if necessary (remove units row)
data = data.iloc[1:]
data.columns = data.columns.str.strip()  # Clean column names
data.replace("#ND", np.nan, inplace=True)  # Replace missing values

# Ensure required columns exist
required_columns = ["Maximum hub height", "Minimum hub height", "Tower weight"]
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in dataset. Check file structure.")

# Compute estimated tower height as the average of max and min hub heights
data["Tower height"] = (
    pd.to_numeric(data["Maximum hub height"], errors="coerce")
    + pd.to_numeric(data["Minimum hub height"], errors="coerce")
) / 2

# Drop rows with missing tower weight
data.dropna(subset=["Tower weight"], inplace=True)

# Convert relevant columns to numeric
data["Tower height"] = pd.to_numeric(data["Tower height"], errors="coerce")
data["Tower weight"] = pd.to_numeric(
    data["Tower weight"].astype(str).str.replace("Tons", "", regex=False),
    errors="coerce",
)

# Drop rows with missing values
data = data.dropna(subset=["Tower height", "Tower weight", "Offshore"])

# Separate offshore data
data_offshore = data.loc[data["Offshore"] == "Yes"]

# Extract values
tower_height = data_offshore["Tower height"].values
tower_mass = data_offshore["Tower weight"].values

# Get min and max tower height
min_height = np.min(tower_height)
min_mass = np.min(tower_mass)
height_range = np.linspace(min_height, np.max(tower_height), 500)

# **Debugging: Print min tower height and min tower mass**
print(f"🔹 Minimum Tower Height: {min_height:.2f} m")
print(f"🔹 Minimum Tower Mass: {min_mass:.2f} tons")


### **Corrected Logarithmic Model (Starts at Smallest Turbine)**
def corrected_log_model(height, a, b, c):
    """
    Logarithmic model forced to start at the smallest turbine mass.
    """
    return np.maximum(
        a * np.log(height - min_height + 1) + b + c * height, min_mass
    )  # Ensures no values below min_mass


### **Polynomial Model (With Constraints)**
def poly_model(height, a, b, c, d):
    return np.maximum(
        a * height**3 + b * height**2 + c * height + d, min_mass
    )  # Ensures no negatives


# **Initial Guesses for the Log Model**
log_init_guess = [1000, min_mass, 0.1]  # Ensure b starts at min_mass
poly_init_guess = [1e-6, 1e-3, 1, min_mass]  # Standard cubic polynomial fit

# **Fit Models**
log_params, _ = curve_fit(
    corrected_log_model, tower_height, tower_mass, p0=log_init_guess, maxfev=50000
)
poly_params, _ = curve_fit(
    poly_model, tower_height, tower_mass, p0=poly_init_guess, maxfev=50000
)

# **Generate Predicted Values**
mass_log_pred = corrected_log_model(height_range, *log_params)
mass_poly_pred = poly_model(height_range, *poly_params)

# **Calculate RMSE**
rmse_log = np.sqrt(
    np.mean((tower_mass - corrected_log_model(tower_height, *log_params)) ** 2)
)
rmse_poly = np.sqrt(np.mean((tower_mass - poly_model(tower_height, *poly_params)) ** 2))

# **Print Results**
print(f"RMSE (Corrected Logarithmic Model): {rmse_log:.2f}")
print(f"RMSE (Polynomial Model): {rmse_poly:.2f}")

# **Display Model Coefficients**
print("\nCorrected Logarithmic Model Coefficients:")
print(f"a = {log_params[0]:.4f}")
print(f"b = {log_params[1]:.4f} (Forced to start at min_mass)")
print(f"c = {log_params[2]:.4f}")

print("\nPolynomial Model Coefficients:")
print(f"a = {poly_params[0]:.8f}")
print(f"b = {poly_params[1]:.6f}")
print(f"c = {poly_params[2]:.4f}")
print(f"d = {poly_params[3]:.2f}")

# **Plot Results**
plt.figure(figsize=(10, 6))

# Scatter plot of observed data
plt.scatter(tower_height, tower_mass, label="Observed Data", color="blue", alpha=0.6)

# Logarithmic model plot
plt.plot(
    height_range,
    mass_log_pred,
    label=f"Corrected Log Model (RMSE={rmse_log:.2f})",
    color="red",
    linestyle="dashed",
)

# Polynomial model plot
# plt.plot(height_range, mass_poly_pred, label=f"Polynomial Model (RMSE={rmse_poly:.2f})", color="green")

plt.xlabel("Tower Height (m)")
plt.ylabel("Tower Mass (tons)")
plt.title("Offshore Turbines: Tower Mass Estimation")
plt.legend()
plt.grid()
plt.show()
