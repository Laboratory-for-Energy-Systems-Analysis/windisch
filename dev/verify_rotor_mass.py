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

# Drop rows with missing 'Rotor weight'
data.dropna(subset=["Rotor weight"], inplace=True)

# Convert relevant columns to numeric
data["Rotor diameter"] = pd.to_numeric(
    data["Rotor diameter"].astype(str).str.replace("m", "", regex=False), errors="coerce"
)
data["Rotor weight"] = pd.to_numeric(
    data["Rotor weight"].astype(str).str.replace("Tons", "", regex=False), errors="coerce"
)

# Drop rows with missing values
data = data.dropna(subset=["Rotor diameter", "Rotor weight", "Offshore"])

# Separate onshore and offshore data
data_offshore = data.loc[data["Offshore"] == "Yes"]
data_onshore = data.loc[data["Offshore"] == "No"]

# Extract values
diameter_offshore = data_offshore["Rotor diameter"].values
mass_offshore = data_offshore["Rotor weight"].values

# **Define New Models**

# **Logarithmic Model** (Prevents overestimation for large diameters)
def log_model(diameter, a, b, c):
    return a * np.log(np.maximum(diameter, 1)) + b + c * diameter  # Prevents log(0)

# **Polynomial Model** (With Constraints)
def poly_model(diameter, a, b, c, d):
    return np.maximum(a * diameter**3 + b * diameter**2 + c * diameter + d, 0)  # Prevents negatives

# **Initial Guesses**
log_init_guess = [500, 2000, 0.1]  # a, b, c
poly_init_guess = [1e-6, 1e-3, 0.1, np.min(mass_offshore)]  # a, b, c, d

# **Fit Models**
log_params, _ = curve_fit(log_model, diameter_offshore, mass_offshore, p0=log_init_guess, maxfev=50000)
poly_params, _ = curve_fit(poly_model, diameter_offshore, mass_offshore, p0=poly_init_guess, maxfev=50000)

# **Generate Predicted Values**
diameter_range = np.linspace(min(diameter_offshore), max(diameter_offshore), 500)
mass_log_pred = log_model(diameter_range, *log_params)
mass_poly_pred = poly_model(diameter_range, *poly_params)

# **Calculate RMSE**
rmse_log = np.sqrt(np.mean((mass_offshore - log_model(diameter_offshore, *log_params)) ** 2))
rmse_poly = np.sqrt(np.mean((mass_offshore - poly_model(diameter_offshore, *poly_params)) ** 2))

# **Print Results**
print(f"RMSE (Logarithmic Model): {rmse_log:.2f}")
print(f"RMSE (Polynomial Model): {rmse_poly:.2f}")

# **Display Model Coefficients**
print(f"Logarithmic Model Coefficients: a={log_params[0]:.4f}, b={log_params[1]:.4f}, c={log_params[2]:.4f}")
print(f"Polynomial Model Coefficients: a={poly_params[0]:.8f}, b={poly_params[1]:.6f}, c={poly_params[2]:.4f}, d={poly_params[3]:.2f}")

# **Plot Results**
plt.figure(figsize=(10, 6))

# Scatter plot of observed data
plt.scatter(diameter_offshore, mass_offshore, label="Observed Data", color="blue", alpha=0.6)

# Logarithmic model plot
plt.plot(diameter_range, mass_log_pred, label=f"Log Model (RMSE={rmse_log:.2f})", color="red", linestyle="dashed")

# Polynomial model plot
plt.plot(diameter_range, mass_poly_pred, label=f"Polynomial Model (RMSE={rmse_poly:.2f})", color="green")

plt.xlabel("Rotor Diameter (m)")
plt.ylabel("Rotor Mass (tons)")
plt.title("Offshore Turbines: Rotor Mass Estimation")
plt.legend()
plt.grid()
plt.show()