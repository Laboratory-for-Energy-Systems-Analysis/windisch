import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Load dataset
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Code/windisch_folder/dev/extra data/Turbines_20230629.xlsx"
data = pd.read_excel(file_path, sheet_name="Turbines")

# Drop the first row if it contains units
data = data.iloc[1:]
data.columns = data.columns.str.strip()  # Clean column names
data.replace("#ND", np.nan, inplace=True)  # Replace missing values
data.dropna(subset=["Nacelle weight"], inplace=True)  # Remove missing nacelle weights

# Convert columns to numeric
data["Nacelle weight"] = (
    pd.to_numeric(data["Nacelle weight"], errors="coerce") * 1000
)  # Convert tons to kg
data["Rated power"] = pd.to_numeric(data["Rated power"], errors="coerce")

# Drop any remaining NaN values
data = data.dropna(subset=["Rated power", "Nacelle weight"])

# Filter offshore turbines
data_offshore = data.loc[data["Offshore"] == "Yes"]

# Extract power and mass data
power = data_offshore["Rated power"].values
mass = data_offshore["Nacelle weight"].values

# Normalize power to improve numerical stability
power_norm = power / np.max(power)


### **Improved Logarithmic Model**
def log_model(power, a, b, c, d):
    return (
        a * np.log(np.maximum(power + c, 1e-2)) + b + d * power
    )  # Ensure positive log input


### **Improved Polynomial Model**
def poly_model(power, a, b, c, d):
    return np.maximum(
        a * power**3 + b * power**2 + c * power + d, 0
    )  # No negative values


# Smarter Initial Guesses
log_init_guess = [np.max(mass), np.min(mass), np.min(power) * 0.5, 0.01]
poly_init_guess = [1e-9, 1e-6, 1e-2, np.min(mass)]

# Fit models using curve_fit with higher maxfev
try:
    log_params, _ = curve_fit(log_model, power, mass, p0=log_init_guess, maxfev=50000)
except RuntimeError:
    print("Log model did not converge. Trying alternative initial guesses...")
    log_init_guess = [100000, 1000, 100, 0.001]  # Alternative
    log_params, _ = curve_fit(log_model, power, mass, p0=log_init_guess, maxfev=50000)

poly_params, _ = curve_fit(poly_model, power, mass, p0=poly_init_guess, maxfev=50000)

# Generate predicted values
power_range = np.linspace(min(power), max(power), 500)
mass_log_pred = log_model(power_range, *log_params)
mass_poly_pred = poly_model(power_range, *poly_params)

# Calculate RMSE for both models
rmse_log = np.sqrt(np.mean((mass - log_model(power, *log_params)) ** 2))
rmse_poly = np.sqrt(np.mean((mass - poly_model(power, *poly_params)) ** 2))

# Print RMSE values
print(f"RMSE (Fixed Logarithmic Model): {rmse_log:.2f}")
print(f"RMSE (Improved Cubic Polynomial Model): {rmse_poly:.2f}")

# Print found coefficients
print("\nLogarithmic Model Coefficients:")
print(f"a = {log_params[0]:.2f}")
print(f"b = {log_params[1]:.2f}")
print(f"c = {log_params[2]:.2f}")
print(f"d = {log_params[3]:.6f}")

print("\nPolynomial Model Coefficients:")
print(f"a = {poly_params[0]:.9f}")
print(f"b = {poly_params[1]:.9f}")
print(f"c = {poly_params[2]:.6f}")
print(f"d = {poly_params[3]:.2f}")

# Plot results
plt.figure(figsize=(10, 6))

# Scatter plot of observed data
plt.scatter(power, mass, label="Observed Data", color="blue", alpha=0.6)

# Logarithmic model plot
plt.plot(
    power_range,
    mass_log_pred,
    label=f"Log Model (RMSE={rmse_log:.2f})",
    color="red",
    linestyle="dashed",
)

# Polynomial model plot
plt.plot(
    power_range,
    mass_poly_pred,
    label=f"Cubic Poly Model (RMSE={rmse_poly:.2f})",
    color="green",
)

plt.xlabel("Rated Power (kW)")
plt.ylabel("Nacelle Mass (kg)")
plt.title("Offshore Turbines: Nacelle Mass Estimation")
plt.legend()
plt.grid()
plt.show()
