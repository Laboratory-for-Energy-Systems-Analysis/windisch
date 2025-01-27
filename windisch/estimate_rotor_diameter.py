import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the exponential function for fitting
def exponential_model(power, coeff_a, coeff_b, coeff_c, coeff_d):
    return coeff_a - coeff_b * np.exp(-(power - coeff_d) / coeff_c)

# Load the dataset
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"  
data = pd.read_csv(file_path, encoding="latin-1")

# Drop the second row, which contains metadata or unit descriptions
data = data.iloc[2:].reset_index(drop=True)

# Ensure Rated power and Rotor diameter columns are converted properly
data["Rated power"] = pd.to_numeric(data["Rated power"].replace("#ND", np.nan), errors="coerce")
data["Rotor diameter"] = pd.to_numeric(data["Rotor diameter"].replace("#ND", np.nan), errors="coerce")

# Filter out rows with invalid data
data = data.dropna(subset=["Rated power", "Rotor diameter"])

# Filter onshore and offshore data
data_onshore = data[data["Offshore"] == "No"]
data_offshore = data[data["Offshore"] == "Yes"]

# Function to fit the model and estimate coefficients
def fit_model(data, title):
    power = data["Rated power"].values
    diameter = data["Rotor diameter"].values
    
    # Initial guesses for coefficients
    initial_guess = [150, 130, 2000, 20]
    
    # Curve fitting
    coeffs, _ = curve_fit(exponential_model, power, diameter, p0=initial_guess)
    
    # Print the estimated coefficients
    print(f"{title} Coefficients: coeff_a={coeffs[0]:.2f}, coeff_b={coeffs[1]:.2f}, coeff_c={coeffs[2]:.2f}, coeff_d={coeffs[3]:.2f}")
    
    # Plot observed data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(power, diameter, label="Observed Data", color="blue", alpha=0.6)
    power_range = np.linspace(power.min(), power.max(), 500)
    fitted_curve = exponential_model(power_range, *coeffs)
    plt.plot(power_range, fitted_curve, label="Fitted Curve", color="red", linewidth=2)
    
    plt.title(f"{title} - Power vs Rotor Diameter")
    plt.xlabel("Rated Power (kW)")
    plt.ylabel("Rotor Diameter (m)")
    plt.legend()
    plt.grid()
    plt.show()
    
    return coeffs

# Fit the model for onshore and offshore turbines
coeffs_onshore = fit_model(data_onshore, "Onshore Turbines")
coeffs_offshore = fit_model(data_offshore, "Offshore Turbines")