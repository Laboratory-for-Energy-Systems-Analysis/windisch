import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Exponential model (for onshore turbines)
def exponential_model(power, coeff_a, coeff_b, coeff_c, coeff_d):
    return coeff_a - coeff_b * np.exp(-(power - coeff_d) / coeff_c)


# Improved model with logarithmic term (for offshore turbines)
def improved_exponential_model(power, coeff_a, coeff_b, coeff_c, coeff_d, coeff_e):
    return (
        coeff_a
        - coeff_b * np.exp(-(power - coeff_d) / coeff_c)
        + coeff_e * np.log(power + 1)
    )


# Load the dataset
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"
data = pd.read_csv(file_path, encoding="latin-1")

# Remove the second row containing units
data = data.iloc[2:].reset_index(drop=True)

# Convert relevant columns to numeric
data["Rated power"] = pd.to_numeric(data["Rated power"], errors="coerce")
data["Rotor diameter"] = pd.to_numeric(data["Rotor diameter"], errors="coerce")

# Drop rows with missing values
data = data.dropna(subset=["Rated power", "Rotor diameter"])

# Filter onshore and offshore data
data_onshore = data[data["Offshore"] == "No"]
data_offshore = data[data["Offshore"] == "Yes"]


# Function to fit the model and estimate coefficients
def fit_model(data, title, model_func, initial_guess):
    power = data["Rated power"].values
    diameter = data["Rotor diameter"].values

    # Curve fitting
    coeffs, _ = curve_fit(model_func, power, diameter, p0=initial_guess)

    # Print the estimated coefficients
    print(
        f"{title} Coefficients: "
        + ", ".join(
            [f"coeff_{chr(97+i)}={coeff:.2f}" for i, coeff in enumerate(coeffs)]
        )
    )

    # Plot observed data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(power, diameter, label="Observed Data", color="blue", alpha=0.6)
    power_range = np.linspace(power.min(), power.max(), 500)
    fitted_curve = model_func(power_range, *coeffs)
    plt.plot(power_range, fitted_curve, label="Fitted Curve", color="red", linewidth=2)

    plt.title(f"{title} - Power vs Rotor Diameter")
    plt.xlabel("Rated Power (kW)")
    plt.ylabel("Rotor Diameter (m)")
    plt.legend()
    plt.grid()
    plt.show()

    return coeffs


# Fit the model for onshore turbines (original exponential model)
coeffs_onshore = fit_model(
    data_onshore,
    "Onshore Turbines",
    exponential_model,
    [150, 130, 2000, 20],  # Original good initial guess
)

# Fit the model for offshore turbines (improved exponential + log model)
coeffs_offshore = fit_model(
    data_offshore,
    "Offshore Turbines",
    improved_exponential_model,
    [
        np.max(data_offshore["Rotor diameter"]),  # coeff_a: Max observed diameter
        np.ptp(data_offshore["Rotor diameter"]) / 2,  # coeff_b: Approx half the range
        np.median(data_offshore["Rated power"]),  # coeff_c: Median power
        np.min(data_offshore["Rated power"]),  # coeff_d: Smallest power
        1,  # coeff_e: Log term initial guess
    ],
)
