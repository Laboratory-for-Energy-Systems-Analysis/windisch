import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Hub height model
def hub_height_model(power, coeff_a, coeff_b, coeff_c):
    return coeff_a - coeff_b * np.exp(-power / coeff_c)


# Load the data
file_path = "/Users/kalenajonsson/Desktop/SemesterProject/Turbines_data.csv"
data = pd.read_csv(file_path, encoding="latin-1")

# Data cleaning and preprocessing
data = data.iloc[2:]  # Skip metadata rows
data["Rated power"] = pd.to_numeric(
    data["Rated power"].replace("#ND", np.nan), errors="coerce"
)
data["Minimum hub height"] = pd.to_numeric(
    data["Minimum hub height"].replace("#ND", np.nan), errors="coerce"
)
data["Maximum hub height"] = pd.to_numeric(
    data["Maximum hub height"].replace("#ND", np.nan), errors="coerce"
)
data.dropna(
    subset=["Rated power", "Minimum hub height", "Maximum hub height"], inplace=True
)

# Calculate average hub height if both min and max are available
data["Observed hub height"] = data[["Minimum hub height", "Maximum hub height"]].mean(
    axis=1
)

# Separate onshore and offshore turbines
data_onshore = data[data["Offshore"] == "No"]
data_offshore = data[data["Offshore"] == "Yes"]


# Fit the model and analyze
def analyze_hub_height(data_subset, turbine_type):
    rated_power = data_subset["Rated power"].values
    observed_hub_height = data_subset["Observed hub height"].values

    # Fit the model to the data
    initial_guess = [120, 90, 2500]  # Initial guesses for coeff_a, coeff_b, coeff_c
    coeffs, _ = curve_fit(
        hub_height_model, rated_power, observed_hub_height, p0=initial_guess
    )

    # Predicted hub heights
    predicted_hub_height = hub_height_model(rated_power, *coeffs)

    # Calculate error metrics
    mae = mean_absolute_error(observed_hub_height, predicted_hub_height)
    rmse = np.sqrt(mean_squared_error(observed_hub_height, predicted_hub_height))

    # Plot observed vs. predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(
        rated_power, observed_hub_height, label="Observed", color="blue", alpha=0.6
    )
    plt.plot(
        np.sort(rated_power),
        hub_height_model(np.sort(rated_power), *coeffs),
        label="Fitted Curve",
        color="red",
    )
    plt.fill_between(
        np.sort(rated_power),
        hub_height_model(np.sort(rated_power), *coeffs) - rmse,
        hub_height_model(np.sort(rated_power), *coeffs) + rmse,
        color="gray",
        alpha=0.3,
        label="RMSE Range",
    )
    plt.title(f"{turbine_type} Turbines: Hub Height Verification")
    plt.xlabel("Rated Power (kW)")
    plt.ylabel("Hub Height (m)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print results
    print(f"{turbine_type} Turbines:")
    print(
        f"Coefficients: coeff_a={coeffs[0]:.2f}, coeff_b={coeffs[1]:.2f}, coeff_c={coeffs[2]:.2f}"
    )
    print(f"Mean Absolute Error (MAE): {mae:.2f} m")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} m\n")


# Analyze onshore and offshore turbines
analyze_hub_height(data_onshore, "Onshore")
analyze_hub_height(data_offshore, "Offshore")
