import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.optimize import curve_fit

# -------------------------------
# 1️⃣ Load Wind Speed Data from ERA5
# -------------------------------

# Load wind data
wind_data_file = "../era5_mean_2013-2022_month_by_hour_corrected.nc"
ds = xr.open_dataset(wind_data_file)

# User input: Location (latitude & longitude)
latitude = 55   # Example: 55.602
longitude = 12 # Example: 12.492

# Interpolate wind speed for given location
V = ds["wind_speed"].interp(latitude=latitude, longitude=longitude).values.item()
print(f"Interpolated Wind Speed at ({latitude}, {longitude}): {V:.2f} m/s")

# -------------------------------
# 2️⃣ ULS Moment Calculation
# -------------------------------

# Given values
Cd = 1.2  # Drag coefficient
rho = 1.225  # Air density (kg/m³)
D = 100  # Rotor diameter (m)
A = (np.pi / 4) * D**2  # Rotor swept area (m²)

# Wind force calculation
F_wind = 0.5 * Cd * rho * A * V**2

# Gravity force calculation
mass_nacelle_rotor = 100000  # kg (100 t)
g = 9.81  # Gravity (m/s²)
F_gravity = mass_nacelle_rotor * g

# Heights
H_hub = 100 + D / 2  # Hub height (m)
H_CoM = 100 + D / 3  # Approximate center of mass height (m)

# ULS Moment calculation
M_ULS = (F_wind * H_hub) + (F_gravity * H_CoM)

# Convert to MN·m
M_ULS_MN = M_ULS / 1e6

# Print the result
print(f"ULS Moment: {M_ULS_MN:.2f} MN·m")

# -------------------------------
# 3️⃣ Regression Models for Bolt, Reinforcement, and Concrete Quantities
# -------------------------------

# Define fitting functions
def linear(x, a, b):
    return a * x + b

def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def logarithmic(x, a, b):
    return a * np.log(x) + b

def exponential(x, a, b):
    return a * np.exp(b * x)

# Data
m_uls_bolt = [
    88.1, 91.9, 133.5, 123.6, 134.8, 134.0, 156.3, 186.9, 255.6,
]
bolt_mass = [
    3.51, 5.31, 3.96, 7.27, 7.14, 7.64, 7.54, 10.64, 10.38,
]

m_uls_reinforcement = [
    88.2, 88.3, 91.4, 123.8, 135.0, 135.1, 135.1, 132.4, 155.3,
    186.9, 186.7, 255.2, 255.1,
]
reinforcement_mass = [
    78.9, 62.3, 42.4, 56.6, 55.8, 63.1, 67.2, 84.3, 110.9,
    88.6, 74.7, 153.5, 186.1,
]

m_uls_concrete = [
    91.3, 87.6, 86.7, 123.2, 131.4, 132.9, 132.9, 132.5, 155.3,
    186.0, 185.8, 255.4, 256.0,
]
concrete_volume = [
    352.4, 553.6, 648.5, 443.6, 555.1, 609.0, 678.8, 767.5, 692.4,
    669.1, 894.9, 940.2, 1166.5,
]

# Organizing datasets
datasets = {
    "Bolt Mass": (np.array(m_uls_bolt), np.array(bolt_mass)),
    "Reinforcement Mass": (np.array(m_uls_reinforcement), np.array(reinforcement_mass)),
    "Concrete Volume": (np.array(m_uls_concrete), np.array(concrete_volume)),
}

# Plot and fit models
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

for ax, (name, (x_data, y_data)) in zip(axes, datasets.items()):
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    
    # Fit and plot each model
    models = {"Linear": linear, "Polynomial": polynomial, "Logarithmic": logarithmic, "Exponential": exponential}
    colors = {"Linear": "red", "Polynomial": "green", "Logarithmic": "blue", "Exponential": "purple"}
    
    for model_name, model_func in models.items():
        try:
            popt, _ = curve_fit(model_func, x_data, y_data)
            y_fit = model_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, label=f"{model_name} Fit", color=colors[model_name])
        except RuntimeError:
            pass  # Skip if the model fails to fit
    
    ax.scatter(x_data, y_data, label="Data", color="black")
    ax.set_xlabel("Design Moment ULS [MN·m]")
    ax.set_ylabel(name)
    ax.legend()
    ax.set_title(f"Fits for {name}")

plt.tight_layout()
plt.show()