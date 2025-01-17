import numpy as np
import matplotlib.pyplot as plt

# Define parameters for each wind turbine
nominal_powers = [500, 1000, 2000, 3000, 5000]  # Example nominal powers in kW
rotor_diameters = [50, 60, 80, 100, 120]  # Example rotor diameters in meters
cut_in_speeds = [3, 3, 3, 3.5, 3.5]  # Cut-in wind speeds in m/s
cut_out_speeds = [25, 25, 25, 24, 24]  # Cut-out wind speeds in m/s

# Environmental and operational constants
Vws = np.arange(0, 30, 0.01)  # Wind speeds for the power curve (m/s)
TI = 0.1  # Turbulence intensity
Shear = 0.15  # Wind shear
Veer = 0  # Wind veer
AirDensity = 1.225  # Air density in kg/m³

# Plot the power curves for all turbines
plt.figure(figsize=(12, 8))

for i, (Pnom, Drotor, Vcutin, Vcutoff) in enumerate(zip(nominal_powers, rotor_diameters, cut_in_speeds, cut_out_speeds)):
    # Generate power curve
    Pwt = GenericWindTurbinePowerCurve(
        Vws=Vws,
        Pnom=Pnom,
        Drotor=Drotor,
        Vcutin=Vcutin,
        Vcutoff=Vcutoff,
        TI=TI,
        Shear=Shear,
        Veer=Veer,
        AirDensity=AirDensity,
    )

    # Plot the power curve
    plt.plot(Vws, Pwt, label=f"Pnom={Pnom} kW, Drotor={Drotor} m")

# Add labels, legend, and grid
plt.title("Generic Wind Turbine Power Curves")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (kW)")
plt.grid()
plt.legend(loc="upper left", fontsize=10)
plt.show()