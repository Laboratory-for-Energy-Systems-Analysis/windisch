"""
model.py contains `WindTurbineModel` which sizes wind turbines
and calculates dimensions and mass attributes.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from windisch.power_curve import calculate_generic_power_curve

from . import DATA_DIR
from .distance_to_coastline import find_nearest_coastline
from .sea_depth import get_sea_depth
from .wind_speed import fetch_terrain_variables, fetch_wind_speed

# material densities, in kg/m3
COPPER_DENSITY = 8960
STEEL_DENSITY = 8000


def func_height_diameter(
    diameter: int, coeff_a: float, coeff_b: float, coeff_c: float
) -> float:
    """
    Returns hub height, in m, based on rated diameter (m).
    :param diameter: diameter (m)
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :param coeff_c: coefficient
    :return: hub height (m)
    """
    return coeff_a - coeff_b * np.exp(-diameter / coeff_c)


# def func_rotor_weight_rotor_diameter(
#    power: int, coeff_a: float, coeff_b: float
# ) -> float:
#    """
#    Returns rotor weight, in kg, based on rotor diameter.
#    :param power: power output (kW)
#    :param coeff_a: coefficient a
#    :param coeff_b: coefficient b
#    :return: rotor weight (in kg)
#    """
#    rotor_mass = coeff_a * power**2 + coeff_b * power
#    return 1e3 * rotor_mass


def func_rotor_weight_rotor_diameter(
    diameter: float, coeff_a: float, coeff_b: float, coeff_c: float, coeff_d: float
) -> float:
    """
    Returns rotor weight (kg) based on rotor diameter (m) using a cubic polynomial model.

    :param diameter: Rotor diameter (m)
    :param coeff_a: Cubic coefficient
    :param coeff_b: Quadratic coefficient
    :param coeff_c: Linear coefficient
    :param coeff_d: Constant term
    :return: Rotor weight (kg)
    """
    rotor_mass = (
        coeff_a * diameter**3 + coeff_b * diameter**2 + coeff_c * diameter + coeff_d
    )
    # return max(0, rotor_mass)  # Ensure non-negative mass
    return rotor_mass


# def func_nacelle_weight_power(power: int, coeff_a: float, coeff_b: float) -> float:
#    """
#    Returns nacelle weight, in kg.
#    :param power: power output (kW)
#    :param coeff_a: coefficient a
#    :param coeff_b: coefficient b
#    :return: nacelle weight (in kg)
#    """
#    nacelle_mass = coeff_a * power**2 + coeff_b * power
#    return 1e3 * nacelle_mass


def func_nacelle_weight_power(
    power: int, coeff_a: float, coeff_b: float, coeff_c: float, coeff_d: float
) -> float:
    """
    Returns nacelle weight, in kg, based on rated power.

    :param power: Power output (kW)
    :param coeff_a: Cubic coefficient
    :param coeff_b: Quadratic coefficient
    :param coeff_c: Linear coefficient
    :param coeff_d: Constant term
    :return: Nacelle weight (in kg)
    """
    nacelle_mass = coeff_a * power**3 + coeff_b * power**2 + coeff_c * power + coeff_d
    # return max(0, nacelle_mass)  # Ensure non-negative mass
    return nacelle_mass


def func_rotor_diameter(
    power: int,
    coeff_a: float,
    coeff_b: float,
    coeff_c: float,
    coeff_d: float,
    coeff_e: float,
) -> float:
    """
    Returns rotor diameter, based on power output and given coefficients
    :param power: power output (kW)
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :param coeff_c: coefficient
    :param coeff_d: coefficient
    :param coeff_e: coefficient
    :return: rotor diameter (m)
    """
    return (
        coeff_a
        - coeff_b * np.exp(-(power - coeff_d) / coeff_c)
        + coeff_e * np.log(power + 1)
    )


def func_mass_reinf_steel_onshore(power: int) -> float:
    """
    Returns mass of reinforcing steel in onshore turbine foundations, based on power output (kW).
    :param power: power output (kW)
    :return:
    """
    return np.interp(power, [750, 2000, 4500], [10210, 27000, 51900])


def penetration_depth_fit() -> np.poly1d:
    """
    Return a penetration depth fit model of the steel pile of the offshore wind turbine.
    :return:
    """
    # meters
    depth = [22.5, 22.5, 23.5, 26, 29.5]
    # kW
    power = [3000, 3600, 4000, 8000, 10000]
    fit_penetration = np.polyfit(power, depth, 1)
    f_fit_penetration = np.poly1d(fit_penetration)
    return f_fit_penetration


def get_pile_height(power: int, sea_depth: float) -> float:
    """
    Returns undersea pile height (m) from rated power output (kW), penetration depth and sea depeth.
    :param power: power output (kW)
    :param sea_depth: sea depth (m)
    :return: pile height (m)
    """
    fit_penetration_depth = penetration_depth_fit()
    return 9 + fit_penetration_depth(power) + sea_depth


def get_pile_mass(power: int, pile_height: float) -> float:
    """
    Return the mass of the steel pile based on the power output of the rotor and the height of the pile.
    :param power: power output (in kW) of the rotor
    :param pile_height: height (in m) of the pile
    :return: mass of the steel pile (in kg)
    """

    # The following lists store data on the relationship
    # between the power output of the rotor and the diameter of the pile.
    # diameters, in meters
    diameter_x = [5, 5.5, 5.75, 6.75, 7.75]
    # kW
    power_y = [3000, 3600, 4000, 8000, 10000]

    # Use polynomial regression to find the function that best fits the data.
    # This function relates the diameter of the pile with the power output of the rotor.
    fit_diameter = np.polyfit(power_y, diameter_x, 1)
    f_fit_diameter = np.poly1d(fit_diameter)

    # Calculate the outer diameter of the pile based on the power output of the rotor.
    outer_diameter = f_fit_diameter(power)

    # Calculate the cross-section area of the pile based on the outer diameter.
    outer_area = (np.pi / 4) * (outer_diameter**2)

    # Calculate the volume of the pile based on the outer area and the pile height.
    outer_volume = outer_area * pile_height

    # Calculate the inner diameter of the pile based on the power output of the rotor and the thickness of the pile.
    inner_diameter = outer_diameter
    pile_thickness = np.interp(
        power,
        [2000, 3000, 3600, 4000, 8000, 10000],
        [0.07, 0.10, 0.13, 0.16, 0.19, 0.22],
    )
    inner_diameter -= 2 * pile_thickness

    # Calculate the cross-section area of the inner part of the pile.
    inner_area = (np.pi / 4) * (inner_diameter**2)

    # Calculate the volume of the inner part of the pile based on the inner area and the pile height.
    inner_volume = inner_area * pile_height

    # Calculate the volume of steel used in the pile by subtracting the inner volume from the outer volume.
    volume_steel = outer_volume - inner_volume

    # Calculate the weight of the steel used in the pile based on its volume and density.
    weight_steel = STEEL_DENSITY * volume_steel

    # Return the weight of the steel pile.
    return weight_steel


def get_transition_height() -> np.poly1d:
    """
    Returns a fitting model for the height of
    transition piece (in m), based on pile height (in m).
    :return:
    """
    pile_length = [35, 50, 65, 70, 80]
    transition_length = [15, 20, 24, 30, 31]
    fit_transition_length = np.polyfit(pile_length, transition_length, 1)
    return np.poly1d(fit_transition_length)


def get_transition_mass(transition_length: float) -> float:
    """
    Returns the mass of transition piece (in kg).
    :return:
    """
    transition_lengths = [15, 20, 24, 30]
    transition_weight = [150, 200, 260, 370]
    fit_transition_weight = np.polyfit(transition_lengths, transition_weight, 1)

    return np.poly1d(fit_transition_weight)(transition_length) * 1000


def get_grout_volume(trans_length: float) -> float:
    """
    Returns grout volume (m3) based on transition piece length (in m).
    :param trans_length: length of the transition piece (in m)
    :return: grout volume (in m3) needed
    """
    transition_length = [15, 20, 15, 20, 15, 24, 20, 30, 20, 31]
    grout = [15, 35, 15, 35, 20, 40, 25, 60, 30, 65]
    fit_grout = np.polyfit(transition_length, grout, 1)
    return np.poly1d(fit_grout)(trans_length)


def get_scour_volume(power: int) -> float:
    """
    Returns scour volume (m3) based on power output (kW).
    Scour is a mix of gravel and cement.
    :param power: power output (kW)
    :return: scour volume (m3)
    """
    scour = [2200, 2200, 2600, 3100, 3600]
    turbine_power = [3000, 3600, 4000, 8000, 10000]
    fit_scour = np.polyfit(turbine_power, scour, 1)
    return np.poly1d(fit_scour)(power)


def func_tower_weight_d2h(
    diameter: float, height: float, coeff_a: float, coeff_b: float
) -> float:
    """
    Returns tower mass, in kg, based on tower diameter and height.
    :param diameter: tower diameter (m)
    :param height: tower height (m)
    :param coeff_a: coefficient a
    :param coeff_b: coefficient b
    :return: tower mass (in kg)
    """
    tower_mass = coeff_a * diameter**2 * height + coeff_b
    return 1e3 * tower_mass


def func_tower_weight_log(
    height: float,
    coeff_a: float,
    coeff_b: float,
    coeff_c: float,
    min_height: float,
    min_mass: float,
) -> float:
    """
    Returns tower mass (kg) based on tower height using a logarithmic model (for offshore turbines).

    :param height: tower height (m)
    :param coeff_a: coefficient a (logarithmic model for offshore)
    :param coeff_b: coefficient b (logarithmic model for offshore)
    :param coeff_c: coefficient c (logarithmic model for offshore)
    :param min_height: fixed minimum tower height in dataset (ensures proper log scaling)
    :param min_mass: fixed minimum tower mass in dataset (ensures no negative values)
    :return: tower mass (kg)
    """
    return np.maximum(
        coeff_a * np.log(height - min_height + 1) + coeff_b + coeff_c * height, min_mass
    )


def set_onshore_cable_requirements(
    power,
    tower_height,
    distance_m=550,
    voltage_kv=33,
    power_factor=0.95,
    resistivity_copper=1.68e-8,
    max_voltage_drop_percent=3,
):
    """
    Calculate the required cross-sectional area of a copper cable for a wind turbine connection.

    :param power: Power output of the wind turbine in MW
    :param distance_m: Distance from the wind turbine to the transformer in meters
    :param voltage_kv: Voltage of the cable in kV
    :param power_factor: Power factor of the wind turbine
    :param resistivity_copper: Resistivity of copper in ohm-meters
    :param max_voltage_drop_percent: Maximum allowable voltage drop as a percentage of the voltage
    :return: Copper mass in kg

    """
    # Convert input parameters to standard units
    voltage_v = voltage_kv * 1e3  # Convert kV to V
    max_voltage_drop = (
        max_voltage_drop_percent / 100
    ) * voltage_v  # Maximum voltage drop in volts

    # Calculate current (I) using the formula: I = P / (sqrt(3) * V * PF)
    current_a = (power * 1000) / (3**0.5 * voltage_v * power_factor)

    # Calculate the total cable length (round trip)
    total_length_m = 2 * distance_m

    # Calculate the required resistance per meter to stay within the voltage drop limit
    max_resistance_per_meter = max_voltage_drop / (current_a * total_length_m)

    # Calculate the required cross-sectional area using R = rho / A
    cross_section_area_m2 = resistivity_copper / max_resistance_per_meter

    # Convert cross-sectional area to mm²
    cross_section_area_mm2 = cross_section_area_m2 * 1e6

    copper_mass = cross_section_area_mm2 * total_length_m * 1e-6 * COPPER_DENSITY

    # Also, add the cable inside the wind turbine, which has a 640 mm2 cross-section
    copper_mass += 640 * 1e-6 * tower_height * COPPER_DENSITY

    return copper_mass


def set_offshore_cable_requirements(
    power: int,
    cross_section: float,
    dist_transfo: float,
    dist_coast: float,
    park_size: int,
) -> Tuple[float, float]:
    """
    Return the required cable mass as well as the energy needed to lay down the cable.
    :param power: rotor power output (in kW)
    :param cross_section: cable cross-section (in mm2)
    :param dist_transfo: distance to transformer (in m)
    :param dist_coast: distance to coastline (in m)
    :param park_size:
    :return:
    """

    m_copper = (cross_section * 1e-6 * dist_transfo * 1000) * COPPER_DENSITY

    # 450 l diesel/hour for the ship that lays the cable at sea bottom
    # 39 MJ/liter, 15 km/h as speed of laying the cable
    energy_cable_laying_ship = 450 * 39 / 15 * dist_transfo

    # Cross-section calculated based on the farm cumulated power,
    # and the transport capacity of the Nexans cables @ 150kV
    # if the cumulated power of the park cannot be transported @ 33kV

    # Test if the cumulated power of the wind farm is inferior to 30 MW,
    # If so, we use 33 kV cables.

    cross_section_ = np.where(
        power * park_size <= 30e3,
        np.interp(
            power * park_size,
            np.array([352, 399, 446, 502, 581, 652, 726, 811, 904, 993]) * 33,
            np.array([95, 120, 150, 185, 240, 300, 400, 500, 630, 800]),
        ),
        np.interp(
            power * park_size,
            np.array([710, 815, 925, 1045, 1160, 1335, 1425, 1560]) * 150,
            np.array([400, 500, 630, 800, 1000, 1200, 1600, 2000]),
        ),
    )

    m_copper += (
        cross_section_ * 1e-6 * (dist_coast * 1e3 / park_size)
    ) * COPPER_DENSITY

    # 450 l diesel/hour for the ship that lays the cable at sea bottom
    # 39 MJ/liter, 15 km/h as speed of laying the cable
    energy_cable_laying_ship += 450 * 39 / 15 * dist_coast / park_size

    return m_copper, energy_cable_laying_ship * 0.5


class WindTurbineModel:
    """
    This class represents the entirety of the turbines considered,
    with useful attributes, such as an array that stores
    all the turbine input parameters.

    :ivar array: multidimensional numpy-like array that contains parameters' value(s)
    :vartype array: xarray.DataArray
    :ivar country: Country where the wind turbines are located
    :vartype country: str
    :ivar location: Location of the wind turbines
    :vartype location: Tuple[float, float]

    """

    def __init__(
        self,
        array: xr.DataArray,
        country: str = None,
        location: Tuple[float, float] = None,
        wind_data: xr.DataArray = None,
        sea_depth_data: xr.DataArray = None,
        power_curve_model="Dai et al. 2016",
    ):
        self.terrain_vars = None
        self.array = array
        self.power_curve = None
        self.__cache = None
        self.country = country or "CH"
        self.location = location or None
        self.wind_data = wind_data
        self.power_curve_model = power_curve_model
        self.sea_depth_data = sea_depth_data

        if self.location:
            self.__fetch_terrain_variables(
                fetch_wind_data=True if not self.wind_data else False
            )

            if self.wind_data:
                self.__fetch_wind_speed()

            if self.terrain_vars["LANDMASK"].max() == 1:
                print("Onshore wind turbines")
                self.array.loc[dict(application="offshore", parameter="power")] = 0
            else:
                print("Offshore wind turbines")
                self.array.loc[dict(application="onshore", parameter="power")] = 0
                if self.sea_depth_data:
                    self.__fetch_sea_depth()

    def __getitem__(self, key: Union[str, List[str]]):
        """
        Make class['foo'] automatically filter for the parameter 'foo'
        Makes the model code much cleaner

        :param key: Parameter name
        :type key: str
        :return: `array` filtered after the parameter selected
        """

        return self.array.sel(parameter=key)

    def __setitem__(self, key: Union[str, List[str]], value: Union[int, float]):
        """
        Allows to directly assign the value `value` for the parameter `key`

        .. code-block:: python

            class['key', 'value']

        :param key: Parameter name
        :param value: Numeric value (int or float)
        :return: Nothing. Modifies in place.
        """
        self.array.loc[{"parameter": key}] = value

    def set_all(self):
        """
        This method runs a series of methods to size the wind turbines,
        evaluate material requirements, etc.

        :returns: Does not return anything. Modifies ``self.array`` in place.

        """

        self.__set_size_rotor()
        self.__set_tower_height()
        self.__set_nacelle_mass()
        self.__set_rotor_mass()
        self.__set_tower_mass()
        self.__set_electronics_mass()
        self.__set_foundation_mass()
        self.__set_assembly_requirements()
        self.__set_installation_requirements()
        self.__set_maintenance_energy()
        # self.disable_unavailable_models()

        self["total mass"] = self[
            [
                "rotor mass",
                "nacelle mass",
                "tower mass",
                "electronics mass",
                "cable mass",
                "foundation mass",
            ]
        ].sum(dim="parameter")

        # if location is given, fetch wind speeds and generate power curve
        if self.location:
            self.__fetch_power_curves()
            self.__calculate_electricity_production()
            self.__calculate_average_load_factor()
        else:
            # otherwise, fetch country-average load factors
            if self.country:
                self.__fetch_country_load_factor()
                self.__calculate_electricity_production()
            else:
                raise ValueError("Location or country must be provided")

    def __fetch_country_load_factor(self):

        df = pd.read_csv(DATA_DIR / "wind_capacity_factors.csv", index_col=0)

        if self.country in df.index:
            if "onshore" in self.array.coords["application"].values:
                self.array.loc[
                    dict(application="onshore", parameter="average load factor")
                ] = df.loc[self.country, "onshore"]

            if "offshore" in self.array.coords["application"].values:
                if df.loc[self.country, "offshore"] > 0:
                    self.array.loc[
                        dict(application="offshore", parameter="average load factor")
                    ] = df.loc[self.country, "offshore"]

        else:
            ValueError(f"Country {self.country} not found in the database")

    def __fetch_sea_depth(self):

        self["sea depth"] = (
            get_sea_depth(self.sea_depth_data, self.location[0], self.location[1])
            * -1
            * (self["offshore"] == 1)
        )

    def __fetch_wind_speed(self):

        wind_speed = fetch_wind_speed(
            self.wind_data.interp(
                latitude=self.location[0],
                longitude=self.location[1],
                method="linear",
            )
        )

        for var in self.terrain_vars.data_vars:
            wind_speed[var] = self.terrain_vars[var]

        for coord in self.terrain_vars.coords:
            wind_speed[coord] = self.terrain_vars[coord]

        self.terrain_vars = wind_speed
        # rename "wind_speed" to "WS"
        self.terrain_vars = self.terrain_vars.rename_vars(
            {
                "wind_speed": "WS",
            }
        )
        self.terrain_vars["WS"] = self.terrain_vars["WS"].astype(float)

        # replace NaNs with zeros
        self.terrain_vars = self.terrain_vars.fillna(0)

    def __fetch_terrain_variables(self, fetch_wind_data: bool):
        """
        Fetch wind speeds and directions, turbulent kinetic energy,
        land mask, and air density at the location of the wind turbines.
        Values are fetched for heights of 50 and 150m.
        :return:
        """
        terrain_vars = fetch_terrain_variables(
            latitude=self.location[0],
            longitude=self.location[1],
            fetch_wind_data=fetch_wind_data,
        )
        self.terrain_vars = terrain_vars

    def __fetch_power_curves(self):

        # we get the power curve
        self.power_curve = calculate_generic_power_curve(
            power=self["power"],
        )

        self.power_curve = self.power_curve.drop_vars("power")
        self.power_curve = self.power_curve.drop_vars("parameter")

    def __calculate_electricity_production(self):
        # we calculate the electricity production
        if self.power_curve is not None:

            # we adjust values to the heights of the wind turbines
            self.terrain_vars = self.terrain_vars.interp(
                height=self["tower height"],
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )

            self.terrain_vars = self.terrain_vars.drop_vars("height")
            self.terrain_vars = self.terrain_vars.drop_vars("parameter")

            self["average wind speed"] = self.terrain_vars["WS"].mean(dim="time")

            self.annual_electricity_production = xr.zeros_like(self.terrain_vars["WS"])

            for application in self.array.coords["application"].values:
                self.annual_electricity_production.loc[
                    dict(application=application)
                ] = self.power_curve.sel(application=application).interp(
                    {
                        "wind speed": self.terrain_vars.sel(application=application)[
                            "WS"
                        ]
                    },
                    method="linear",
                )

            self["lifetime electricity production"] = (
                self.annual_electricity_production.sum(dim="time") * self["lifetime"]
            )
        else:
            if self.country:

                self["lifetime electricity production"] = (
                    self["average load factor"]
                    * self["power"]
                    * 24
                    * 365
                    * self["lifetime"]
                )

    def __calculate_average_load_factor(self):
        # we calculate the average load factor
        self["average load factor"] = self.annual_electricity_production.sum(
            dim="time"
        ) / (8760 * self["power"])

    def __set_size_rotor(self):
        """
        This method defines the rotor diameter, based on its power output.
        :return:
        """

        self["rotor diameter"] = func_rotor_diameter(
            self["power"], 179.23, 164.92, 3061.77, -24.98, 00.0
        ) * (1 - self["offshore"])

        self["rotor diameter"] += (
            func_rotor_diameter(
                self["power"], 15662.58, 9770.48, 2076442.81, 994711.94, 24.40
            )
            * self["offshore"]
        )

    def __set_tower_height(self):
        """
        This method defines the hub height based on the power output.
        :return:
        """

        self["tower height"] = func_height_diameter(
            self["rotor diameter"], -611916.49, -611936.55, -862547.53
        ) * (1 - self["offshore"])

        self["tower height"] += (
            func_height_diameter(self["rotor diameter"], 127.97, 127.08, 82.23)
            * self["offshore"]
        )

    def __set_nacelle_mass(self):
        """
        This method defines the mass of the nacelle on the power output.
        :return:
        """

        self["nacelle mass"] = func_nacelle_weight_power(
            self["power"], -0.000000323, 0.006014822, 21.251620, 2265.25
        ) * (1 - self["offshore"])

        self["nacelle mass"] += (
            func_nacelle_weight_power(
                self["power"], -0.000000323, 0.006014822, 21.251620, 2265.25
            )
            * self["offshore"]
        )

    # def __set_rotor_mass(self):
    #    """
    #    This method defines the mass of the rotor based on its diameter.
    #    :return:
    #    """
    #
    #    self["rotor mass"] = func_rotor_weight_rotor_diameter(
    #        self["rotor diameter"], 0.00460956, 0.11199577
    #    ) * (1 - self["offshore"])
    #
    #    self["rotor mass"] += (
    #        func_rotor_weight_rotor_diameter(
    #            self["rotor diameter"], 0.0088365, -0.16435292
    #        )
    #        * self["offshore"]
    #    )

    def __set_rotor_mass(self):
        """
        This method defines the mass of the rotor based on its diameter.
        Uses different models for onshore and offshore wind turbines.
        :return: None
        """

        # Onshore turbine mass calculation (old quadratic model)
        self["rotor mass"] = func_rotor_weight_rotor_diameter(
            self["rotor diameter"], 0.00460956, 0.11199577, 0, 0
        ) * (1 - self["offshore"])

        # Offshore turbine mass calculation (new polynomial model)
        self["rotor mass"] += (
            func_rotor_weight_rotor_diameter(
                self["rotor diameter"], -0.00008445, 0.030281, -1.8606, 37.51
            )
            * self["offshore"]
        )

    # def __set_tower_mass(self):
    #    """
    #    This method defines the mass of the tower (kg) based on the rotor diameter (m) and tower height (m).
    #    :return:
    #    """
    #
    #    self["tower mass"] = func_tower_weight_d2h(
    #        self["rotor diameter"], self["tower height"], 3.03584782e-04, 9.68652909e00
    #    )

    def __set_tower_mass(self):
        """
        This method defines the mass of the tower (kg) based on tower height (m).
        """

        ### **Onshore Turbines (Quadratic Model)**
        coeff_a_onshore = 3.03584782e-04  # Keep the existing onshore coefficient
        coeff_b_onshore = 9.68652909e00  # Keep the existing onshore coefficient

        # Compute onshore tower mass using the quadratic model
        self["tower mass"] = func_tower_weight_d2h(
            self["rotor diameter"],
            self["tower height"],
            coeff_a_onshore,
            coeff_b_onshore,
        ) * (
            1 - self["offshore"]
        )  # Apply only for onshore turbines

        ### **Offshore Turbines (Logarithmic Model)**
        # Replace these with actual optimized coefficients from your curve fitting
        coeff_a_offshore = 1287.34  # Example: Replace with actual value
        coeff_b_offshore = 10.78  # Example: Replace with actual value
        coeff_c_offshore = 1.27  # Example: Replace with actual value

        # **Fixed Min Values (Ensure Consistency with Dataset)**
        min_height = 31.00  # Fixed minimum tower height (m)
        min_mass = 13.50 * 1000  # Fixed minimum tower mass (converted to kg)

        # Compute offshore tower mass using the logarithmic model
        self["tower mass"] += (
            func_tower_weight_log(
                self["tower height"],
                coeff_a_offshore,
                coeff_b_offshore,
                coeff_c_offshore,
                min_height,
                min_mass,
            )
            * self["offshore"]
        )  # Apply only for offshore turbines

    def __set_electronics_mass(self):
        """
        Define mass of electronics based on rated power output (kW)
        :return:
        """
        self["electronics mass"] = np.interp(
            self["power"], [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946]
        )

    # this is another comment

    def __set_foundation_mass(self):
        """
        Define mass of foundation.
        For onshore turbines, this consists of concrete and reinforcing steel.
        For offshore turbines, this consists of anti-scour materials at the sea bottom,
        the steel pile, the grout, the transition piece (incl. the platform) as well as the cables.
        :return:
        """

        if "onshore" in self.array.coords["application"].values:
            self.func_mass_foundation_onshore()

        self["reinforcing steel in foundation mass"] = func_mass_reinf_steel_onshore(
            self["power"]
        ) * (1 - self["offshore"])

        self["concrete in foundation mass"] = (
            self["foundation mass"] - self["reinforcing steel in foundation mass"]
        ) * (1 - self["offshore"])

        self["pile height"] = (
            get_pile_height(self["power"], self["sea depth"]) * self["offshore"]
        )
        self["pile mass"] = (
            get_pile_mass(self["power"], self["pile height"]) * self["offshore"]
        )
        self["transition length"] = (
            get_transition_height()(self["pile height"]) * self["offshore"]
        )
        self["transition mass"] = (
            get_transition_mass(self["transition length"]) * self["offshore"]
        )
        self["grout volume"] = (
            get_grout_volume(self["transition length"]) * self["offshore"]
        )
        self["scour volume"] = get_scour_volume(self["power"]) * self["offshore"]

        self["foundation mass"] += (self["pile mass"] + self["transition mass"]) * self[
            "offshore"
        ]

        if self.location:
            self["distance to coastline"] = (
                find_nearest_coastline(self.location[0], self.location[1]) / 1000
            )

        cable_mass, energy = set_offshore_cable_requirements(
            self["power"],
            self["offshore farm cable cross-section"],
            self["distance to transformer"],
            self["distance to coastline"],
            self["turbines per farm"],
        )
        self["cable mass"] = cable_mass * self["offshore"]
        self["energy for cable lay-up"] = energy * self["offshore"]

        self["cable mass"] += set_onshore_cable_requirements(
            self["power"], self["tower height"]
        ) * (1 - self["offshore"])

    def __set_assembly_requirements(self):
        """
        Assembly requirements: components supply, electricity
        :return:
        """

        # 0.5 kWh per kg of wind turbine
        self["assembly electricity"] = (
            self[["rotor mass", "nacelle mass", "tower mass", "electronics mass"]].sum(
                dim="parameter"
            )
        ) * 0.5

        # transport to assembly plant
        self["transport to assembly"] = (
            (
                self[
                    ["rotor mass", "nacelle mass", "tower mass", "electronics mass"]
                ].sum(dim="parameter")
            )
            / 1000  # kg to tons
            * self["distance to assembly plant"]
        )

    def __set_installation_requirements(self):
        """
        Amount of transport demand for installation.
        And fuel use for installation.
        :return:
        """

        # 1 liter diesel (0.85 kg, 37 MJ) per kilowatt of power
        # assumed burned in a "building machine"
        self["installation energy"] = 37 * self["power"] * (1 - self["offshore"])

        # 46 liters diesel (46.5 kg, 1'680 MJ) per kilowatt of power
        # assumed burned in a "building machine"
        self["installation energy"] += 1680 * self["power"] * self["offshore"]

        # transport to installation site
        # tons over km
        self["installation transport, by truck"] = (
            self["nacelle transport to site"]
            * self["share nacelle transport by truck"]
            * self["nacelle mass"]
            / 1000
        )
        self["installation transport, by truck"] += (
            self["rotor transport to site"]
            * self["share rotor transport by truck"]
            * self["rotor mass"]
            / 1000
        )
        self["installation transport, by truck"] += (
            self["tower transport to site"]
            * self["share tower transport by truck"]
            * self["tower mass"]
            / 1000
        )
        self["installation transport, by truck"] += (
            self["foundation transport to site"]
            * self["share foundation transport by truck"]
            * self["foundation mass"]
            / 1000
        )

        self["installation transport, by rail"] = (
            self["nacelle transport to site"]
            * self["share nacelle transport by rail"]
            * self["nacelle mass"]
            / 1000
        )
        self["installation transport, by rail"] += (
            self["rotor transport to site"]
            * self["share rotor transport by rail"]
            * self["rotor mass"]
            / 1000
        )
        self["installation transport, by rail"] += (
            self["tower transport to site"]
            * self["share tower transport by rail"]
            * self["tower mass"]
            / 1000
        )
        self["installation transport, by rail"] += (
            self["foundation transport to site"]
            * self["share foundation transport by rail"]
            * self["foundation mass"]
            / 1000
        )

        self["installation transport, by sea"] = (
            self["nacelle transport to site"]
            * self["share nacelle transport by sea"]
            * self["nacelle mass"]
            / 1000
        )
        self["installation transport, by sea"] += (
            self["rotor transport to site"]
            * self["share rotor transport by sea"]
            * self["rotor mass"]
            / 1000
        )
        self["installation transport, by sea"] += (
            self["tower transport to site"]
            * self["share tower transport by sea"]
            * self["tower mass"]
            / 1000
        )
        self["installation transport, by sea"] += (
            self["foundation transport to site"]
            * self["share foundation transport by sea"]
            * self["foundation mass"]
            / 1000
        )

        self["installation transport, by sea"] += (
            self["distance to coastline"] * self["total mass"] / 1000  # kg/ton
        ) * self["offshore"]

    def __set_maintenance_energy(self):
        """
        An amount of transport per wind turbine per year is given.
        :return:
        """

        self["maintenance transport"] = (500 * 100 / 8) * (1 - self["offshore"])

        # 7'500 liters (7'575 kg) heavy fuel oil per turbine per year
        # assumed equivalent to 257'000 ton-km
        # by a ferry boat @ 2.95 kg/100 ton-km
        self["maintenance transport"] += (7575 * 100 / 2.95) * self["offshore"]

    def func_mass_foundation_onshore(self) -> None:
        """
        Calculates and sets the total mass of onshore wind turbine foundations.

        The function estimates the total foundation mass based on the wind turbine's
        ultimate limit state (ULS), which considers forces from wind loading and gravity.
        It calculates the mass of bolts, reinforcing steel, and concrete, and stores
        only the total mass in `self["foundation mass"]`.

        :return: None. The total foundation mass is stored in `self["foundation mass"]`.
        """

        # Compute the ultimate limit state (ULS) moment
        uls = self.__get_ultimate_limit_state()

        # Calculate masses based on ULS
        bolt_mass = (0.04009378 * uls) + 1.23107196  # in tons
        bolt_mass *= 1000  # Convert to kg

        reinf_mass = (0.63267732 * uls) - 9.30963858  # in tons
        reinf_mass *= 1000  # Convert to kg

        # Calculate concrete volume and mass
        concrete_vol = (3.23575233 * uls) + 203.0179  # in cubic meters

        # Store only the total foundation mass in `self["foundation mass"]`
        self["foundation volume concrete"] = concrete_vol  # in m3
        self["foundation mass steel"] = reinf_mass + bolt_mass  # kg
        self["foundation mass"] = reinf_mass + bolt_mass + (concrete_vol * 2400)  # kg

    def __get_ultimate_limit_state(self):
        """
        Calculates the Ultimate Limit State (ULS) moment for the wind turbine foundation.

        The function estimates the ULS moment by computing the forces acting on the wind turbine
        due to wind loading and gravitational forces. The wind force is based on the drag coefficient,
        air density, and rotor swept area, while the gravitational force is based on the nacelle and rotor mass.

        :return: The ultimate limit state (ULS) moment in MN·m (Meganewton-meters).
        :rtype: float
        """
        # Given values
        Cd = 1.2  # Drag coefficient
        # Air density (kg/m³)
        try:
            rho = self.terrain_vars["RHO"]
        except TypeError:
            rho = 1.225

        D = self["rotor diameter"]  # Rotor diameter (m)
        A = (np.pi / 4) * D**2  # Rotor swept area (m²)

        # Maximum wind force calculation
        try:
            max_wind_speed = self.terrain_vars["WS"].max()
        except TypeError:
            max_wind_speed = 13  # m/s, if not available
        F_wind = 0.5 * Cd * rho * A * max_wind_speed**2

        # Gravity force calculation
        mass_nacelle_rotor = self["nacelle mass"]  # kg (100 t)
        g = 9.81  # Gravity (m/s²)
        F_gravity = mass_nacelle_rotor * g

        # Heights
        H_hub = 100 + D / 2  # Hub height (m)
        H_CoM = 100 + D / 3  # Approximate center of mass height (m)

        # ULS Moment calculation
        M_ULS = (F_wind * H_hub) + (F_gravity * H_CoM)

        # Convert to MN·m
        M_ULS_MN = M_ULS / 1e6

        return M_ULS_MN

    def disable_unavailable_models(self):
        # disable offshore wind turbines with a rated power output inferior to 1'000 kW
        if "offshore" in self.array.coords["application"]:
            self.array.loc[
                dict(
                    application="offshore",
                    parameter="power",
                    size=[s for s in self.array.coords["size"] if s < 1000],
                )
            ] = 0
