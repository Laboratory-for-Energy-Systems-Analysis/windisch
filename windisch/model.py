"""
model.py contains `WindTurbineModel` which sizes wind turbines
and calculates dimensions and mass attributes.
"""

from typing import List, Tuple, Union

import numpy as np

# material densities, in kg/m3
COPPER_DENSITY = 8960
STEEL_DENSITY = 8000


def func_height_power(
    power: int, coeff_a: float, coeff_b: float, coeff_c: float
) -> float:
    """
    Returns hub height, in m, based on rated power output (kW).
    :param power: power output (kW)
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :param coeff_c: coefficient
    :return: hub height (m)
    """
    return coeff_a - coeff_b * np.exp(-(power) / coeff_c)


def func_rotor_weight_rotor_diameter(
    power: int, coeff_a: float, coeff_b: float
) -> float:
    """
    Returns rotor weight, in kg, based on rotor diameter.
    :param power: power output (kW)
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :return: nacelle weight (in kg)
    """
    rotor_mass = coeff_a * power ** 2 + coeff_b * power
    return 1e3 * rotor_mass


def func_nacelle_weight_power(power: int, coeff_a: float, coeff_b: float) -> float:
    """
    Returns nacelle weight, in kg.
    :param power: power output (kW)
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :return: nacelle weight (in kg)
    """
    nacelle_mass = coeff_a * power ** 2 + coeff_b * power
    return 1e3 * nacelle_mass


def func_rotor_diameter(
    power: int, coeff_a: float, coeff_b: float, coeff_c: float, coeff_d: float
) -> float:
    """
    Returns rotor diameter, based on power output and given coefficients
    :param power: power output (kW)
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :param coeff_c: coefficient
    :param coeff_d: coefficient
    :return: rotor diameter (m)
    """
    return coeff_a - coeff_b * np.exp(-(power - coeff_d) / coeff_c)


def func_mass_foundation_onshore(height: float, diameter: float) -> float:
    """
    Returns mass of onshore turbine foundations
    :param height: tower height (m)
    :param diameter: rotor diameter (m)
    :return:
    """
    return 1696e3 * height / 80 * diameter ** 2 / (100 ** 2)


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
    :return: pile height (m)
    """
    fit_penetration_depth = penetration_depth_fit()
    return 9 + fit_penetration_depth(power) + sea_depth


def get_pile_mass(power: int, pile_height: float) -> float:
    """
    Return the mass of the steel pile
    :param power: power output (in kW) of the rotor
    :param pile_height: height (in m) of the pile
    :return: mass of the steel pile
    """
    # diameters, in meters
    diameter_x = [5, 5.5, 5.75, 6.75, 7.75]
    # kW
    power_y = [3000, 3600, 4000, 8000, 10000]
    fit_diameter = np.polyfit(power_y, diameter_x, 1)
    f_fit_diameter = np.poly1d(fit_diameter)

    # diameter for given power, in m
    outer_diameter = f_fit_diameter(power)
    # Cross section area of pile
    outer_area = (np.pi / 4) * (outer_diameter ** 2)
    # Pile volume, in m3
    outer_volume = outer_area * pile_height

    inner_diameter = outer_diameter
    pile_thickness = np.interp(
        power,
        [2000, 3000, 3600, 4000, 8000, 10000],
        [0.07, 0.10, 0.13, 0.16, 0.19, 0.22],
    )
    inner_diameter -= 2 * pile_thickness
    inner_area = (np.pi / 4) * (inner_diameter ** 2)
    inner_volume = inner_area * pile_height
    volume_steel = outer_volume - inner_volume
    weight_steel = STEEL_DENSITY * volume_steel
    return weight_steel


def get_transition_height() -> np.poly1d:
    """
    Returns a fitting model for the height of transition piece (in m), based on pile height (in m).
    :return:
    """
    pile_length = [35, 55, 35, 60, 40, 65, 50, 70, 50, 80]
    transition_length = [15, 20, 15, 20, 15, 24, 20, 30, 20, 31]
    fit_transition_length = np.polyfit(pile_length, transition_length, 1)
    return np.poly1d(fit_transition_length)


def get_transition_mass(pile_height: float) -> float:
    """
    Returns the mass of transition piece (in kg).
    :return:
    """
    transition_length = [15, 20, 15, 20, 15, 24, 20, 30, 20, 31]
    transition_weight = [150, 250, 150, 250, 160, 260, 200, 370, 250, 420]
    fit_transition_weight = np.polyfit(transition_length, transition_weight, 1)

    trans_height = get_transition_height()

    return np.poly1d(fit_transition_weight)(trans_height(pile_height)) * 1000


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
    :param coeff_a: coefficient
    :param coeff_b: coefficient
    :return: tower mass (in kg)
    """
    tower_mass = coeff_a * diameter ** 2 * height + coeff_b
    return 1e3 * tower_mass


def set_cable_requirements(
    power: int,
    cross_section: float,
    dist_transfo: float,
    dist_coast: float,
    park_size: int,
) -> Tuple[float, float]:
    """
    Return the required cable mass as well as the energy needed to lay down the cable.
    :param power: rotor power output (in kW)
    :param cross_section: cable cross section (in mm2)
    :param dist_transfo: distance to transformer (in m)
    :param dist_coast: distance to coastline (in m)
    :param park_size:
    :return:
    """

    m_copper = (cross_section * 1e-6 * (dist_transfo * 1e3)) * COPPER_DENSITY

    # 450 l diesel/hour for the ship that lays the cable at sea bottom
    # 39 MJ/liter, 15 km/h as speed of laying the cable
    energy_cable_laying_ship = 450 * 39 / 15 * dist_transfo

    # Cross section calculated based on the farm cumulated power,
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

    m_cable = m_copper * 617 / 220
    # FIXME: ask why * 0.5
    return m_cable * 0.5, energy_cable_laying_ship * 0.5


class WindTurbineModel:

    """
    This class represents the entirety of the turbines considered,
    with useful attributes, such as an array that stores
    all the turbine input parameters.

    :ivar array: multi-dimensional numpy-like array that contains parameters' value(s)
    :vartype array: xarray.DataArray

    """

    def __init__(self, array):
        self.__cache = None
        self.array = array

    def __call__(self, key):
        """
        This method fixes a dimension of the `array` attribute
        given an `application` technology selected.

        Set up this class as a context manager, so we can have some nice syntax

        .. code-block:: python

            with class('some powertrain') as cpm:
                cpm['something']. # Will be filtered for the correct powertrain

        On with block exit, this filter is cleared
        https://stackoverflow.com/a/10252925/164864

        :param key: A powertrain type, e.g., "FCEV"
        :type key: str
        :return: An instance of `array` filtered after the powertrain selected.

        """
        self.__cache = self.array
        self.array = self.array.sel(application=key)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.array = self.__cache
        del self.__cache

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

            class['key', 'value]

        :param key: Parameter name
        :param value: Numeric value (int or float)
        :return: Nothing. Modifies in place.
        """
        self.array.loc[{"parameter": key}] = value

    def set_all(self):
        """
        This method runs a series of methods to size the wind turbines, evaluate material requirements, etc.

        :returns: Does not return anything. Modifies ``self.array`` in place.

        """

        self.__set_size_rotor()
        self.__set_tower_height()
        self.__set_nacelle_mass()
        self.__set_rotor_mass()
        self.__set_tower_mass()
        self.__set_electronics_mass()
        self.__set_foundation_mass()

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

        # we remove wind turbines that are unlikely to exist

        self.array.loc[
            dict(
                size=[
                    s
                    for s in self.array.coords["size"].values
                    if s in ["100kW", "500kW"]
                ],
                application="offshore",
            )
        ] = 0

    def __set_size_rotor(self):
        """
        This method defines the rotor diameter, based on its power output.
        :return:
        """

        if "onshore" in self.array.application:
            with self("onshore") as onshore:
                power = onshore["rated power"]
                onshore["rotor diameter"] = func_rotor_diameter(
                    power, 152.66222073, 136.56772435, 2478.03511414, 16.44042379
                )

        if "offshore" in self.array.application:
            with self("offshore") as offshore:
                power = offshore["rated power"]
                offshore["rotor diameter"] = func_rotor_diameter(
                    power, 191.83651588, 147.37205671, 5101.28555377, 376.62814798
                )

    def __set_tower_height(self):
        """
        This method defines the hub height based on the power output.
        :return:
        """

        if "onshore" in self.array.application:
            with self("onshore") as onshore:
                power = onshore["rated power"]
                onshore["tower height"] = func_height_power(
                    power, 116.43035193, 91.64953366, 2391.88662558
                )

        if "offshore" in self.array.application:
            with self("offshore") as offshore:
                power = offshore["rated power"]
                offshore["tower height"] = func_height_power(
                    power, 120.75491612, 82.75390577, 4177.56520433
                )

    def __set_nacelle_mass(self):
        """
        This method defines the mass of the nacelle on the power output.
        :return:
        """

        if "onshore" in self.array.application:
            with self("onshore") as onshore:
                power = onshore["rated power"]
                onshore["nacelle mass"] = func_nacelle_weight_power(
                    power, 1.66691134e-06, 3.20700974e-02
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                power = off["rated power"]
                off["nacelle mass"] = func_nacelle_weight_power(
                    power, 2.15668283e-06, 3.24712680e-02
                )

    def __set_rotor_mass(self):
        """
        This method defines the mass of the rotor based on its diameter.
        :return:
        """

        if "onshore" in self.array.application:
            with self("onshore") as onshore:
                diameter = onshore["rotor diameter"]
                onshore["rotor mass"] = func_rotor_weight_rotor_diameter(
                    diameter, 0.00460956, 0.11199577
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                diameter = onshore["rotor diameter"]
                off["rotor mass"] = func_rotor_weight_rotor_diameter(
                    diameter, 0.0088365, -0.16435292
                )

    def __set_tower_mass(self):
        """
        This method defines the mass of the tower (kg) based on the rotor diameter (m) and tower height (m).
        :return:
        """

        rotor_diameter = self.array.loc[
            dict(application="onshore", parameter="rotor diameter")
        ]
        rotor_height = self.array.loc[
            dict(application="onshore", parameter="tower height")
        ]
        self.array.loc[
            dict(application="onshore", parameter="tower mass")
        ] = func_tower_weight_d2h(
            rotor_diameter, rotor_height, 3.03584782e-04, 9.68652909e00
        )

    def __set_electronics_mass(self):
        """
        Define mass of electronics based on rated power output (kW)
        :return:
        """
        self["electronics mass"] = np.interp(
            self["rated power"], [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946]
        )

    def __set_foundation_mass(self):
        """
        Define mass of foundation.
        For onshore turbines, this consists of concrete and reinforcing steel.
        For offhore turbines, this consists of anti-scour materials at the sea bottom,
        the steel pile, the grout, the transition piece (incl. the platform) as well as the cables.
        :return:
        """

        if "onshore" in self.array.application:
            with self("onshore") as onshore:
                height = onshore["tower height"]
                diameter = onshore["rotor diameter"]
                power = onshore["rated power"]
                onshore["foundation mass"] = func_mass_foundation_onshore(
                    height, diameter
                )
                onshore[
                    "reinforcing steel in foundation mass"
                ] = func_mass_reinf_steel_onshore(power)
                onshore["concrete in foundation mass"] = (
                    onshore["foundation mass"]
                    - onshore["reinforcing steel in foundation mass"]
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                power = off["rated power"]
                sea_depth = off["sea depth"]
                cross_section = off["offshore farm cable cross-section"]
                dist_transf = off["distance to transformer"]
                dist_coast = off["distance to coastline"]
                park_size = off["turbines per farm"]

                off["pile height"] = get_pile_height(power, sea_depth)
                off["pile mass"] = get_pile_mass(power, off["pile height"])
                off["transition length"] = get_transition_height()(off["pile height"])
                off["transition mass"] = get_transition_mass(off["pile height"])
                off["grout volume"] = get_grout_volume(off["transition length"])
                off["scour volume"] = get_scour_volume(off["rated power"])

                off["foundation mass"] = self[["pile mass", "transition mass",]].sum(
                    dim="parameter"
                )

                cable_mass, energy = set_cable_requirements(
                    power, cross_section, dist_transf, dist_coast, park_size,
                )
                off["cable mass"] = cable_mass
                off["energy for cable lay-up"] = energy

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
            / 100
            * self["distance to assembly plant"]
        )

    def __set_installation_requirements(self):
        """
        Amount of transport demand for installation.
        And fuel use for installation.
        :return:
        """

        if "onshore" in self.array.application:
            # 1 liter diesel (0.85 kg, 37 MJ) per kilowatt of power
            # assumed burned in a "building machine"

            with self("onshore") as onshore:
                onshore["installation energy"] = 37 * onshore["rated power"]

        if "offshore" in self.array.application:
            # 46 liters diesel (46.5 kg, 1'680 MJ) per kilowatt of power
            # assumed burned in a "building machine"
            with self("offshore") as offshore:
                offshore["installation energy"] = 1680 * offshore["rated power"]

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

        if "offshore" in self.array.application:
            # additional transport to offshore location
            with self("offshore") as offshore:
                offshore["installation transport, by ship"] = (
                    self["distance to coastline"] * self["total mass"] / 1000
                )

        if "onshore" in self.array.application:
            # access road to onshore turbine
            with self("onshore") as onshore:
                onshore["access road"] = np.interp(
                    self["rated power"], [0, 2000], [0, 8000]
                )

    def __set_maintenance_energy(self):
        """
        An amount of transport per wind turbine per year is given.
        :return:
        """

        if "onshore" in self.array.application:
            with self("onshore") as onshore:
                onshore["maintenance transport"] = 500 * 100 / 8

        if "offshore" in self.array.application:
            # 7'500 liters (7'575 kg) heavy fuel oil per turbine per year
            # assumed equivalent to 257'000 ton-km
            # by a ferry boat @ 2.95 kg/100 ton-km
            with self("offshore") as offshore:
                offshore["maintenance transport"] = 7575 * 100 / 2.95
