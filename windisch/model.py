import numpy as np


def finite(array, mask_value=0):
    return np.where(np.isfinite(array), array, mask_value)


class WindTurbineModel:

    """
    This class represents the entirety of the turbines considered, with useful attributes, such as an array that stores
    all the turbine input parameters.

    :ivar array: multi-dimensional numpy-like array that contains parameters' value(s)
    :vartype array: xarray.DataArray

    """

    def __init__(self, array):
        self.array = array

    def __call__(self, key):
        """
        This method fixes a dimension of the `array` attribute given an `application` technology selected.

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

    def __getitem__(self, key):
        """
        Make class['foo'] automatically filter for the parameter 'foo'
        Makes the model code much cleaner

        :param key: Parameter name
        :type key: str
        :return: `array` filtered after the parameter selected
        """

        return self.array.sel(parameter=key)

    def __setitem__(self, key, value):
        self.array.loc[{"parameter": key}] = value

    def set_all(self):
        """
        This method runs a series of other methods to size the wind turbines, evaluate material requirements, etc.

        :returns: Does not return anything. Modifies ``self.array`` in place.

        """

        self.set_size_rotor()
        self.set_tower_height()
        self.set_nacelle_mass()
        self.set_rotor_mass()
        self.set_tower_mass()
        self.set_electronics_mass()
        self.set_foundation_mass()

        self["total mass"] = self[
            [
                "rotor mass",
                "nacelle mass",
                "tower mass",
                "electronics mass",
                "cable mass",
                "pile mass",
                "reinforcing steel mass",
                "transition mass",
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

    def set_size_rotor(self):
        """
        This method defines the rotor diameter, based on its power output.
        :return:
        """

        def func_rotor_power(x, a, b, c, d):
            """
            Returns rotor diameter, based on power output and given coefficients
            :param x: power output (kW)
            :param a: coefficient
            :param b: coefficient
            :param c: coefficient
            :param d: coefficient
            :return: rotor diameter (m)
            """
            return a - b * np.exp(-(x - d) / c)

        if "onshore" in self.array.application:
            with self("onshore") as on:
                p = on["rated power"]
                on["rotor diameter"] = func_rotor_power(
                    p, 152.66222073, 136.56772435, 2478.03511414, 16.44042379
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                p = off["rated power"]
                off["rotor diameter"] = func_rotor_power(
                    p, 191.83651588, 147.37205671, 5101.28555377, 376.62814798
                )

    def set_tower_height(self):
        """
        This method defines the hub height based on the power output.
        :return:
        """

        def func_height_power(x, a, b, c):
            """
            Returns hub height, in m, based on rated power output (kW).
            :param x: power output (kW)
            :param a: coefficient
            :param b: coefficient
            :param c: coefficient
            :return: hub height (m)
            """
            return a - b * np.exp(-(x) / c)

        if "onshore" in self.array.application:
            with self("onshore") as on:
                p = on["rated power"]
                on["tower height"] = func_height_power(
                    p, 116.43035193, 91.64953366, 2391.88662558
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                p = off["rated power"]
                off["tower height"] = func_height_power(
                    p, 120.75491612, 82.75390577, 4177.56520433
                )

    def set_nacelle_mass(self):
        """
        This method defines the mass of the nacelle on the power output.
        :return:
        """

        def func_nacelle_weight_power(x, a, b):
            """
            Returns nacelle weight, in m.
            :param x: power output (kW)
            :param a: coefficient
            :param b: coefficient
            :return: nacelle weight (in kg)
            """
            y = a * x ** 2 + b * x
            return 1e3 * y

        if "onshore" in self.array.application:
            with self("onshore") as on:
                p = on["rated power"]
                on["nacelle mass"] = func_nacelle_weight_power(
                    p, 1.66691134e-06, 3.20700974e-02
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                p = off["rated power"]
                off["nacelle mass"] = func_nacelle_weight_power(
                    p, 2.15668283e-06, 3.24712680e-02
                )

    def set_rotor_mass(self):
        """
        This method defines the mass of the rotor based on its diameter.
        :return:
        """

        def func_rotor_weight_rotor_diameter(x, a, b):
            """
            Returns rotor weight, in m, based on rotor diameter.
            :param x: power output (kW)
            :param a: coefficient
            :param b: coefficient
            :return: nacelle weight (in kg)
            """
            y = a * x ** 2 + b * x
            return 1e3 * y

        if "onshore" in self.array.application:
            with self("onshore") as on:
                d = on["rotor diameter"]
                on["rotor mass"] = func_rotor_weight_rotor_diameter(
                    d, 0.00460956, 0.11199577
                )

        if "offshore" in self.array.application:
            with self("offshore") as off:
                d = on["rotor diameter"]
                off["rotor mass"] = func_rotor_weight_rotor_diameter(
                    d, 0.0088365, -0.16435292
                )

    def set_tower_mass(self):
        """
        This method defines the mass of the tower (kg) based on the rotor diameter (m) and tower height (m).
        :return:
        """

        def func_tower_weight_d2h(d, h, a, b):
            """
            Returns tower mass, in kg, based on tower diameter and height.
            :param d: tower diameter (m)
            :param h: tower height (m)
            :param a: coefficient
            :param b: coefficient
            :return: tower mass (in kg)
            """
            y = a * d ** 2 * h + b
            return 1e3 * y

        d = self.array.loc[dict(application="onshore", parameter="rotor diameter")]
        h = self.array.loc[dict(application="onshore", parameter="tower height")]
        self.array.loc[
            dict(application="onshore", parameter="tower mass")
        ] = func_tower_weight_d2h(d, h, 3.03584782e-04, 9.68652909e00)

    def set_electronics_mass(self):
        """
        Define mass of electronics based on rated power output (kW
        :return:
        """
        self["electronics mass"] = np.interp(
            self["rated power"], [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946]
        )

    def set_foundation_mass(self):
        def func_mass_foundation_onshore(h, d):
            """
            Returns mass of onshore turbine foundations
            :param h: tower height (m)
            :param d: rotor diameter (m)
            :return:
            """
            return 1696e3 * h / 80 * d ** 2 / (100 ** 2)

        def func_mass_reinf_steel_onshore(p):
            """
            Returns mass of reinforcing steel in onshore turbine foundations, based on power output (kW).
            :param p: power output (kW)
            :return:
            """
            return np.interp(p, [750, 2000, 4500], [10210, 27000, 51900])

        if "onshore" in self.array.application:
            with self("onshore") as on:
                h = on["tower height"]
                d = on["rotor diameter"]
                p = on["rated power"]
                on["foundation mass"] = func_mass_foundation_onshore(h, d)
                on["reinforcing steel mass"] = func_mass_reinf_steel_onshore(p)

        def penetration_depth():
            # meters
            depth = [22.5, 22.5, 23.5, 26, 29.5]
            # kW
            P = [3000, 3600, 4000, 8000, 10000]
            fit_penetration = np.polyfit(P, depth, 1)
            f_fit_penetration = np.poly1d(fit_penetration)
            return f_fit_penetration

        def get_pile_height(p, sea_depth):
            """
            Returns undersea pile height (m) from rated power output (kW), penetration depth and sea depeth.
            :param p: power output (kW)
            :return: pile height (m)
            """
            fit_penetration_depth = penetration_depth()
            return 9 + fit_penetration_depth(p) + sea_depth

        def get_pile_mass(p, pile_height):
            # diameters, in meters
            diameter = [5, 5.5, 5.75, 6.75, 7.75]
            # kW
            power = [3000, 3600, 4000, 8000, 10000]
            fit_diameter = np.polyfit(power, diameter, 1)
            f_fit_diameter = np.poly1d(fit_diameter)

            # diameter for given power, in m
            outer_diameter = f_fit_diameter(p)
            # Cross section area of pile
            outer_area = (np.pi / 4) * (outer_diameter ** 2)
            # Pile volume, in m3
            outer_volume = outer_area * pile_height

            inner_diameter = outer_diameter
            pile_thickness = np.interp(
                p,
                [2000, 3000, 3600, 4000, 8000, 10000],
                [0.07, 0.10, 0.13, 0.16, 0.19, 0.22],
            )
            inner_diameter -= 2 * pile_thickness
            inner_area = (np.pi / 4) * (inner_diameter ** 2)
            inner_volume = inner_area * pile_height
            volume_steel = outer_volume - inner_volume
            weight_steel = 8000 * volume_steel
            return weight_steel

        def get_transition_height():
            """
            Returns height of transition piece (m), based on pile height (m).
            :return:
            """
            pile_length = [35, 55, 35, 60, 40, 65, 50, 70, 50, 80]
            transition_length = [15, 20, 15, 20, 15, 24, 20, 30, 20, 31]
            fit_transition_length = np.polyfit(pile_length, transition_length, 1)
            return np.poly1d(fit_transition_length)

        def get_transition_mass(pile_height):
            """
            Returns mass of transition piece (kg).
            :return:
            """
            transition_length = [15, 20, 15, 20, 15, 24, 20, 30, 20, 31]
            transition_weight = [150, 250, 150, 250, 160, 260, 200, 370, 250, 420]
            fit_transition_weight = np.polyfit(transition_length, transition_weight, 1)

            trans_height = get_transition_height()

            return np.poly1d(fit_transition_weight)(trans_height(pile_height)) * 1000

        def get_grout_volume(trans_length):
            """
            Returns grout volume (m3) based on tranistion piece length (m).
            :param trans_length:
            :return:
            """
            transition_length = [15, 20, 15, 20, 15, 24, 20, 30, 20, 31]
            grout = [15, 35, 15, 35, 20, 40, 25, 60, 30, 65]
            fit_grout = np.polyfit(transition_length, grout, 1)
            return np.poly1d(fit_grout)(trans_length)

        def get_scour_volume(p):
            """
            Returns scour volume (m3) based on power output (kW).
            Scour is a mix of gravel and cement.
            :param p: power output (kW)
            :return: scour mass (kg)
            """
            scour = [2200, 2200, 2600, 3100, 3600]
            turbine_power = [3000, 3600, 4000, 8000, 10000]
            fit_scour = np.polyfit(turbine_power, scour, 1)
            return np.poly1d(fit_scour)(p)

        def set_cable_requirements(
            p, cross_section, dist_transfo, dist_coast, copper_density, park_size
        ):

            m_copper = (cross_section * 1e-6 * (dist_transfo * 1e3)) * copper_density

            # 450 l diesel/hour for the ship that lays the cable at sea bottom
            # 39 MJ/liter, 15 km/h as speed of laying the cable
            energy_cable_laying_ship = 450 * 39 / 15 * dist_transfo

            # Cross section calculated based on the farm cumulated power,
            # and the transport capacity of the Nexans cables @ 150kV
            # if the cumulated power of the park cannot be transported @ 33kV

            # Test if the cumulated power of the wind farm is inferior to 30 MW,
            # If so, we use 33 kV cables.

            cross_section_ = np.zeros_like(p * park_size)

            cross_section_ = np.where(
                p * park_size <= 30e3,
                np.interp(
                    p * park_size,
                    np.array([352, 399, 446, 502, 581, 652, 726, 811, 904, 993]) * 33,
                    np.array([95, 120, 150, 185, 240, 300, 400, 500, 630, 800]),
                ),
                np.interp(
                    p * park_size,
                    np.array([710, 815, 925, 1045, 1160, 1335, 1425, 1560]) * 150,
                    np.array([400, 500, 630, 800, 1000, 1200, 1600, 2000]),
                ),
            )

            m_copper += (
                cross_section_ * 1e-6 * (dist_coast * 1e3 / park_size)
            ) * copper_density

            # 450 l diesel/hour for the ship that lays the cable at sea bottom
            # 39 MJ/liter, 15 km/h as speed of laying the cable
            energy_cable_laying_ship += 450 * 39 / 15 * dist_coast / park_size

            m_cable = m_copper * 617 / 220
            # TODO: ask why * 0.5
            return m_cable * 0.5, energy_cable_laying_ship * 0.5

        if "offshore" in self.array.application:
            with self("offshore") as off:
                p = off["rated power"]
                sea_depth = off["sea depth"]
                cross_section = off["offshore farm cable cross-section"]
                dist_transf = off["distance to transformer"]
                dist_coast = off["distance to coastline"]
                copper_density = off["copper density"]
                park_size = off["turbines per farm"]

                off["pile height"] = get_pile_height(p, sea_depth)
                off["pile mass"] = get_pile_mass(p, off["pile height"])
                off["transition length"] = get_transition_height()(off["pile height"])
                off["transition mass"] = get_transition_mass(off["pile height"])
                off["grout volume"] = get_grout_volume(off["transition length"])
                off["scour volume"] = get_scour_volume(off["rated power"])

                cable_mass, energy = set_cable_requirements(
                    p, cross_section, dist_transf, dist_coast, copper_density, park_size
                )
                off["cable mass"] = cable_mass
                off["energy for cable lay-up"] = energy
