import pandas as pd
from . import DATA_DIR

def data_to_dict(csv_list):
    """
    Returns a dictionary from a sequence of items.
    :param data: list
    :return: dict
    """

    (_, *header), *data = csv_list
    csv_dict = {}
    for row in data:
        key, *values = row
        csv_dict[key] = {key: value for key, value in zip(header, values)}

    return csv_dict


def get_electricity_losses():
    """
    Retrieve cumulative electricity losses from high to medium and low voltage.
    Source: `ecoinvent v.3.6 <https://www.ecoinvent.org/>`_.

    :returns: dictionary
    :rtype: dict

    """
    filename = "cumulative_electricity_losses.csv"
    filepath = DATA_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The CSV file that contains electricity mixes could not be found."
        )
    with open(filepath) as f:
        csv_list = [[val.strip() for val in r.split(";")] for r in f.readlines()]

    return data_to_dict(csv_list)


def get_region_mapping():
    """
    Retrieve mapping between ISO country codes and REMIND regions.

    :returns: dictionary
    :rtype: dict

    """
    filename = "region_mapping.csv"
    filepath = DATA_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The CSV file that contains correspondences between REMIND region names and ISO country codes "
            "could not be found."
        )
    with open(filepath) as f:
        csv_list = [[val.strip() for val in r.split(";")] for r in f.readlines()]

    return data_to_dict(csv_list)


def get_electricity_mix():
    """
    Retrieve electricity mixes and shape them into an xarray.
    Source:
        * for European countries (`EU Reference Scenario 2016 <https://ec.europa.eu/energy/en/data-analysis/energy-modelling/eu-reference-scenario-2016>`_),
        * for African countries (`TEMBA <http://www.osemosys.org/temba.html>`_ model)
        * and for other countries (`IEA World Energy outlook 2017 <https://www.iea.org/reports/world-energy-outlook-2017>`_)

    :returns: An axarray with 'country' and 'year' as dimensions
    :rtype: xarray.core.dataarray.DataArray

    """
    filename = "electricity_mixes.csv"
    filepath = DATA_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The CSV file that contains electricity mixes could not be found."
        )

    df = pd.read_csv(filepath, sep=";", index_col=["country", "year"])
    df = df.reset_index()

    array = (
        df.melt(id_vars=["country", "year"], value_name="value")
        .groupby(["country", "year", "variable"])["value"]
        .mean()
        .to_xarray()
    )
    array = array.interpolate_na(
        dim="year", method="linear", fill_value="extrapolate"
    ).clip(0, 1)
    array /= array.sum(axis=2)

    return array


class BackgroundSystemModel:
    """
    Retrieve and build dictionaries that contain important information to model in the background system:

        * gross electricity production mixes from nearly all countries in the world, from 2015 to 2050.
        * cumulative electricity transformation/transmission/distribution losses from high voltage to medium and low voltage.
        * share of biomass-derived fuel in the total consumption of liquid fuel in the transport sector. Source: REMIND.
    """

    def __init__(self):
        self.electricity_mix = get_electricity_mix()
        self.losses = get_electricity_losses()
        self.region_map = get_region_mapping()
