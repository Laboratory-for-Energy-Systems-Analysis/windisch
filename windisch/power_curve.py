import numpy as np
import pandas as pd
import xarray as xr

from . import DATA_DIR


def load_optimized_power_curve_parameters():
    """
    Load the optimized power curve parameters.
    Coordinates: power.

    :return: xr.DataArray with optimized parameters for each turbine.
    """
    file_path = DATA_DIR / "optimized_power_curve_parameters.csv"
    return pd.read_csv(file_path, index_col=0).to_xarray()


def calculate_generic_power_curve(power: xr.DataArray):
    """
    Compute the power output for a wind turbine at wind speed v based on a piecewise model.

    :param:power (float): Rated power output (kW)
    :return:Power output (kW) for the given wind speeds.
    """

    parameters = load_optimized_power_curve_parameters().interp(
        power=power,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Wind speed range, by intervals of 0.5 m/s, from 0 to 35 m/s
    # using the same shape as `power`, which is a xr.DataArray
    # and has an additional dimension `data` with the wind speed values
    power_curve = xr.DataArray(
        np.zeros_like(np.arange(0, 35.5, 0.5)),
        dims=("wind speed"),
        coords={"wind speed": np.arange(0, 35.5, 0.5)},
    )

    power_curve = power_curve.broadcast_like(power)

    # Between cut-in and rated speed - ramp-up phase
    mask_ramp = (
        power_curve.coords["wind speed"] >= parameters["cut-in wind speed"]
    ) & (power_curve.coords["wind speed"] < parameters["rated power wind speed"])

    power_curve = (
        power
        * (
            (power_curve.coords["wind speed"] - parameters["cut-in wind speed"])
            / (parameters["rated power wind speed"] - parameters["cut-in wind speed"])
        )
        ** parameters["power exponent"]
    ) * mask_ramp

    # Between rated speed and cut-out - constant power output
    mask_flat = (
        power_curve.coords["wind speed"] >= parameters["rated power wind speed"]
    ) & (power_curve.coords["wind speed"] <= parameters["cut-out wind speed"])
    mask_flat = mask_flat.transpose(*power_curve.dims)
    power_curve = xr.where(mask_flat, power, power_curve)

    return power_curve.fillna(0)
