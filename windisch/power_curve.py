"""
Wind Turbine Power Curve Module

This module contains functions to calculate wind turbine power curves using various models
for power coefficient (Cp) and includes environmental effects such as turbulence intensity (TI).

Functions:
    - calculate_cp: Computes the power coefficient (Cp) for given tip-speed ratio (TSR) and pitch angle (Beta).
    - calculate_raw_power_curve: Calculates the raw power curve of a wind turbine without environmental effects.
    - apply_turbulence_effect: Adjusts the power curve based on turbulence intensity (TI).
    - calculate_rews: Calculates the rotor equivalent wind speed (REWS) accounting for wind shear and veer effects.
    - calculate_generic_power_curve: Computes the complete power curve of a wind turbine considering environmental factors.
"""

from typing import List, Union

import numpy as np
import xarray as xr


def calculate_cp(
    model: str, tsr: np.ndarray, beta: Union[np.ndarray, List[float]] = []
) -> np.ndarray:
    """
    Calculate the power coefficient (Cp) based on the tip-speed ratio (TSR) and pitch angle (Beta).

    :param model: The model to use for Cp calculation.
    :type model: str
    :param tsr: Tip-speed ratio values.
    :type tsr: np.ndarray
    :param beta: Pitch angle values, defaults to zero if not provided.
    :type beta: Union[np.ndarray, List[float]], optional
    :return: Power coefficient values.
    :rtype: np.ndarray
    """
    tsr = np.maximum(0.0001, tsr)
    if isinstance(beta, list) and not beta:
        beta = np.zeros_like(tsr)

    if model == "constant":
        return np.ones_like(tsr) * 0.49

    model_parameters = {
        "Slootweg et al. 2003": (
            0.73,
            151,
            0.58,
            0,
            0.002,
            13.2,
            18.4,
            0,
            -0.02,
            0.003,
            2.14,
        ),
        "Heier 2009": (0.5, 116, 0.4, 0, 0, 5, 21, 0, 0.089, 0.035, 0),
        "Thongam et al. 2009": (
            0.5176,
            116,
            0.4,
            0,
            0,
            5,
            21,
            0.006795,
            0.089,
            0.035,
            0,
        ),
        "De Kooning et al. 2010": (0.77, 151, 0, 0, 0, 13.65, 18.4, 0, 0, 0, 0),
        "Ochieng et Manyonge 2014": (0.5, 116, 0, 0.4, 0, 5, 21, 0, 0.08, 0.035, 0),
        "Dai et al. 2016": (0.22, 120, 0.4, 0, 0, 5, 12.5, 0, 0.08, 0.035, 0),
    }

    if model not in model_parameters:
        raise ValueError(f"Model '{model}' is not recognized.")

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, x = model_parameters[model]
    li = 1 / (1 / (tsr + c9 * beta) - c10 / (beta**3 + 1))
    cp = np.maximum(
        0,
        c1
        * (c2 / li - c3 * beta - c4 * li * beta - c5 * beta**x - c6)
        * np.exp(-c7 / li)
        + c8 * tsr,
    )
    return cp


def calculate_raw_power_curve(
    vws: np.ndarray,
    p_nom: float,
    d_rotor: float,
    r_min: Union[float, List[float]] = [],
    r_max: Union[float, List[float]] = [],
    cp_max: Union[float, List[float]] = [],
    model: str = "Dai et al. 2016",
    beta: Union[np.ndarray, List[float]] = [],
    air_density: float = 1.225,
    conv_eff: float = 0.92,
) -> np.ndarray:
    """
    Calculate the raw wind turbine power curve without considering environmental effects.

    :param vws: Wind speed values (m/s).
    :type vws: np.ndarray
    :param p_nom: Nominal power of the wind turbine (kW).
    :type p_nom: float
    :param d_rotor: Rotor diameter (m).
    :type d_rotor: float
    :param r_min: Minimum rotor speed, defaults to calculated value.
    :type r_min: Union[float, List[float]], optional
    :param r_max: Maximum rotor speed, defaults to calculated value.
    :type r_max: Union[float, List[float]], optional
    :param cp_max: Maximum power coefficient, defaults to calculated value.
    :type cp_max: Union[float, List[float]], optional
    :param model: Cp model, defaults to "Dai et al. 2016".
    :type model: str
    :param beta: Pitch angle, defaults to zero.
    :type beta: Union[np.ndarray, List[float]], optional
    :param air_density: Air density (kg/m^3), defaults to 1.225.
    :type air_density: float
    :param conv_eff: Conversion efficiency, defaults to 0.92.
    :type conv_eff: float
    :return: Power output at different wind speeds (kW).
    :rtype: np.ndarray
    """
    r_rotor = d_rotor / 2
    a_rotor = np.pi * r_rotor**2

    # Calculate minimum and maximum tip speeds
    if isinstance(r_min, list) and not r_min:
        r_min = 188.8 * d_rotor**-0.7081
    if isinstance(r_max, list) and not r_max:
        r_max = 793.7 * d_rotor**-0.8504

    vtip_min = r_min * (2 * np.pi * r_rotor) / 60
    vtip_max = r_max * (2 * np.pi * r_rotor) / 60

    # Optimal tip speed ratio and scaling factor
    tsr_range = np.arange(0, 12, 0.001)
    cp_values = calculate_cp(
        model=model, tsr=tsr_range, beta=beta
    )  # Explicit argument names
    tsr_opt = tsr_range[np.argmax(cp_values)]

    if isinstance(cp_max, list) and not cp_max:
        cp_max = max(cp_values)
    cp_scale = cp_max / max(cp_values)

    # Tip speed and power coefficient

    vtip = np.clip(
        tsr_opt * vws, vtip_min.values[..., None], vtip_max.values[..., None]
    )

    tsr = np.divide(vtip, vws, out=np.zeros_like(vws), where=vws > 0)

    cp0 = np.clip(
        conv_eff
        * cp_scale
        * calculate_cp(model=model, tsr=tsr, beta=beta),  # Explicit argument names
        0,
        None,
    )

    # Calculate power output
    # even though we have RHO per hour
    # we will use the average value
    pin = (
        0.5
        * air_density.mean(dim="time").values
        * a_rotor.values[..., None]
        * (vws**3)
        / 1000
    )  # Power input in kW (0.5 * rho * A * V^3)

    # Ensure pin is safe (avoid division by zero)
    pin_safe = np.where(pin > 0, pin, np.nan)

    # If pin or other inputs are xarray objects, convert to NumPy
    if isinstance(pin, xr.DataArray):
        pin_safe = pin_safe
        cp0 = cp0.values if isinstance(cp0, xr.DataArray) else cp0
        p_nom = p_nom.values if isinstance(p_nom, xr.DataArray) else p_nom

    # Apply np.minimum safely
    cp = np.minimum(
        cp0,
        np.minimum(1.0, p_nom.values[..., None] / pin_safe),
        where=~np.isnan(pin_safe),
    )

    p_out = cp * pin

    return p_out


def apply_turbulence_and_direction_effect(
    vws: xr.DataArray,
    pwt: xr.DataArray,
    tke: xr.DataArray,
    wind_dir: xr.DataArray,
    turbine_dir: xr.DataArray,
    v_cutin: float = 3,
    v_cutoff: float = 25,
) -> xr.DataArray:
    """
    Apply turbulence effects (via TKE) and wind direction effects on power output.

    :param vws: Wind speed values (xarray.DataArray).
    :param pwt: Power output without turbulence or wind direction effects (xarray.DataArray).
    :param tke: Turbulent Kinetic Energy (TKE) values (xarray.DataArray).
    :param wind_dir: Wind direction values (degrees, xarray.DataArray).
    :param turbine_dir: Turbine orientation (degrees, xarray.DataArray).
    :param v_cutin: Cut-in wind speed (m/s), defaults to 3.
    :param v_cutoff: Cut-off wind speed (m/s), defaults to 25.
    :return: Power output with turbulence and wind direction effects (xarray.DataArray).
    """
    # Calculate turbulence intensity (TI) from TKE
    # Cap vws to the range of mean_ws before interpolation
    mean_tke = tke["TKE"].mean(
        dim=[
            "year",
            "size",
            "value",
            "application",
        ]
    )
    mean_ws = tke["WS"].mean(
        dim=[
            "year",
            "size",
            "value",
            "application",
        ]
    )
    vws_clipped = np.clip(vws, mean_ws.min().values, mean_ws.max().values)
    interp_tke = np.interp(vws_clipped, mean_ws, mean_tke)

    # Avoid division by zero or invalid values
    ti = xr.apply_ufunc(
        lambda interp_tke, vws: np.where(
            (vws > 0) & (interp_tke > 0), np.sqrt(2 / 3 * interp_tke) / (vws + 1e-6), 0
        ),
        interp_tke,
        vws,
        dask="allowed",
        vectorize=True,
        keep_attrs=True,
    )

    # Replace NaNs with 0
    ti = np.nan_to_num(ti, nan=0)

    # Apply Gaussian smoothing for turbulence effect
    def gaussian_smoothing(tWS, vWS, Pwt, TI):
        # Ensure vWS and tWS are properly aligned
        vWS = vWS[
            ..., np.newaxis
        ]  # Add an extra dimension to vWS for alignment (1, 1, 1, 1, 31, 1)
        tWS = tWS[
            ..., np.newaxis, :
        ]  # Add an extra dimension to tWS for alignment (1, 1, 1, 1, 1, 31)

        # Adjust sigma for turbulence intensity
        sigma = np.maximum(TI[..., None] * tWS, 0.1 * tWS)  # Scale sigma with tWS
        sigma = np.maximum(sigma, 1e-6)  # Absolute minimum threshold

        # Calculate Gaussian weights
        gaussian_exponent = -0.5 * ((vWS - tWS) / sigma) ** 2
        weights = np.exp(gaussian_exponent)

        # Restrict weights to ±3 sigma
        valid_range = (vWS >= tWS - 3 * sigma) & (vWS <= tWS + 3 * sigma)
        weights = np.where(valid_range, weights, 0)

        # Normalize weights along the wind speed axis (-2)
        weights_sum = weights.sum(axis=-2, keepdims=True)  # Sum along vWS axis
        weights = np.divide(
            weights, weights_sum, out=np.zeros_like(weights), where=weights_sum > 0
        )

        # Collapse the extra dimension
        smoothed_power = np.sum(
            Pwt[..., np.newaxis] * weights, axis=-2
        )  # Collapse vWS axis (-2)

        return smoothed_power

    pwt = xr.apply_ufunc(
        gaussian_smoothing,
        vws,
        vws,
        pwt,
        ti,
        input_core_dims=[
            ["wind_speed"],
            ["wind_speed"],
            ["wind_speed"],
            ["wind_speed"],
        ],
        output_core_dims=[["wind_speed"]],
        vectorize=True,
        dask="allowed",
        keep_attrs=True,
    )

    # Set power to zero outside cut-in and cut-off wind speeds
    pwt = np.where(
        (vws >= v_cutin.values[..., None]) & (vws <= v_cutoff.values[..., None]), pwt, 0
    )

    return pwt


def calculate_rews(
    vws: np.ndarray, zhub: float, d_rotor: float, shear: float, veer: float
) -> np.ndarray:
    """
    Calculate the Rotor Equivalent Wind Speed (REWS) considering wind shear and veer effects.

    :param vws: Wind speed values (m/s).
    :type vws: np.ndarray
    :param zhub: Hub height (m).
    :type zhub: float
    :param d_rotor: Rotor diameter (m).
    :type d_rotor: float
    :param shear: Wind shear exponent.
    :type shear: float
    :param veer: Wind veer angle (degrees).
    :type veer: float
    :return: Rotor equivalent wind speed (REWS) values.
    :rtype: np.ndarray
    """
    n = 10  # Number of integration points along rotor span
    dz = d_rotor / n  # Differential element height (m)

    # Reshape zhub to match the shape of np.linspace output
    # Convert xarray.DataArray objects to NumPy arrays

    zhub = zhub.values  # Shape: (5, 2, 6, 1)
    zhub[zhub == 0] = 1
    start = (-d_rotor / 2 + dz / 2).values  # Shape: (5, 2, 6, 1)
    stop = (d_rotor / 2 - dz / 2).values  # Shape: (5, 2, 6, 1)

    # Generate np.linspace and add zhub
    zi = zhub[..., None] + np.linspace(
        start, stop, n, axis=-1
    )  # Final Shape: (5, 2, 6, 1, 10000)

    ai = (
        2 * np.sqrt((d_rotor.values[..., None] / 2) ** 2 - (zi - zhub[..., None]) ** 2)
    ) * dz.values[
        ..., None
    ]  # Area of each element (m^2)
    total_area = np.sum(ai, axis=-1)  # Total rotor area (m^2)

    coeff_shear = (
        zi / zhub[..., None]
    ) ** shear  # Shear coefficient across rotor height
    coeff_veer = np.cos(
        (zi - zhub[..., None]) * veer * np.pi / 180
    )  # Veer angle adjustment factor

    vi = vws * coeff_shear[..., None] * coeff_veer[..., None]
    rews = (np.sum((vi**3) * ai[..., None] / total_area[..., None, None], axis=-2)) ** (
        1 / 3
    )  # Rotor-equivalent wind speed (m/s)

    return rews


def calculate_generic_power_curve(
    vws: np.array,
    p_nom: xr.DataArray,
    d_rotor: xr.DataArray,
    zhub: xr.DataArray,
    v_cutin: xr.DataArray,
    v_cutoff: xr.DataArray,
    tke: float,
    shear: float = 0.15,
    veer: float = 0,
    r_min: Union[float, List[float]] = [],
    r_max: Union[float, List[float]] = [],
    cp_max: Union[float, List[float]] = [],
    model: str = "Dai et al. 2016",
    beta: Union[np.ndarray, List[float]] = [],
    air_density: float = 1.225,
    conv_eff: Union[float, List[float]] = [],
) -> np.ndarray:
    """
    Compute the complete wind turbine power curve considering environmental factors.

    :param vws: Wind speed values (m/s).
    :type vws: np.ndarray
    :param p_nom: Nominal power of the wind turbine (kW).
    :type p_nom: float
    :param d_rotor: Rotor diameter (m).
    :type d_rotor: float
    :param zhub: Hub height (m), defaults to calculated value.
    :type zhub: Union[float, List[float]], optional
    :param v_cutin: Cut-in wind speed (m/s), defaults to 3.
    :type v_cutin: float
    :param v_cutoff: Cut-off wind speed (m/s), defaults to 25.
    :type v_cutoff: float
    :param tke: Turbulent Kinetic Energy (TKE) value.
    :type tke: float
    :param shear: Wind shear exponent, defaults to 0.15.
    :type shear: float
    :param veer: Wind veer angle (degrees), defaults to 0.
    :type veer: float
    :param r_min: Minimum rotor speed, defaults to calculated value.
    :type r_min: Union[float, List[float]], optional
    :param r_max: Maximum rotor speed, defaults to calculated value.
    :type r_max: Union[float, List[float]], optional
    :param cp_max: Maximum power coefficient, defaults to calculated value.
    :type cp_max: Union[float, List[float]], optional
    :param model: Cp model, defaults to "Dai et al. 2016".
    :type model: str
    :param beta: Pitch angle, defaults to zero.
    :type beta: Union[np.ndarray, List[float]], optional
    :param air_density: Air density (kg/m^3), defaults to 1.225.
    :type air_density: float
    :param conv_eff: Conversion efficiency, defaults to calculated value.
    :type conv_eff: Union[float, List[float]], optional
    :return: Power output at different wind speeds (kW).
    :rtype: np.ndarray
    """
    if isinstance(conv_eff, list) and not conv_eff:
        gear_loss_const = 0.01
        gear_loss_var = 0.014
        generator_loss = 0.03
        converter_loss = 0.03
        conv_eff = (
            (1 - gear_loss_const)
            * (1 - gear_loss_var)
            * (1 - generator_loss)
            * (1 - converter_loss)
        )

    rews = calculate_rews(vws=vws, zhub=zhub, d_rotor=d_rotor, shear=shear, veer=veer)

    power_curve = calculate_raw_power_curve(
        vws=rews,
        p_nom=p_nom,
        d_rotor=d_rotor,
        r_min=r_min,
        r_max=r_max,
        cp_max=cp_max,
        model=model,
        beta=beta,
        air_density=air_density,
        conv_eff=conv_eff,
    )
    # fill NaNs with zeroes
    power_curve = np.nan_to_num(power_curve, nan=0)

    power_curve = apply_turbulence_and_direction_effect(
        vws=rews,
        pwt=power_curve,
        tke=tke,
        v_cutin=v_cutin,
        v_cutoff=v_cutoff,
        wind_dir=0,
        turbine_dir=0,
    )

    return power_curve
