"""
fill_xarray_from_input_parameters creates an xarray and fills it with sampled input parameter values
if a distribution is defined.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import stats_arrays as sa
import xarray as xr

from .turbines_input_parameters import TurbinesInputParameters as t_i_p


def fill_xarray_from_input_parameters(
    tip: "t_i_p", sensitivity: bool = False, scope: dict = None
) -> Tuple[Tuple, xr.DataArray]:
    """Create an `xarray` labeled array from the sampled input parameters.

    This function extracts the parameters' names and values contained in the
    `parameters` attribute of the :class:`TurbinesInputParameters` class in :mod:`turbines_input_parameters` and insert them into a
    multi-dimensional numpy-like array from the *xarray* package
    (http://xarray.pydata.org/en/stable/).


    :param sensitivity:
    :param tip: Instance of the :class:`TurbinesInputParameters` class in :mod:`turbines_input_parameters`.
    :param scope: a dictionary to narrow down the scope of vehicles to consider
    :returns: `tuple`, `xarray.DataArray`
    - tuple (`size_dict`, `application_dict`, `parameter_dict`, `year_dict`)
    - array

    Dimensions of `array`:

        0. Turbine size (in kW), e.g. "100", "500". str.
        1. Application, e.g. "onshore", "offshore". str.
        2. Year. int.
        3. Samples.

    """

    # Check whether the argument passed is a cip object
    if not isinstance(tip, t_i_p):
        raise TypeError(
            "The argument passed is not an object of the TurbinesInputParameters class"
        )

    if scope is None:
        scope = {"size": tip.sizes, "application": tip.application, "year": tip.years}
    else:
        if "size" not in scope:
            scope["size"] = tip.sizes
        if "application" not in scope:
            scope["application"] = tip.application
        if "year" not in scope:
            scope["year"] = tip.years

    if any(s for s in scope["size"] if s not in tip.sizes):
        raise ValueError("One of the size types is not valid.")

    if any(y for y in scope["year"] if y not in tip.years):
        raise ValueError("One of the years defined is not valid.")

    if any(app for app in scope["application"] if app not in tip.application):
        raise ValueError("One of the application types is not valid.")

    if not sensitivity:
        array = xr.DataArray(
            np.zeros(
                (
                    len(scope["size"]),
                    len(scope["application"]),
                    len(tip.parameters),
                    len(scope["year"]),
                    tip.iterations or 1,
                )
            ),
            coords=[
                scope["size"],
                scope["application"],
                tip.parameters,
                scope["year"],
                np.arange(tip.iterations or 1),
            ],
            dims=["size", "application", "parameter", "year", "value"],
        )
    else:
        params = ["reference"]
        params.extend(tip.input_parameters)
        array = xr.DataArray(
            np.zeros(
                (
                    len(scope["size"]),
                    len(scope["application"]),
                    len(tip.parameters),
                    len(scope["year"]),
                    len(params),
                )
            ),
            coords=[tip.sizes, tip.application, tip.parameters, tip.years, params],
            dims=["size", "application", "parameter", "year", "value"],
        )

    size_dict = {k: i for i, k in enumerate(scope["size"])}
    application_dict = {k: i for i, k in enumerate(scope["application"])}
    year_dict = {k: i for i, k in enumerate(scope["year"])}
    parameter_dict = {k: i for i, k in enumerate(tip.parameters)}

    if not sensitivity:
        for param in tip:
            pwt = (
                set(tip.metadata[param]["application"])
                if isinstance(tip.metadata[param]["application"], list)
                else {tip.metadata[param]["application"]}
            )

            size = (
                set(tip.metadata[param]["sizes"])
                if isinstance(tip.metadata[param]["sizes"], list)
                else {tip.metadata[param]["sizes"]}
            )

            year = (
                set(tip.metadata[param]["year"])
                if isinstance(tip.metadata[param]["year"], list)
                else {tip.metadata[param]["year"]}
            )

            if (
                pwt.intersection(scope["application"])
                and size.intersection(scope["size"])
                and year.intersection(scope["year"])
            ):
                array.loc[
                    dict(
                        application=[p for p in pwt if p in scope["application"]],
                        size=[s for s in size if s in scope["size"]],
                        year=[y for y in year if y in scope["year"]],
                        parameter=tip.metadata[param]["name"],
                    )
                ] = tip.values[param]
    else:
        for param in tip.input_parameters:
            names = [n for n in tip.metadata if tip.metadata[n]["name"] == param]

            for name in names:
                pwt = (
                    set(tip.metadata[name]["application"])
                    if isinstance(tip.metadata[name]["application"], list)
                    else {tip.metadata[name]["application"]}
                )

                size = (
                    set(tip.metadata[name]["sizes"])
                    if isinstance(tip.metadata[name]["sizes"], list)
                    else {tip.metadata[name]["sizes"]}
                )

                year = (
                    set(tip.metadata[name]["year"])
                    if isinstance(tip.metadata[name]["year"], list)
                    else {tip.metadata[name]["year"]}
                )

                vals = [
                    tip.values[name] for _ in range(0, len(tip.input_parameters) + 1)
                ]
                vals[tip.input_parameters.index(param) + 1] *= 1.1

                array.loc[
                    dict(
                        application=[p for p in pwt if p in scope["application"]],
                        size=[s for s in size if s in scope["size"]],
                        year=[y for y in year if y in scope["year"]],
                        parameter=tip.metadata[name]["name"],
                    )
                ] = vals

    return (size_dict, application_dict, parameter_dict, year_dict), array


def modify_xarray_from_custom_parameters(
    filepath: Union[str, dict], array: xr.DataArray
) -> xr.DataArray:
    """
    Override default parameters values in `xarray` based on values provided by the user.

    This function allows to override one or several default parameter values by providing either:

        * a file path to an Excel workbook that contains the new values
        * or a dictionary

    The dictionary must be of the following format:

    .. code-block:: python

            {
                (parameter category,
                    application,
                    size,
                    parameter name,
                    uncertainty type): {
                                        (year, 'loc'): value,
                                        (year, 'scale'): value,
                                        (year, 'shape'): value,
                                        (year, 'minimum'): value,
                                        (year, 'maximum'): value
                }

            }

    For example:

    .. code-block:: python

            {
                ('Operation',
                'all',
                'all',
                'lifetime',
                'none'): {
                    (2018, 'loc'): 15, (2040, 'loc'): 25
                    }

            }

    :param array: the array to modify
    :param filepath: File path of workbook with new values or dictionary.
    :type filepath: str or dict
    :return: the original array, but modified

    """

    if isinstance(filepath, str):
        try:
            dataframe = pd.read_excel(
                filepath,
                header=[0, 1],
                index_col=[0, 1, 2, 3, 4],
                sheet_name="Custom_parameters",
            ).to_dict(orient="index")
        except Exception as err:
            raise FileNotFoundError("Custom parameters file not found.") from err
    elif isinstance(filepath, dict):
        dataframe = filepath
    else:
        raise TypeError("The format passed as parameter is not valid.")

    forbidden_keys = ["Background", "Functional unit"]

    for row in dataframe:
        if row[0] not in forbidden_keys:
            if not isinstance(row[1], str):
                application_type = [p.strip() for p in row[1] if p]
                application_type = [p for p in application_type if p]
                application_type = list(application_type)
            elif row[1] == "all":
                application_type = array.coords["application"].values
            else:
                if row[1] in array.coords["application"].values:
                    application_type = [row[1]]
                elif all(
                    p
                    for p in row[1].split(", ")
                    if p in array.coords["application"].values
                ):
                    application_type = row[1].split(", ")
                else:
                    print(
                        f"{row[1]} is not a recognized application. It will be skipped."
                    )
                    continue

            if not isinstance(row[2], str):
                sizes = [s.strip() for s in row[2] if s]
                sizes = [s for s in sizes if s]
                sizes = list(sizes)
            elif row[2] == "all":
                sizes = array.coords["size"].values
            else:
                if row[2] in array.coords["size"].values:
                    sizes = [row[2]]
                elif all(
                    s for s in row[2].split(", ") if s in array.coords["size"].values
                ):
                    sizes = row[2].split(", ")
                else:
                    print(
                        f"{row[2]} is not a recognized size category. It will be skipped."
                    )
                    continue

            param = row[3]

            if not param in array.coords["parameter"].values:
                print(f"{param} is not a recognized parameter. It will be skipped.")
                continue

            val = dataframe[row]

            distr_dic = {
                "triangular": 5,
                "lognormal": 2,
                "normal": 3,
                "uniform": 4,
                "none": 1,
            }
            distr = distr_dic[row[4]]

            years = {v[0] for v in val}

            for year in years:
                # No uncertainty parameters given
                if distr == 1:
                    # There should be at least a `loc`
                    if ~np.isnan(val[(year, "loc")]):
                        for size in sizes:
                            for app in application_type:
                                array.loc[
                                    dict(
                                        application=app,
                                        size=size,
                                        year=year,
                                        parameter=param,
                                    )
                                ] = val[(year, "loc")]
                    # Otherwise warn
                    else:
                        print(f"`loc`parameter missing for {param} in {year}.")
                        continue

                elif distr in [2, 3, 4, 5]:
                    # Check if the correct parameters are present
                    # Triangular

                    if distr == 5:
                        if (
                            np.isnan(val[(year, "loc")])
                            or np.isnan(val[(year, "minimum")])
                            or np.isnan(val[(year, "maximum")])
                        ):
                            missing_param_for_distribution(param, "triangular", year)
                            continue

                    # Lognormal
                    if distr == 2:
                        if np.isnan(val[(year, "loc")]) or np.isnan(
                            val[(year, "scale")]
                        ):
                            missing_param_for_distribution(param, "lognormal", year)
                            continue

                    # Normal
                    if distr == 3:
                        if np.isnan(val[(year, "loc")]) or np.isnan(
                            val[(year, "scale")]
                        ):
                            missing_param_for_distribution(param, "normal", year)
                            continue

                    # Uniform
                    if distr == 4:
                        if np.isnan(val[(year, "minimum")]) or np.isnan(
                            val[(year, "maximum")]
                        ):
                            missing_param_for_distribution(param, "uniform", year)
                            continue

                    distribution_def = sa.UncertaintyBase.from_dicts(
                        {
                            "loc": val[year, "loc"],
                            "scale": val[year, "scale"],
                            "shape": val[year, "shape"],
                            "minimum": val[year, "minimum"],
                            "maximum": val[year, "maximum"],
                            "uncertainty_type": distr,
                        }
                    )

                    # Stochastic mode
                    if array.sizes["value"] > 1:
                        rng = sa.MCRandomNumberGenerator(distribution_def)

                        for size in sizes:
                            for app in application_type:
                                array.loc[
                                    dict(
                                        application=app,
                                        size=size,
                                        year=year,
                                        parameter=param,
                                    )
                                ] = rng.generate(array.sizes["value"]).reshape((-1,))
                    else:
                        dist = sa.uncertainty_choices[distr]
                        median = float(dist.ppf(distribution_def, np.array((0.5,))))

                        for size in sizes:
                            for app in application_type:
                                array.loc[
                                    dict(
                                        application=app,
                                        size=size,
                                        year=year,
                                        parameter=param,
                                    )
                                ] = median

                else:
                    print(
                        f"The uncertainty type is not recognized for {param} in {year}.\n The parameter is skipped and default value applies"
                    )
                    continue

    return array


def missing_param_for_distribution(param: str, dist_type: str, year: int):
    """
    Print a warning message in case of misisng parameters for a distribution.
    :param param: input parameter for which the distribution parameter is missing, str.
    :param dist_type: distribution type, str.
    :param year: year, int
    :return: nothing.
    """

    print(
        f"One or more parameters for the {dist_type} distribution is/are missing for {year} in {param}.\n "
        f"The parameter is skipped and the default value applies."
    )
