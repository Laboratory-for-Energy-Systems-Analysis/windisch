"""
inventory.py contains InventoryCalculation which provides all methods to solve inventories.
"""

import csv
import itertools
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyprind
import xarray as xr
import yaml
from scipy import sparse

from . import DATA_DIR
from .export import ExportInventory

IAM_FILES_DIR = DATA_DIR / "lcia"


def check_scenario(scenario):
    """Check if scenario is a valid scenario."""
    valid_scenarios = ["SSP2-NPi", "SSP2-PkBudg1150", "SSP2-PkBudg500", "static"]
    if scenario not in valid_scenarios:
        raise ValueError(
            f"Scenario must be one of " f"{valid_scenarios}, " f"not {scenario}"
        )
    return scenario


def get_dict_impact_categories(method, indicator) -> dict:
    """
    Load a dictionary with available impact assessment
    methods as keys, and assessment level and categories as values.

    :return: dictionary
    :rtype: dict
    """
    filename = "dict_impact_categories.csv"
    filepath = DATA_DIR / "lcia" / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            "The dictionary of impact categories could not be found."
        )

    csv_dict = {}

    with open(filepath, encoding="utf-8") as f:
        input_dict = csv.reader(f, delimiter=",")
        for row in input_dict:
            if row[0] == method and row[1] == indicator:
                csv_dict[row[3]] = {
                    "method": row[1],
                    "category": row[2],
                    "type": row[3],
                    "abbreviation": row[4],
                    "unit": row[5],
                    "source": row[6],
                }

    return csv_dict


def get_dict_input() -> dict:
    """
    Load a dictionary with tuple ("name of activity", "location", "unit",
    "reference product") as key, row/column
    indices as values.

    :return: dictionary with `label:index` pairs.
    :rtype: dict

    """
    filename = f"dict_inputs_A_matrix.csv"
    filepath = DATA_DIR / "lcia" / filename

    if not filepath.is_file():
        raise FileNotFoundError("The dictionary of activity labels could not be found.")

    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        raw = list(reader)
        for _r, r in enumerate(raw):
            if len(r) == 3:
                r[1] = eval(r[1])
            raw[_r] = tuple(r)

        return {j: i for i, j in enumerate(list(raw))}


def format_array(array):
    # Transpose the array as needed
    transposed_array = array.transpose(
        "value",
        "parameter",
        "size",
        "application",
        "year",
    )

    # Determine the new shape for the reshaping operation
    new_shape = (
        array.sizes["value"],
        array.sizes["parameter"],
        -1,
        array.sizes["year"],
    )

    # Reshape the array while keeping it as an xarray DataArray
    reshaped_array = transposed_array.data.reshape(new_shape)

    combined_coords = [
        " - ".join(list(x))
        for x in list(
            itertools.product(
                [str(s) for s in array.coords["size"].values],
                array.coords["application"].values,
            )
        )
    ]

    # Convert the reshaped numpy array back to xarray DataArray
    reshaped_dataarray = xr.DataArray(
        reshaped_array,
        dims=["value", "parameter", "combined_dim", "year"],
        coords=[
            array.coords["value"].values,
            array.coords["parameter"].values,
            combined_coords,
            array.coords["year"].values,
        ],
    )

    return reshaped_dataarray


class Inventory:
    """
    Build and solve the inventory for results characterization and inventory export

    :ivar model: object from the WindTurbine class
    :ivar background_configuration: dictionary that contains choices for background system
    :ivar scenario: IAM energy scenario to use (
        "SSP2-NPi": Nationally implemented policies, limits temperature increase by 2100 to 3.3 degrees Celsius,
        "SSP2-PkBudg1150": limits temperature increase by 2100 to 2 degrees Celsius,
        "SSP2-PkBudg500": limits temperature increase by 2100 to 1.5 degrees Celsius,
        "static": no forward-looking modification of the background inventories).
        "SSP2-NPi" selected by default.)
    :ivar method: impact assessment method to use ("recipe" or "ef" for Environmental Footprint 3.1)
    :ivar indicator: impact assessment indicator to use ("midpoint" or "endpoint")

    """

    def __init__(
        self,
        model,
        background_configuration: dict = None,
        scenario: str = "SSP2-NPi",
        method: str = "recipe",
        indicator: str = "midpoint",
    ) -> None:
        self.model = model

        self.scope = {
            "size": model.array.coords["size"].values.tolist(),
            "application": model.array.coords["application"].values.tolist(),
            "year": model.array.coords["year"].values.tolist(),
        }
        self.scenario = check_scenario(scenario)

        self.method = method
        self.indicator = indicator if method == "recipe" else "midpoint"

        self.array = format_array(model.array)
        self.iterations = len(model.array.value.values)

        self.number_of_turbines = (
            (self.model["lifetime electricity production"] > 0).sum().values
        )

        self.background_configuration = {}
        self.background_configuration.update(background_configuration or {})

        self.inputs = get_dict_input()

        self.add_additional_activities()
        self.rev_inputs = {v: k for k, v in self.inputs.items()}

        self.A = self.get_A_matrix()
        # Create electricity and fuel market datasets
        self.rev_inputs = {v: k for k, v in self.inputs.items()}

        self.list_cat, self.split_indices = self.get_split_indices()

        self.impact_categories = get_dict_impact_categories(
            method=self.method, indicator=self.indicator
        )

        # Create the B matrix
        self.B = self.get_B_matrix()
        self.rev_inputs = {v: k for k, v in self.inputs.items()}

        self.fill_in_A_matrix()
        self.add_wind_turbines_to_electricity_production_dataset()
        self.remove_non_compliant_turbines()

    def get_results_table(self, sensitivity: bool = False) -> xr.DataArray:
        """
        Format a xarray.DataArray array to receive the results.

        :param sensitivity: if True, the results table will
        be formatted to receive sensitivity analysis results
        :return: xarrray.DataArray
        """

        params = [a for a in self.array.value.values]
        response = xr.DataArray(
            np.zeros(
                (
                    len(self.impact_categories),
                    len(self.scope["size"]),
                    len(self.scope["application"]),
                    len(self.scope["year"]),
                    len(self.list_cat),
                    self.iterations,
                )
            ),
            coords=[
                list(self.impact_categories.keys()),
                self.scope["size"],
                self.scope["application"],
                self.scope["year"],
                self.list_cat,
                np.arange(0, self.iterations) if not sensitivity else params,
            ],
            dims=[
                "impact_category",
                "size",
                "application",
                "year",
                "impact",
                "value",
            ],
        )

        if sensitivity:
            # remove the `impact` dimension
            response = response.sum(dim="impact")

        return response

    def get_split_indices(self):
        """
        Return list of indices to split the results into categories.

        :return: list of indices
        :rtype: list
        """
        # read `impact_source_categories.yaml` file
        with open(
            DATA_DIR / "lcia" / "impact_source_categories.yaml", "r", encoding="utf-8"
        ) as stream:
            source_cats = yaml.safe_load(stream)

        idx_cats = defaultdict(list)

        for cat, name in source_cats.items():
            for n in name:
                idx = self.find_input_indices((n,))
                if idx and idx not in idx_cats[cat]:
                    idx_cats[cat].extend(idx)
            # remove duplicates
            idx_cats[cat] = list(set(idx_cats[cat]))

        # idx for an input that has no burden
        # oxygen in this case
        extra_idx = [j for i, j in self.inputs.items() if i[0].lower() == "oxygen"][0]

        list_ind = [val for val in idx_cats.values()]
        maxLen = max(map(len, list_ind))
        for row in list_ind:
            while len(row) < maxLen:
                row.append(extra_idx)

        return list(idx_cats.keys()), list_ind

    def calculate_impacts(self, sensitivity=False):
        if self.scenario != "static":
            list_b_arrays = []
            for year in self.scope["year"]:
                if year < min(self.B.year.values):
                    arr = self.B.sel(
                        year=min(self.B.year.values),
                    ).values
                elif year > max(self.B.year.values):
                    arr = self.B.sel(
                        year=max(self.B.year.values),
                    ).values
                else:
                    arr = self.B.interp(
                        year=year, method="linear", kwargs={"fill_value": "extrapolate"}
                    ).values
                list_b_arrays.append(arr)
            B = np.array(list_b_arrays)
        else:
            # if static scenario, use the B matrix
            # but we duplicate it for each year
            B = np.repeat(self.B.values, len(self.scope["year"]), axis=0)

        # Prepare an array to store the results
        results = self.get_results_table(sensitivity=sensitivity)

        new_arr = np.zeros((self.A.shape[1], self.B.shape[1], self.A.shape[-1]))

        f_vector = np.zeros((np.shape(self.A)[1]))

        # Collect indices of activities contributing to the first level
        idx_wind_elec = [
            x
            for x, y in self.rev_inputs.items()
            if y[0].startswith(f"electricity production, wind, ")
        ]

        idx_wind_power_plant = [
            x
            for x, y in self.rev_inputs.items()
            if y[0].startswith(f"wind turbine production, ")
        ]

        idx_others = [
            i
            for i in self.inputs.values()
            if not any(i in x for x in [idx_wind_elec, idx_wind_power_plant])
        ]

        arr = (
            self.A[
                np.ix_(
                    np.arange(self.iterations),
                    idx_others,
                    np.array(idx_wind_power_plant + idx_wind_elec),
                )
            ]
            .sum(axis=0)
            .sum(axis=1)
        )

        nonzero_idx = np.argwhere(arr)

        # use pyprind to display a progress bar
        bar = pyprind.ProgBar(len(nonzero_idx), stream=1, title="Calculating impacts")

        for a in nonzero_idx:
            bar.update()

            if isinstance(self.rev_inputs[a[0]][1], tuple):
                # it's a biosphere flow, hence no need to calculate LCA
                new_arr[a[0], :, a[1]] = B[a[1], :, a[0]]

            else:
                f_vector[:] = 0
                f_vector[a[0]] = 1
                X = sparse.linalg.spsolve(
                    sparse.csr_matrix(self.A[0, ..., a[1]]), f_vector.T
                )
                _X = (X * B[a[1]]).sum(axis=-1).T
                new_arr[a[0], :, a[1]] = _X

        new_arr = new_arr.transpose(1, 0, 2)

        arr = (
            self.A[:, :, idx_wind_elec].reshape(
                self.iterations,
                -1,
                len(self.scope["size"]),
                len(self.scope["application"]),
                len(self.scope["year"]),
            )
            * new_arr[:, None, :, None, None, :]
            * -1
        )

        arr += (
            self.A[:, :, idx_wind_power_plant].reshape(
                self.iterations,
                -1,
                len(self.scope["size"]),
                len(self.scope["application"]),
                len(self.scope["year"]),
            )
            * new_arr[:, None, :, None, None, :]
            * self.A[:, idx_wind_power_plant, idx_wind_elec].reshape(
                self.iterations,
                -1,
                len(self.scope["size"]),
                len(self.scope["application"]),
                len(self.scope["year"]),
            )
        )

        arr = arr[:, :, self.split_indices].sum(axis=3)

        # fetch indices not contained in self.split_indices
        # to see if there are other flows unaccounted for
        idx = [
            i
            for i in range(self.B.shape[-1])
            if i not in list(itertools.chain.from_iterable(self.split_indices))
        ]
        # check if any of the first items of nonzero_idx
        # are in idx
        for i in nonzero_idx:
            if i[0] in idx:
                print(f"The flow {self.rev_inputs[i[0]][0]} is not accounted for.")

        # reshape the array to match the dimensions of the results table
        arr = arr.transpose(0, 3, 4, 5, 2, 1)

        if sensitivity:
            results[...] = arr.sum(axis=-2)
            results /= results.sel(value="reference")
        else:
            results[...] = arr

        return results

    def add_additional_activities(self):
        # Add as many rows and columns as wind turbines to consider

        maximum = max(self.inputs.values())

        for size in self.scope["size"]:
            for application in self.scope["application"]:
                for year in self.scope["year"]:
                    unit = "kilowatt hour"

                    name = (
                        f"electricity production, wind, {application}, {size}kW, {year}"
                    )
                    ref = f"electricity, high voltage"

                    # add electricity production activity
                    key = (name, self.model.country, unit, ref)
                    if key not in self.inputs:
                        maximum += 1
                        self.inputs[(name, self.model.country, unit, ref)] = maximum

                    # add wind turbine production activity
                    key = (
                        name.replace(
                            f"electricity production, wind",
                            "wind turbine production",
                        ),
                        self.model.country,
                        "unit",
                        "wind turbine",
                    )

                    if key not in self.inputs:
                        maximum += 1
                        self.inputs[key] = maximum

    def get_A_matrix(self):
        """
        Load the A matrix. The matrix contains exchanges of products (rows)
        between activities (columns).

        :return: A matrix with three dimensions of shape (number of values,
        number of products, number of activities).
        :rtype: numpy.ndarray

        """

        filename = "A_matrix.npz"
        filepath = DATA_DIR / "lcia" / filename
        if not filepath.is_file():
            raise FileNotFoundError("The A matrix file could not be found.")

        # load matrix A
        initial_A = sparse.load_npz(filepath).toarray()

        new_A = np.identity(len(self.inputs))
        new_A[0 : np.shape(initial_A)[0], 0 : np.shape(initial_A)[0]] = initial_A

        # Resize the matrix to fit the number of `value` in `self.array`
        new_A = np.resize(
            new_A,
            (
                self.iterations,
                len(self.inputs),
                len(self.inputs),
            ),
        )

        # add a `year`dimension, with length equal to the number of years
        # in the scope
        new_A = np.repeat(new_A[:, :, :, None], len(self.scope["year"]), axis=-1)

        return new_A

    def get_B_matrix(self) -> xr.DataArray:
        """
        Load the B matrix. The B matrix contains impact assessment
        figures for a give impact assessment method,
        per unit of activity. Its length column-wise equals
        the length of the A matrix row-wise.
        Its length row-wise equals the number of
        impact assessment methods.

        :return: an array with impact values per unit
        of activity for each method.
        :rtype: numpy.ndarray

        """

        filepaths = [
            str(fp)
            for fp in list(Path(IAM_FILES_DIR).glob("*.npz"))
            if all(x in str(fp) for x in [self.method, self.indicator, self.scenario])
        ]

        if self.scenario != "static":
            filepaths = sorted(filepaths, key=lambda x: int(x[-8:-4]))

        B = np.zeros((len(filepaths), len(self.impact_categories), len(self.inputs)))

        for f, filepath in enumerate(filepaths):
            initial_B = sparse.load_npz(filepath).toarray()

            new_B = np.zeros(
                (
                    initial_B.shape[0],
                    len(self.inputs),
                )
            )

            new_B[0 : initial_B.shape[0], 0 : initial_B.shape[1]] = initial_B
            B[f, :, :] = new_B

        return xr.DataArray(
            B,
            coords=[
                (
                    [2005, 2010, 2020, 2030, 2040, 2050]
                    if self.scenario != "static"
                    else [2020]
                ),
                np.asarray(list(self.impact_categories.keys()), dtype="object"),
                np.asarray(list(self.inputs.keys()), dtype="object"),
            ],
            dims=["year", "category", "activity"],
        )

    def get_index_of_flows(self, items_to_look_for, search_by="name"):
        """
        Return list of row/column indices of self.A of labels that contain the string defined in `items_to_look_for`.

        :param items_to_look_for: string
        :param search_by: "name" or "compartment" (for elementary flows)
        :return: list of row/column indices
        :rtype: list
        """
        if search_by == "name":
            return [
                int(self.inputs[c])
                for c in self.inputs
                if all(ele in c[0].lower() for ele in items_to_look_for)
            ]
        if search_by == "compartment":
            return [
                int(self.inputs[c])
                for c in self.inputs
                if all(ele in c[1] for ele in items_to_look_for)
            ]

    def find_input_indices(
        self, contains: [tuple, str], excludes: tuple = (), excludes_in: int = 0
    ) -> list:
        """
        This function finds the indices of the inputs in the A matrix
        that contain the strings in the contains list, and do not
        contain the strings in the excludes list.
        :param contains: list of strings
        :param excludes: list of strings
        :param excludes_in: integer of item position to apply excludes filter
        :return: list of indices
        """
        indices = []

        if not isinstance(contains, tuple):
            contains = tuple(contains)

        if not isinstance(excludes, tuple):
            excludes = tuple(excludes)

        for i, input in enumerate(self.inputs):
            if all([c in input[0] for c in contains]) and not any(
                [e in input[excludes_in] for e in excludes]
            ):
                indices.append(i)

        if len(indices) == 0:
            print(
                f"No input found for {contains} and exclude {excludes} in the A matrix."
            )

        return indices

    def fill_in_A_matrix(self):
        """
        Fill-in the A matrix. Does not return anything. Modifies in place.
        Shape of the A matrix (values, products, activities).

        """

        for component in ["rotor", "nacelle", "tower"]:
            for location in ["onshore", "offshore"]:
                combined_dim_filter = [
                    d for d in self.array.coords["combined_dim"].values if location in d
                ]
                input_indices = [
                    j
                    for i, j in self.inputs.items()
                    if i[0].startswith("wind turbine production, ") and location in i[0]
                ]

                self.A[
                    :,
                    self.find_input_indices(
                        (f"{component} production, for {location} wind turbine",)
                    ),
                    input_indices,
                ] = (
                    self.array.sel(
                        parameter=f"{component} mass", combined_dim=combined_dim_filter
                    )
                    * -1
                )

                self.A[
                    :,
                    self.find_input_indices(
                        (f"treatment of {component}, for {location} wind turbine",)
                    ),
                    input_indices,
                ] = self.array.sel(
                    parameter=f"{component} mass", combined_dim=combined_dim_filter
                )

        # add electronic cabinet
        input_indices = [
            j
            for i, j in self.inputs.items()
            if i[0].startswith("wind turbine production, ")
        ]
        self.A[
            :,
            self.find_input_indices(
                (f"electronic cabinet production, for wind turbine",)
            ),
            input_indices,
        ] = -1

        # electronic cabinet EoL
        self.A[
            :,
            self.find_input_indices(
                (f"treatment of electronic cabinet, for wind turbine",)
            ),
            input_indices,
        ] = 1

        # grid connection
        self.A[
            :,
            self.find_input_indices(
                (f"grid connector production, per kg of copper, for wind turbine",)
            ),
            input_indices,
        ] = (
            self.array.sel(
                parameter=f"cable mass",
            )
            * -1
        )

        # grid connection EoL
        self.A[
            :,
            self.find_input_indices(
                (f"treatment of grid connector, for wind turbine",)
            ),
            input_indices,
        ] = self.array.sel(
            parameter=f"cable mass",
        )

        # medium-voltage transformer
        # we oversize it by 10%
        self.A[
            :,
            self.find_input_indices(
                (f"medium-voltage transformer production, for wind turbine",)
            ),
            input_indices,
        ] = (
            self.array.sel(
                parameter=f"power",
            )
            / 1000
            * 1.1
            * -1
        )

        # medium-voltage transformer EoL
        self.A[
            :,
            self.find_input_indices(
                (f"treatment of medium-voltage transformer, for wind turbine",)
            ),
            input_indices,
        ] = (
            self.array.sel(
                parameter=f"power",
            )
            / 1000
            * 1.1
        )

        # transition platform, for offshore wind turbines
        input_indices = [
            j
            for i, j in self.inputs.items()
            if i[0].startswith("wind turbine production, ") and "offshore" in i[0]
        ]
        self.A[
            :,
            self.find_input_indices(
                (f"platform production, for offshore wind turbine",)
            ),
            input_indices,
        ] = (
            self.array.sel(
                parameter=f"transition mass",
                combined_dim=[
                    d
                    for d in self.array.coords["combined_dim"].values
                    if "offshore" in d
                ],
            )
            * -1
        )

        # transition platform, EoL
        self.A[
            :,
            self.find_input_indices(
                (f"treatment of platform, for offshore wind turbine",)
            ),
            input_indices,
        ] = self.array.sel(
            parameter=f"transition mass",
            combined_dim=[
                d for d in self.array.coords["combined_dim"].values if "offshore" in d
            ],
        )

        # high-voltage transformer
        # allocated evenly among the number of turbines in the park
        self.A[
            :,
            self.find_input_indices(
                (f"high-voltage transformer production, for wind turbine",)
            ),
            input_indices,
        ] = 1 / (
            self.array.sel(
                parameter="turbines per farm",
                combined_dim=[
                    d
                    for d in self.array.coords["combined_dim"].values
                    if "offshore" in d
                ],
            )
            * -1
        )

        # high-voltage transformer EoL
        self.A[
            :,
            self.find_input_indices(
                (f"treatment of high-voltage transformer, for wind turbine",)
            ),
            input_indices,
        ] = 1 / (
            self.array.sel(
                parameter="turbines per farm",
                combined_dim=[
                    d
                    for d in self.array.coords["combined_dim"].values
                    if "offshore" in d
                ],
            )
        )

        # maintenance
        input_indices = [
            j
            for i, j in self.inputs.items()
            if i[0].startswith("electricity production, wind, ")
        ]
        self.A[
            :,
            self.find_input_indices(
                (f"market for lubricating oil",), excludes=("RoW",), excludes_in=1
            ),
            input_indices,
        ] = (
            (
                63  # 63 kg/year
                * self.array.sel(
                    parameter="lifetime",
                )
            )
            / self.array.sel(
                parameter="lifetime electricity production",
            )
            * -1
        )

    def add_wind_turbines_to_electricity_production_dataset(self):
        self.A[
            :,
            [
                x
                for x, y in self.rev_inputs.items()
                if y[0].startswith(f"wind turbine production, ")
            ],
            [
                x
                for x, y in self.rev_inputs.items()
                if y[0].startswith(f"electricity production, wind, ")
            ],
        ] = -1 / self.array.sel(parameter="lifetime electricity production")

    def remove_non_compliant_turbines(self):
        """
        Remove turbines from self.A that do not have an electricity production superior to 0.
        """
        # Get the indices of the wind turbines that are not compliant
        self.A = np.nan_to_num(self.A)
        idx = [
            x
            for x, y in self.rev_inputs.items()
            if y[0].startswith(f"wind turbine production")
        ]

        self.A[
            :,
            :,
            idx,
        ] *= (self.array.sel(parameter=["lifetime electricity production"]) > 0).values
        self.A[:, idx, idx] = 1

        idx = self.find_input_indices((f"electricity production, wind, ",))

        self.A[
            :,
            :,
            idx,
        ] *= (self.array.sel(parameter=["lifetime electricity production"]) > 0).values
        self.A[:, idx, idx] = 1

    def export_lci(
        self,
        ecoinvent_version="3.10",
        filename=f"windisch_lci",
        directory=None,
        software="brightway2",
        format="bw2io",
    ):
        """
        Export the inventory. Can export to Simapro (as csv), or brightway2 (as bw2io object, file or string).
        :param ecoinvent_version: str. "3.9" or "3.10"
        :param filename: str. Name of the file to be exported
        :param directory: str. Directory where the file is saved
        :param software: str. "brightway2" or "simapro"
        :param format: str. "bw2io" or "file" or "string"
        ::return: inventory, or the filepath where the file is saved.
        :rtype: list
        """

        if ecoinvent_version not in ["3.9", "3.10"]:
            raise ValueError("ecoinvent_version must be either '3.9' or '3.10'")

        lci = ExportInventory(
            array=self.A,
            model=self.model,
            indices=self.rev_inputs,
            db_name=f"{filename}_{datetime.now().strftime('%Y%m%d')}",
        )

        if software == "brightway2":
            return lci.write_bw2_lci(
                ecoinvent_version=ecoinvent_version,
                directory=directory,
                filename=f"{filename}_{datetime.now().strftime('%Y%m%d')}",
                export_format=format,
            )

        else:
            return lci.write_simapro_lci(
                ecoinvent_version=ecoinvent_version,
                directory=directory,
                filename=f"{filename}_{datetime.now().strftime('%Y%m%d')}",
                export_format=format,
            )
