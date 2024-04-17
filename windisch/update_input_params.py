"""
update_input_params.py extracts input parameters values from `data/Input data.xlsx`
and format that into the json file `default_parameters.json`.
"""

import json

import numpy as np
import pandas as pd

from . import DATA_DIR

FILEPATH_TO_INPUT_DATA = DATA_DIR / "Input data.xlsx"


def update_input_parameters():
    """
    This function extract input parameter values from "Input data.xlsx"
    and store them in a json file, further used to build an input array.
    Does not return anything.
    """
    dataframe = pd.read_excel(FILEPATH_TO_INPUT_DATA)
    dataframe = dataframe.replace(np.nan, "", regex=True)

    dict_for_json = {}
    count = 0

    for _, row in dataframe.iterrows():
        if row["parameter"] != "":
            category = row["category"]

            if row["application"] == "all":
                application = ["onshore", "offshore"]
            else:
                application = [x.strip() for x in row["application"].split(",")]

            if row["sizes"] == "all":
                size = ["100kW", "500kW", "1000kW", "3000kW", ,"3400kW", "8000kW", "10000kW", "15000kW"]

            else:
                size = [x.strip() for x in str(row["sizes"]).split(",")]

            param = row["parameter"]
            unit = row["unit"]
            importance = row["importance"]
            status = row["status"]
            source = row["source"]
            comment = row["comment"]

            list_years = [2000, 2010, 2020, 2030, 2040, 2050]

            for year in list_years:
                if row[year] != "":
                    name = str(count) + "-" + str(year) + "-" + param

                    if row[str(year) + ".1"] != "" and row[str(year) + ".1"] != "":
                        dict_for_json[name] = {
                            "amount": row[year],
                            "category": category,
                            "application": application,
                            "sizes": size,
                            "year": year,
                            "name": param,
                            "unit": unit,
                            "importance": importance,
                            "source": source,
                            "comment": comment,
                            "status": status,
                            "kind": "distribution",
                            "uncertainty_type": 5,
                            "loc": row[year],
                            "minimum": row[str(year) + ".1"],
                            "maximum": row[str(year) + ".2"],
                        }
                    else:
                        dict_for_json[name] = {
                            "amount": row[year],
                            "category": category,
                            "application": application,
                            "sizes": size,
                            "year": year,
                            "name": param,
                            "unit": unit,
                            "importance": importance,
                            "source": source,
                            "comment": comment,
                            "status": status,
                            "kind": "distribution",
                            "uncertainty_type": 1,
                            "loc": row[year],
                        }

                    count += 1

    with open(DATA_DIR / "default_parameters.json", "w", encoding="utf-8") as filepath:
        json.dump(dict_for_json, filepath, indent=4)
