import pandas as pd
import json
import numpy as np
from . import DATA_DIR

FILEPATH_TO_INPUT_DATA = DATA_DIR / "Input data.xlsx"


def update_input_parameters():
    """
    This function extract input parameter values from "Input data.xlsx"
    and store them in a json file, further used to build an input array.
    Does not return anything.
    """
    df = pd.read_excel(FILEPATH_TO_INPUT_DATA)
    df = df.replace(np.nan, "", regex=True)

    d = {}
    count = 0

    d_dist = {"triangular": 5, "None": 1}

    for i, j in df.iterrows():
        if j["parameter"] != "":

            category = j["category"]

            if j["application"] == "all":
                application = ["onshore", "offshore"]
            else:
                application = [x.strip() for x in j["application"].split(",")]

            if j["sizes"] == "all":
                size = ["100kW", "500kW", "1000kW", "3000kW", "8000kW"]

            else:
                size = [x.strip() for x in str(j["sizes"]).split(",")]

            param = j["parameter"]
            unit = j["unit"]
            importance = j["importance"]
            status = j["status"]
            source = j["source"]
            comment = j["comment"]

            list_years = [2000, 2010, 2020, 2030, 2040, 2050]

            for y in list_years:
                if j[y] != "":
                    name = str(count) + "-" + str(y) + "-" + param

                    if j[str(y) + ".1"] != "" and j[str(y) + ".1"] != "":
                        d[name] = {
                            "amount": j[y],
                            "category": category,
                            "application": application,
                            "sizes": size,
                            "year": y,
                            "name": param,
                            "unit": unit,
                            "importance": importance,
                            "source": source,
                            "comment": comment,
                            "status": status,
                            "kind": "distribution",
                            "uncertainty_type": 5,
                            "loc": j[y],
                            "minimum": j[str(y) + ".1"],
                            "maximum": j[str(y) + ".2"],
                        }
                    else:
                        d[name] = {
                            "amount": j[y],
                            "category": category,
                            "application": application,
                            "sizes": size,
                            "year": y,
                            "name": param,
                            "unit": unit,
                            "importance": importance,
                            "source": source,
                            "comment": comment,
                            "status": status,
                            "kind": "distribution",
                            "uncertainty_type": 1,
                            "loc": j[y],
                        }

                    count += 1

    with open(DATA_DIR / "default_parameters.json", 'w') as fp:
            json.dump(d, fp, indent=4)
