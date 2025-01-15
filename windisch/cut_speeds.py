"""
THIS MODULE LOADS THE CUT-IN CUT-OFF SPEEDS
"""
import csv
from pathlib import Path

FILEPATH_CUT_SPPEDS = Path("data") / "cut_speeds.csv"

def load_cut_in_off_speeds(powers: list):

    # load csv
    with open(FILEPATH_CUT_SPPEDS) as file:
        data = csv.reader(file)

        return data


