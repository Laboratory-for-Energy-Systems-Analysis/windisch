"""
__init__ file.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


"""

Submodules
==========

.. autosummary::
    :toctree: _autosummary


"""

__all__ = (
    "TurbinesInputParameters",
    "fill_xarray_from_input_parameters",
    "modify_xarray_from_custom_parameters",
    "WindTurbineModel",
    "update_input_parameters",
)
__version__ = (0, 0, 1)

from wind_array import (
    fill_xarray_from_input_parameters,
    modify_xarray_from_custom_parameters,
)
from model import WindTurbineModel
from turbines_input_parameters import TurbinesInputParameters
from update_input_params import update_input_parameters

print("hello")

