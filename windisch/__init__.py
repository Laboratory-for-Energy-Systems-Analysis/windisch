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
    "InventoryCalculation",
    "BackgroundSystemModel",
    "ExportInventory",
    "update_input_parameters"
)
__version__ = (0, 0, 1)

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

from .array import (
    fill_xarray_from_input_parameters,
    modify_xarray_from_custom_parameters,
)
from .background_systems import BackgroundSystemModel
from .turbines_input_parameters import TurbinesInputParameters
from .export import ExportInventory
from .inventory import InventoryCalculation
from .model import WindTurbineModel
from .update_input_params import update_input_parameters
