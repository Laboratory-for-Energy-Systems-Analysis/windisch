import json
from pathlib import Path

from klausen import NamedParameters

DEFAULT = Path(__file__, "..").resolve() / "data" / "default_parameters.json"
EXTRA = Path(__file__, "..").resolve() / "data" / "extra_parameters.json"


def load_parameters(obj):
    if isinstance(obj, (str, Path)):
        assert Path(obj).exists(), "Can't find this filepath"
        return json.load(open(obj))
    else:
        # Already in correct form, just return
        return obj


class TurbinesInputParameters(NamedParameters):
    """
    A class used to represent wind turbines with associated type, size, technology, year and parameters.

    This class inherits from NamedParameters, located in the *klausen* package.
    It sources default parameters for all vehicle types from a dictionary in
    default_parameters and format them into an array following the structured described
    in the *klausen* package.

    :ivar sizes: List of string items e.g., ['100kW', '500kW', '1000kW']
    :vartype sizes: list
    :ivar application: List of string items e.g., ['onshore', 'offshore']
    :vartype application: list
    :ivar parameters: List of string items e.g., ['lifetime', 'power output', ...]
    :vartype parameters: list
    :ivar years: List of integers e.g., [2000, 2010, 2020, 2040]
    :vartype years: list
    :ivar metadata: Dictionary for metadata.
    :vartype metadata: dict
    :ivar values: Dictionary for storing values, of format {'param':[value]}.
    :vartype values: dict
    :ivar iterations: Number of iterations executed by the method :func:`~turbines_input_parameters.TurbinesInputParameters.stochastic`.
        None if :func:`~turbines_input_parameters.TurbinesInputParameters.static` used instead.
    :vartype iterations: int


    """

    def __init__(self, parameters=None, extra=None, limit=None):
        """Create a `klausen <https://github.com/cmutel/klausen>`__ model with the car input parameters."""
        super().__init__(None)

        parameters = load_parameters(DEFAULT if parameters is None else parameters)
        extra = set(load_parameters(EXTRA if extra is None else extra))

        if not isinstance(parameters, dict):
            raise ValueError(
                "Parameters are not correct type (expected `dict`, got `{}`)".format(
                    type(parameters)
                )
            )
        if not isinstance(extra, set):
            raise ValueError(
                "Extra parameters are not correct type (expected `set`, got `{}`)".format(
                    type(extra)
                )
            )
        self.sizes = sorted(
            {size for o in parameters.values() for size in o.get("sizes", [])}
        )
        self.application = sorted(
            {pt for o in parameters.values() for pt in o.get("application", [])}
        )
        self.parameters = sorted(
            {o["name"] for o in parameters.values()}.union(set(extra))
        )

        # keep a list of input parameters, for sensitivity purpose
        self.input_parameters = sorted({o["name"] for o in parameters.values()})

        self.years = sorted({o["year"] for o in parameters.values()})
        self.add_turbine_parameters(parameters)

    def add_turbine_parameters(self, parameters):
        """
        Split data and metadata according to ``klausen`` convention.

        The parameters are split into the *metadata* and *values* attributes
        of the CarInputParameters class by the add_parameters() method of the parent class.

        :param parameters: A dictionary that contains parameters.
        :type parameters: dict
        """
        KEYS = {"kind", "uncertainty_type", "amount", "loc", "minimum", "maximum"}

        reformatted = {}
        for key, dct in parameters.items():
            reformatted[key] = {k: v for k, v in dct.items() if k in KEYS}
            reformatted[key]["metadata"] = {
                k: v for k, v in dct.items() if k not in KEYS
            }

        self.add_parameters(reformatted)
