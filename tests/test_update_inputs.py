import json

import numpy as np
import pytest

from windisch.update_input_params import *


def test_output_format():
    """
    Change a value in `default_parameters.json`
    Run `update_input_parameters()`
    Check that the value has changed after update
    :return:
    """
    # we set the default values in the json file
    update_input_parameters()

    # we open it
    with open(DATA_DIR / "default_parameters.json", encoding="utf-8") as file:
        data = json.load(file)

    # the original value is 25
    assert data["216-2000-lifetime"]["amount"] == 25

    # we modify it to 30
    data["216-2000-lifetime"]["amount"] = 30
    assert data["216-2000-lifetime"]["amount"] == 30

    # we save the json file
    with open(DATA_DIR / "default_parameters.json", "w", encoding="utf-8") as filepath:
        json.dump(data, filepath, indent=4)

    # we run `update_input_parameters()` once again
    update_input_parameters()

    # we re-open it
    with open(DATA_DIR / "default_parameters.json", encoding="utf-8") as file:
        data = json.load(file)

    # we check that the value is overwritten and back to 20
    assert data["216-2000-lifetime"]["amount"] == 25
