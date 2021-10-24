import json
from pathlib import Path


import windisch.turbines_input_parameters as tip

DEFAULT = Path(__file__, "..").resolve() / "fixtures" / "default_test.json"
EXTRA = Path(__file__, "..").resolve() / "fixtures" / "extra_test.json"


def test_retrieve_list_powertrains():
    assert isinstance(tip.TurbinesInputParameters().application, list)
    assert len(tip.TurbinesInputParameters().application) > 1


def test_can_pass_directly():
    d, e = json.load(open(DEFAULT)), set(json.load(open(EXTRA)))
    e.remove("foobazzle")
    assert len(tip.TurbinesInputParameters(d, e).application) == 2
    assert len(tip.TurbinesInputParameters(d, e).parameters) == 9


def test_alternate_filepath():
    assert len(tip.TurbinesInputParameters(DEFAULT, EXTRA).application) == 2
    assert len(tip.TurbinesInputParameters(DEFAULT, EXTRA).parameters) == 10
