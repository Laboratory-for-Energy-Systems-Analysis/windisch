import numpy as np
import pytest

from windisch import (
    TurbinesInputParameters,
    fill_xarray_from_input_parameters,
    modify_xarray_from_custom_parameters,
)


def test_type_cip():
    with pytest.raises(TypeError) as wrapped_error:
        fill_xarray_from_input_parameters("bla")
    assert wrapped_error.type == TypeError


def test_format_array():
    cip = TurbinesInputParameters()
    cip.static()
    dcts, array = fill_xarray_from_input_parameters(cip)

    assert np.shape(array)[0] == len(dcts[0])
    assert np.shape(array)[1] == len(dcts[1])
    assert np.shape(array)[2] == len(dcts[2])
    assert np.shape(array)[3] == len(dcts[3])


def test_modify_array():
    cip = TurbinesInputParameters()
    cip.static()
    _, array = fill_xarray_from_input_parameters(cip)

    dict_param = {
        ("Operation", "all", "all", "lifetime", "none"): {
            (2020, "loc"): 30,
            (2040, "loc"): 30,
        }
    }

    modify_xarray_from_custom_parameters(dict_param, array)
    assert (
        array.sel(
            application="onshore",
            size="100kW",
            year=2020,
            parameter="lifetime",
            value=0
        ).values
        == 30
    )


def test_wrong_param_modify_array():
    cip = TurbinesInputParameters()
    cip.static()
    _, array = fill_xarray_from_input_parameters(cip)

    dict_param = {
        ("Operation", "all", "all", "foo", "none"): {
            (2020, "loc"): 15,
            (2040, "loc"): 15,
        }
    }

    modify_xarray_from_custom_parameters(dict_param, array)
    with pytest.raises(KeyError) as wrapped_error:
        array.sel(application="onshore", size="100kW", year=2020, parameter="foo")
    assert wrapped_error.type == KeyError


def test_scope():
    """Test that the use of scope dictionary works as intended"""
    cip = TurbinesInputParameters()
    cip.static()
    scope = {"application": ["offshore"], "size": ["1000kW"]}
    _, array = fill_xarray_from_input_parameters(cip, scope=scope)

    assert "onshore" not in array.coords["application"].values
    assert "500kW" not in array.coords["size"].values
