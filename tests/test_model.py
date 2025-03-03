from windisch import *
import numpy as np

def test_attributes():
    """
    Test that resulting wind turbines have plausible dimensions and weights.
    """

    update_input_parameters()
    tip = TurbinesInputParameters()
    tip.static()
    _, array = fill_xarray_from_input_parameters(
        tip,
        scope={
            "year": [2020, 2050],
            "size": [1000, ],
        }
    )

    wt = WindTurbineModel(
        array,
        country="DK"
    )
    wt.set_all()

    # make sure that no mass is equal to zero
    assert np.all(wt.array.sel(parameter="total mass").values > 0)
    # make sure that no rotor diameter is equal to zero
    assert np.all(wt.array.sel(parameter="rotor diameter").values > 0)
    # make sure that rotor diameter is inferior to 100m
    assert np.all(wt.array.sel(parameter="rotor diameter").values < 100)
    # make sure that nacelle mass is inferior to 100_000 kg
    assert np.all(wt.array.sel(parameter="nacelle mass").values < 100_000)
    # make sure that rotor mass is inferior to 75_000 kg
    assert np.all(wt.array.sel(parameter="rotor mass").values < 75_000)
    # make sure that tower mass is inferior to 500_000 kg
    assert np.all(wt.array.sel(parameter="tower mass").values < 500_000)

    # make sure that electricity production is superior to
    # a production using a capacity factor of 0.1
    assert np.all(wt.array.sel(parameter="lifetime electricity production") >= wt.array.sel(parameter="power") * 0.1 * 24 * 365 * 20)

    # make sure that electricity production is inferior to a production
    # using a capacity factor of 1
    assert np.all(wt.array.sel(parameter="lifetime electricity production") <= wt.array.sel(parameter="power") * 24 * 365 * 20)