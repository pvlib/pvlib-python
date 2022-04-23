from numpy import nan
from numpy.random import uniform
from pandas import Series
from pandas import date_range
from pytest import approx
from pytest import mark

from pvlib.battery import boc
from pvlib.battery import fit_boc
from pvlib.powerflow import self_consumption
from pvlib.powerflow import self_consumption_ac_battery


@mark.parametrize(
    "generation,load,flow",
    [
        (
            42,
            20,
            {
                "Generation": 42,
                "Load": 20,
                "System to load": 20,
                "System to grid": 22,
                "Grid to load": 0,
                "Grid to system": 0,
                "Grid": 0,
            },
        ),
        (
            42,
            42,
            {
                "Generation": 42,
                "Load": 42,
                "System to load": 42,
                "System to grid": 0,
                "Grid to load": 0,
                "Grid to system": 0,
                "Grid": 0,
            },
        ),
        (
            42,
            50,
            {
                "Generation": 42,
                "Load": 50,
                "System to load": 42,
                "System to grid": 0,
                "Grid to load": 8,
                "Grid to system": 0,
                "Grid": 8,
            },
        ),
        (
            -3,
            0,
            {
                "Generation": 0,
                "Load": 0,
                "System to load": 0,
                "System to grid": 0,
                "Grid to load": 0,
                "Grid to system": 3,
                "Grid": 3,
            },
        ),
        (
            -3,
            42,
            {
                "Generation": 0,
                "Load": 42,
                "System to load": 0,
                "System to grid": 0,
                "Grid to load": 42,
                "Grid to system": 3,
                "Grid": 45,
            },
        ),
    ],
    ids=[
        "Positive generation with lower load",
        "Positive generation with same load",
        "Positive generation with higher load",
        "Negative generation with zero load",
        "Negative generation with positive load",
    ],
)
def test_self_consumption(generation, load, flow):
    """
    Check multiple conditions with well-known cases:

    - Excess generation must flow into grid
    - Load must be fed with the system when possible, otherwise from grid
    - Grid must provide energy for the system when required (i.e.: night hours)
    - Negative values from the input generation are removed from the output
      generation and added to the grid-to-system flow
    """
    result = (
        self_consumption(
            generation=Series([generation]),
            load=Series([load]),
        )
        .iloc[0]
        .to_dict()
    )
    assert approx(result) == flow


def test_self_consumption_sum():
    """
    The sum of the flows with respect to the system, load and grid must be
    balanced.
    """
    flow = self_consumption(
        generation=Series(uniform(0, 1, 1000)),
        load=Series(uniform(0, 1, 1000)),
    )
    assert (
        approx(flow["Generation"])
        == flow["System to load"] + flow["System to grid"]
    )
    assert (
        approx(flow["Grid"]) == flow["Grid to load"] + flow["Grid to system"]
    )
    assert (
        approx(flow["Load"]) == flow["System to load"] + flow["Grid to load"]
    )
    assert (
        approx(flow["Load"] + flow["Grid to system"])
        == flow["System to load"] + flow["Grid"]
    )


def test_self_consumption_ac_battery_sum(datasheet_battery_params):
    """
    The sum of the flows with respect to the system, load, grid and battery
    must be balanced.
    """
    index = date_range(
        "2022-01-01",
        periods=1000,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    _, flow = self_consumption_ac_battery(
        generation=Series(uniform(0, 1, 1000), index=index),
        load=Series(uniform(0, 1, 1000), index=index),
        battery=fit_boc(datasheet_battery_params),
        model=boc,
    )
    assert (
        approx(flow["Generation"])
        == flow["System to load"]
        + flow["System to battery"]
        + flow["System to grid"]
    )
    assert (
        approx(flow["Grid"]) == flow["Grid to load"] + flow["Grid to system"]
    )
    assert (
        approx(flow["Load"])
        == flow["System to load"]
        + flow["Battery to load"]
        + flow["Grid to load"]
    )


@mark.parametrize(
    "charge_efficiency,discharge_efficiency,efficiency",
    [
        (1.0, 1.0, 1.0),
        (0.97, 1.0, 0.97),
        (1.0, 0.95, 0.95),
        (0.97, 0.95, 0.97 * 0.95),
    ],
)
def test_self_consumption_ac_battery_losses(
    datasheet_battery_params,
    residential_generation_profile,
    residential_load_profile,
    charge_efficiency,
    discharge_efficiency,
    efficiency,
):
    """
    AC-DC conversion losses must be taken into account.

    With the BOC model these losses are easy to track if we setup a simulation
    in which we make sure to begin and end with an "empty" battery.
    """
    datasheet_battery_params["charge_efficiency"] = charge_efficiency
    datasheet_battery_params["discharge_efficiency"] = discharge_efficiency
    battery = fit_boc(datasheet_battery_params)
    residential_generation_profile.iloc[:1000] = 0.0
    residential_generation_profile.iloc[-1000:] = 0.0
    _, lossy = self_consumption_ac_battery(
        generation=residential_generation_profile,
        load=residential_load_profile,
        battery=battery,
        model=boc,
    )
    lossy = lossy.iloc[1000:]
    assert lossy["Battery to load"].sum() / lossy[
        "System to battery"
    ].sum() == approx(efficiency)


def test_self_consumption_nan_load():
    """
    When the load is unknown (NaN), the calculated flow to load should also be
    unknown.
    """
    flow = self_consumption(
        generation=Series([1, -2, 3, -4]),
        load=Series([nan, nan, nan, nan]),
    )
    assert flow["System to load"].isna().all()
    assert flow["Grid to load"].isna().all()
