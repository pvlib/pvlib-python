from numpy import array
from numpy import nan
from numpy.random import uniform
from pandas import Series
from pandas import Timestamp
from pandas import date_range
from pytest import approx
from pytest import mark

from pvlib.battery import boc
from pvlib.battery import fit_boc
from pvlib.powerflow import multi_dc_battery
from pvlib.powerflow import self_consumption
from pvlib.powerflow import self_consumption_ac_battery
from pvlib.pvsystem import retrieve_sam


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
        inclusive="left",
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


@mark.parametrize(
    "inputs,outputs",
    [
        (
            {"pv_power": 800, "dispatch": -400},
            {
                "Battery power flow": -400,
                "AC power": 400,
                "Clipping": 0,
                "Battery factor": 0,
            },
        ),
        (
            {"pv_power": 200, "dispatch": -600},
            {
                "Battery power flow": -200,
                "AC power": 0,
                "Clipping": 0,
                "Battery factor": nan,
            },
        ),
        (
            {"pv_power": 1200, "dispatch": 400},
            {
                "Battery power flow": -200,
                "AC power": 1000,
                "Clipping": 0,
                "Battery factor": 0,
            },
        ),
        (
            {"pv_power": 2000, "dispatch": 400},
            {
                "Battery power flow": -850,
                "AC power": 1000,
                "Clipping": 150,
                "Battery factor": 0,
            },
        ),
        (
            {"pv_power": 100, "dispatch": 400},
            {
                "Battery power flow": 400,
                "AC power": 500,
                "Clipping": 0,
                "Battery factor": 0.8,
            },
        ),
        (
            {"pv_power": 400, "dispatch": 1000},
            {
                "Battery power flow": 600,
                "AC power": 1000,
                "Clipping": 0,
                "Battery factor": 0.6,
            },
        ),
    ],
    ids=[
        "Charging is prioritized over AC conversion while enough PV power is available",
        "Charging is limited by the available PV power",
        "Clipping forces battery to charge, even when dispatch is set to discharge",
        "Clipping cannot be avoided if the battery is unable to handle too much input power",
        "Battery discharge can be combined with PV power to provide higher AC output power",
        "Battery discharge is limited to the inverter's nominal power",
    ],
)
def test_multi_dc_battery(inputs, outputs, datasheet_battery_params):
    """
    Test well-known cases of a multi-input (PV) and DC-connected battery
    inverter.

    - Assume an ideal inverter with 100 % DC-AC conversion efficiency
    - Assume an ideal battery with "infinite" capacity and 100 % efficiency
    - The inverter must try to follow the custom dispatch series as close as
      possible
    - Battery can only charge from PV
    - The inverter is smart enough to charge the battery in order to avoid
      clipping losses while still maintaining the MPP tracking
    """
    datasheet_battery_params.update({
        "dc_energy_wh": 100000,
        "dc_max_power_w": 850,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
    })
    inverter = {
        'Vac': '240',
        'Pso': 0.0,
        'Paco': 1000.0,
        'Pdco': 1000.0,
        'Vdco': 325.0,
        'C0': 0.0,
        'C1': 0.0,
        'C2': 0.0,
        'C3': 0.0,
        'Pnt': 0.5,
        'Vdcmax': 600.0,
        'Idcmax': 12.0,
        'Mppt_low': 100.0,
        'Mppt_high': 600.0,
    }
    dispatch = array([inputs["dispatch"]])
    index = date_range(
        "2022-01-01",
        periods=len(dispatch),
        freq="1H",
        tz="Europe/Madrid",
        inclusive="left",
    )
    dispatch = Series(data=dispatch, index=index)
    result = multi_dc_battery(
        v_dc=[array([400] * len(dispatch))],
        p_dc=[array([inputs["pv_power"]])],
        inverter=inverter,
        battery_dispatch=dispatch,
        battery_parameters=fit_boc(datasheet_battery_params),
        battery_model=boc,
    )
    assert approx(result.iloc[0].to_dict(), nan_ok=True) == outputs
