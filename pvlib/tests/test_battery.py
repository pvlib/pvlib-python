from itertools import product

import pytest
from pandas import Series
from pandas import date_range
from pvlib.tests.conftest import requires_pysam
from pytest import approx
from pytest import mark
from pytest import raises

from pvlib.battery import boc
from pvlib.battery import fit_boc
from pvlib.battery import fit_sam
from pvlib.battery import power_to_energy
from pvlib.battery import sam

all_models = mark.parametrize(
    "fit,run",
    [
        (fit_boc, boc),
        pytest.param(fit_sam, sam, marks=requires_pysam),
    ],
)

all_efficiencies = mark.parametrize(
    "charge_efficiency,discharge_efficiency",
    list(product([1.0, 0.98, 0.95], repeat=2)),
)


@mark.parametrize(
    "power_value,frequency,energy_value",
    [
        (1000, "H", 1000),
        (1000, "2H", 2000),
        (1000, "15T", 250),
        (1000, "60T", 1000),
    ],
)
def test_power_to_energy(power_value, frequency, energy_value):
    """
    The function should be able to convert power to energy for different power
    series' frequencies.
    """
    index = date_range(
        start="2022-01-01",
        periods=10,
        freq=frequency,
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(power_value, index=index)
    energy = power_to_energy(power)
    assert approx(energy) == energy_value


def test_power_to_energy_unsupported_frequency():
    """
    When the power series' frequency is unsupported, the function raises an
    exception.
    """
    index = date_range(
        start="2022-01-01",
        periods=10,
        freq="1M",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(1000, index=index)
    with raises(ValueError, match=r"Unsupported offset"):
        power_to_energy(power)


def test_fit_boc(datasheet_battery_params):
    """
    The function returns a dictionary with the BOC model parameters.
    """
    model = fit_boc(datasheet_battery_params)
    assert model["soc_percent"] == 50.0


@requires_pysam
def test_fit_sam(datasheet_battery_params):
    """
    The function returns a dictionary with a `"sam"` key that must be
    assignable to a SAM BatteryStateful model. Parameters like the nominal
    voltage and the battery energy must also be properly inherited.
    """
    from PySAM.BatteryStateful import new

    model = fit_sam(datasheet_battery_params)
    battery = new()
    battery.assign(model["sam"])
    assert approx(battery.value("nominal_voltage")) == 204
    assert approx(battery.value("nominal_energy")) == 5.12


@requires_pysam
def test_fit_sam_controls(datasheet_battery_params):
    """
    The controls group should not be exported as part of the battery state when
    creating a new SAM battery.
    """
    model = fit_sam(datasheet_battery_params)
    assert "Controls" not in set(model.keys())


@requires_pysam
def test_sam_controls(datasheet_battery_params):
    """
    The controls group should not be exported as part of the battery state when
    running a simulation with the SAM model.
    """
    index = date_range(
        start="2022-01-01",
        periods=100,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(1000.0, index=index)
    state = fit_sam(datasheet_battery_params)
    state, _ = sam(state, power)
    assert "Controls" not in set(state.keys())


@all_models
def test_model_return_index(datasheet_battery_params, fit, run):
    """
    The returned index must match the index of the given input power series.
    """
    index = date_range(
        start="2022-01-01",
        periods=100,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(1000.0, index=index)
    state = fit(datasheet_battery_params)
    _, result = run(state, power)
    assert all(result.index == power.index)


@all_models
def test_model_offset_valueerror(datasheet_battery_params, fit, run):
    """
    When the power series' frequency is unsupported, the models must raise an
    exception.
    """
    index = date_range(
        start="2022-01-01",
        periods=10,
        freq="1M",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(1000, index=index)
    state = fit(datasheet_battery_params)
    with raises(ValueError, match=r"Unsupported offset"):
        run(state, power)


@all_models
@all_efficiencies
def test_model_dispatch_power(
    datasheet_battery_params,
    fit,
    run,
    charge_efficiency,
    discharge_efficiency,
):
    """
    The dispatch power series represents the power flow as seen from the
    outside. As long as the battery is capable to provide sufficient power and
    sufficient energy, the resulting power flow should match the provided
    dispatch series independently of the charging/discharging efficiencies.
    """
    index = date_range(
        start="2022-01-01",
        periods=10,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    dispatch = Series(100.0, index=index)
    state = fit(datasheet_battery_params)
    _, result = run(state, dispatch)
    assert approx(result["Power"], rel=0.005) == dispatch


@all_models
@all_efficiencies
def test_model_soc_value(
    datasheet_battery_params,
    fit,
    run,
    charge_efficiency,
    discharge_efficiency,
):
    """
    The SOC should be updated according to the power flow and battery capacity.
    """
    index = date_range(
        start="2022-01-01",
        periods=20,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    step_percent = 1.5
    step_power = step_percent / 100 * datasheet_battery_params["dc_energy_wh"]
    power = Series(step_power, index=index)

    state = fit(datasheet_battery_params)
    _, result = run(state, power)
    assert (
        approx(result["SOC"].diff().iloc[-10:].mean(), rel=0.05)
        == -step_percent / datasheet_battery_params["discharge_efficiency"]
    )


@all_models
def test_model_charge_convergence(datasheet_battery_params, fit, run):
    """
    Charging only should converge into almost no power flow and maximum SOC.
    """
    index = date_range(
        start="2022-01-01",
        periods=100,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(-1000, index=index)
    state = fit(datasheet_battery_params)
    _, result = run(state, power)
    assert result["Power"].iloc[-1] == approx(0, abs=0.01)
    assert result["SOC"].iloc[-1] == approx(90, rel=0.01)


@all_models
def test_model_discharge_convergence(datasheet_battery_params, fit, run):
    """
    Discharging only should converge into almost no power flow and minimum SOC.
    """
    index = date_range(
        start="2022-01-01",
        periods=100,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(1000, index=index)
    state = fit(datasheet_battery_params)
    _, result = run(state, power)
    assert result["Power"].iloc[-1] == approx(0, abs=0.01)
    assert result["SOC"].iloc[-1] == approx(10, rel=0.01)


@all_models
def test_model_chain(datasheet_battery_params, fit, run):
    """
    The returning state must be reusable. Simulating continuously for ``2n``
    steps should be the same as splitting the simulation in 2 for ``n`` steps.
    """
    index = date_range(
        start="2022-01-01",
        periods=100,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    power = Series(2000.0, index=index)
    power.iloc[::2] = -2000.0
    half_length = int(len(power) / 2)

    continuous_state = fit(datasheet_battery_params)
    continuous_state, continuous_power = run(continuous_state, power)

    split_state = fit(datasheet_battery_params)
    split_state, split_power_0 = run(split_state, power[:half_length])
    split_state, split_power_1 = run(split_state, power[half_length:])
    split_power = split_power_0.append(split_power_1)

    assert split_state == continuous_state
    assert approx(split_power) == continuous_power


@all_models
def test_model_equivalent_periods(datasheet_battery_params, fit, run):
    """
    The results of a simulation with a 1-hour period should match those of a
    simulation with a 60-minutes period.
    """
    battery = fit(datasheet_battery_params)
    hourly_index = date_range(
        start="2022-01-01",
        periods=50,
        freq="1H",
        tz="Europe/Madrid",
        closed="left",
    )
    minutely_index = date_range(
        start="2022-01-01",
        periods=50,
        freq="60T",
        tz="Europe/Madrid",
        closed="left",
    )

    _, hourly = run(battery, Series(20.0, index=hourly_index))
    _, minutely = run(battery, Series(20.0, index=minutely_index))

    assert approx(hourly) == minutely


@all_models
def test_model_equivalent_power_timespan(datasheet_battery_params, fit, run):
    """
    Simulating with the same constant input power over the same time span but
    with different frequency should yield similar results.
    """
    battery = fit(datasheet_battery_params)
    half_index = date_range(
        start="2022-01-01 00:00",
        end="2022-01-02 00:00",
        freq="30T",
        tz="Europe/Madrid",
        closed="left",
    )
    double_index = date_range(
        start="2022-01-01 00:00",
        end="2022-01-02 00:00",
        freq="60T",
        tz="Europe/Madrid",
        closed="left",
    )

    _, half = run(battery, Series(20.0, index=half_index))
    _, double = run(battery, Series(20.0, index=double_index))

    assert approx(half.iloc[-1]["SOC"], rel=0.001) == double.iloc[-1]["SOC"]
    assert (
        approx(power_to_energy(half["Power"]).sum(), rel=0.001)
        == power_to_energy(double["Power"]).sum()
    )
