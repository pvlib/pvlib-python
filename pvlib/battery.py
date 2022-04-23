"""
This module contains functions for modeling batteries.
"""
from pandas import DataFrame

try:
    from PySAM.BatteryStateful import default as sam_default
    from PySAM.BatteryStateful import new as sam_new
    from PySAM.BatteryTools import battery_model_sizing as sam_sizing
except ImportError:  # pragma: no cover

    def missing_nrel_pysam(*args, **kwargs):
        raise ImportError(
            "NREL's PySAM package required! (`pip install nrel-pysam`)"
        )

    sam_default = missing_nrel_pysam
    sam_new = missing_nrel_pysam
    sam_sizing = missing_nrel_pysam


def offset_to_hours(offset):
    """
    Convert a Pandas offset into hours.

    Parameters
    ----------
    offset : pd.tseries.offsets.BaseOffset
        The input offset to convert.

    Returns
    -------
    numeric
        The resulting period, in hours.
    """
    if offset.name == "H":
        return offset.n
    if offset.name == "T":
        return offset.n / 60
    raise ValueError("Unsupported offset {}".format(offset))


def power_to_energy(power):
    """
    Converts a power series to an energy series.

    Assuming Watts as the input power unit, the output energy unit will be
    Watt Hours.

    Parameters
    ----------
    power : Series
        The input power series. [W]

    Returns
    -------
    The converted energy Series. [Wh]
    """
    return power * offset_to_hours(power.index.freq)


def fit_boc(model):
    """
    Determine the BOC model matching the given characteristics.

    Parameters
    ----------
    datasheet : dict
        The datasheet parameters of the battery.

    Returns
    -------
    dict
        The BOC parameters.

    Notes
    -----
    This function does not really perform a fitting procedure. Instead, it just
    calculates the model parameters that match the provided information from a
    datasheet.
    """
    params = {
        "soc_percent": 50,
    }
    model.update(params)
    return model


def boc(model, dispatch):
    """
    Run a battery simulation with a provided dispatch series. Positive power
    represents the power provided by the battery (i.e.: discharging) while
    negative power represents power provided to the battery (i.e.: charging).

    The provided dispatch series is the goal/target power, but the battery may
    not be able to provide or store that much energy given its characteristics.
    This function will calculate how much power will actually flow from/into
    the battery.

    Uses a simple "bag of Coulombs" model.

    Parameters
    ----------
    model : dict
        The initial BOC parameters.
    dispatch : Series
        The target power series. [W]

    Returns
    -------
    export : dict
        The final BOC parameters.
    results : DataFrame
        The resulting:

        - Power flow. [W]
        - SOC. [%]
    """
    min_soc = model.get("min_soc_percent", 10)
    max_soc = model.get("max_soc_percent", 90)
    factor = offset_to_hours(dispatch.index.freq)

    states = []
    current_energy = model["dc_energy_wh"] * model["soc_percent"] / 100
    max_energy = model["dc_energy_wh"] * max_soc / 100
    min_energy = model["dc_energy_wh"] * min_soc / 100

    dispatch = dispatch.copy()
    discharge_efficiency = model.get("discharge_efficiency", 1.0)
    charge_efficiency = model.get("charge_efficiency", 1.0)
    dispatch.loc[dispatch < 0] *= charge_efficiency
    dispatch.loc[dispatch > 0] /= discharge_efficiency

    for power in dispatch:
        if power > 0:
            power = min(power, model["dc_max_power_w"])
            energy = power * factor
            available = current_energy - min_energy
            energy = min(energy, available)
            power = energy / factor * discharge_efficiency
        else:
            power = max(power, -model["dc_max_power_w"])
            energy = power * factor
            available = current_energy - max_energy
            energy = max(energy, available)
            power = energy / factor / charge_efficiency
        current_energy -= energy
        soc = current_energy / model["dc_energy_wh"] * 100
        states.append((power, soc))

    results = DataFrame(states, index=dispatch.index, columns=["Power", "SOC"])

    final_state = model.copy()
    final_state["soc_percent"] = results.iloc[-1]["SOC"]

    return (final_state, results)


def fit_sam(datasheet):
    """
    Determine the SAM BatteryStateful model matching the given characteristics.

    Parameters
    ----------
    datasheet : dict
        The datasheet parameters of the battery.

    Returns
    -------
    dict
        The SAM BatteryStateful parameters.

    Notes
    -----
    This function does not really perform a fitting procedure. Instead, it just
    calculates the model parameters that match the provided information from a
    datasheet.
    """
    chemistry = {
        "LFP": "LFPGraphite",
    }
    model = sam_default(chemistry[datasheet["chemistry"]])
    sam_sizing(
        model=model,
        desired_power=datasheet["dc_max_power_w"] / 1000,
        desired_capacity=datasheet["dc_energy_wh"] / 1000,
        desired_voltage=datasheet["dc_nominal_voltage"],
    )
    model.ParamsCell.initial_SOC = 50
    model.ParamsCell.minimum_SOC = datasheet.get("min_soc_percent", 10)
    model.ParamsCell.maximum_SOC = datasheet.get("max_soc_percent", 90)
    export = model.export()
    del export["Controls"]
    result = {}
    result["sam"] = export
    result["charge_efficiency"] = datasheet.get("charge_efficiency", 1.0)
    result["discharge_efficiency"] = datasheet.get("discharge_efficiency", 1.0)
    return result


def sam(model, power):
    """
    Run a battery simulation with a provided dispatch series. Positive power
    represents the power provided by the battery (i.e.: discharging) while
    negative power represents power provided to the battery (i.e.: charging).

    The provided dispatch series is the goal/target power, but the battery may
    not be able to provide or store that much energy given its characteristics.
    This function will calculate how much power will actually flow from/into
    the battery.

    Uses SAM's BatteryStateful model.

    Parameters
    ----------
    model : dict
        The initial SAM BatteryStateful parameters.
    dispatch : Series
        The target dispatch power series. [W]

    Returns
    -------
    export : dict
        The final SAM BatteryStateful parameters.
    results : DataFrame
        The resulting:

        - Power flow. [W]
        - SOC. [%]
    """
    battery = sam_new()
    battery.ParamsCell.assign(model["sam"].get("ParamsCell", {}))
    battery.ParamsPack.assign(model["sam"].get("ParamsPack", {}))
    battery.Controls.replace(
        {
            "control_mode": 1,
            "dt_hr": offset_to_hours(power.index.freq),
            "input_power": 0,
        }
    )
    battery.setup()
    battery.StateCell.assign(model["sam"].get("StateCell", {}))
    battery.StatePack.assign(model["sam"].get("StatePack", {}))

    battery_dispatch = power.copy()
    discharge_efficiency = model.get("discharge_efficiency", 1.0)
    charge_efficiency = model.get("charge_efficiency", 1.0)
    battery_dispatch.loc[power < 0] *= charge_efficiency
    battery_dispatch.loc[power > 0] /= discharge_efficiency

    states = []
    for p in battery_dispatch:
        battery.Controls.input_power = p / 1000
        battery.execute(0)
        states.append((battery.StatePack.P * 1000, battery.StatePack.SOC))

    results = DataFrame(states, index=power.index, columns=["Power", "SOC"])
    results.loc[results["Power"] < 0, "Power"] /= charge_efficiency
    results.loc[results["Power"] > 0, "Power"] *= discharge_efficiency
    export = battery.export()
    del export["Controls"]

    state = model.copy()
    state["sam"] = export
    return (state, results)
