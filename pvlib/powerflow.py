"""
This module contains functions for simulating power flow.
"""
import numpy as np
from pandas import DataFrame

from pvlib.inverter import _sandia_eff


def self_consumption(generation, load):
    """
    Calculate the power flow for a self-consumption use case. It assumes the
    system is connected to the grid.

    Parameters
    ----------
    generation : Series
        The AC generation profile. [W]
    load : Series
        The load profile. [W]

    Returns
    -------
    DataFrame
        The resulting power flow provided by the system and the grid into the
        system, grid and load. [W]
    """
    df = DataFrame(index=generation.index)
    df["Grid to system"] = -generation.loc[generation < 0]
    df["Grid to system"] = df["Grid to system"].fillna(0.0)
    df["Generation"] = generation.loc[generation > 0]
    df["Generation"] = df["Generation"].fillna(0.0)
    df["Load"] = load
    df["System to load"] = df[["Generation", "Load"]].min(axis=1, skipna=False)
    df.loc[df["System to load"] < 0, "System to load"] = 0.0
    df["System to grid"] = df["Generation"] - df["System to load"]
    df["Grid to load"] = df["Load"] - df["System to load"]
    df["Grid"] = df[["Grid to system", "Grid to load"]].sum(
        axis=1, skipna=False
    )
    return df


def self_consumption_ac_battery(df, dispatch, battery, model):
    """
    Calculate the power flow for a self-consumption use case with an
    AC-connected battery and a custom dispatch series. It assumes the system is
    connected to the grid.

    Parameters
    ----------
    df : DataFrame
        The self-consumption power flow solution. [W]
    dispatch : Series
        The dispatch series to use.
    battery : dict
        The battery parameters.
    model : str
        The battery model to use.

    Returns
    -------
    DataFrame
        The resulting power flow provided by the system, the grid and the
        battery into the system, grid, battery and load. [W]
    """
    final_state, results = model(battery, dispatch)
    df = df.copy()
    df["System to battery"] = -results["Power"]
    df.loc[df["System to battery"] < 0, "System to battery"] = 0.0
    df["System to battery"] = df[["System to battery", "System to grid"]].min(
        axis=1
    )
    df["System to grid"] -= df["System to battery"]
    df["Battery to load"] = results["Power"]
    df.loc[df["Battery to load"] < 0, "Battery to load"] = 0.0
    df["Battery to load"] = df[["Battery to load", "Grid to load"]].min(axis=1)
    df["Grid to load"] -= df["Battery to load"]
    df["Grid"] = df[["Grid to system", "Grid to load"]].sum(
        axis=1, skipna=False
    )
    return final_state, df


def self_consumption_dc_battery(dc_solution, load):
    """
    Calculate the power flow for a self-consumption use case with a
    DC-connected battery. It assumes the system is connected to the grid.

    Parameters
    ----------
    dc_solution : DataFrame
        The DC-connected inverter power flow solution. [W]
    load : Series
        The load profile. [W]

    Returns
    -------
    DataFrame
        The resulting power flow provided by the system, the grid and the
        battery into the system, grid, battery and load. [W]
    """
    df = self_consumption(dc_solution["AC power"], load)
    df["Battery"] = df["Generation"] * dc_solution["Battery factor"]
    df["Battery to load"] = df[["Battery", "System to load"]].min(axis=1)
    df["Battery to grid"] = df["Battery"] - df["Battery to load"]
    df["PV to battery"] = -dc_solution["Battery power flow"]
    df.loc[df["PV to battery"] < 0, "PV to battery"] = 0.0
    df["PV to load"] = df["System to load"] - df["Battery to load"]
    return df


def multi_dc_battery(
    v_dc, p_dc, inverter, battery_dispatch, battery_parameters, battery_model
):
    """
    Calculate the power flow for a self-consumption use case with a
    DC-connected battery. It assumes the system is connected to the grid.

    Parameters
    ----------
    v_dc : numeric
        DC voltage input to the inverter. [V]
    p_dc : numeric
        DC power input to the inverter. [W]
    inverter : dict
        Inverter parameters.
    battery_dispatch : Series
        Battery power dispatch series. [W]
    battery_parameters : dict
        Battery parameters.
    battery_model : str
        Battery model.

    Returns
    -------
    DataFrame
        The resulting inverter power flow.
    """
    dispatch = battery_dispatch.copy()

    # Limit charging to the available DC power
    power_dc = sum(p_dc)
    max_charging = -power_dc
    charging_mask = dispatch < 0
    dispatch[charging_mask] = np.max([dispatch, max_charging], axis=0)[
        charging_mask
    ]

    # Limit discharging to the inverter's maximum output power (approximately)
    # Note this can revert the dispatch and charge when there is too much DC
    # power (prevents clipping)
    max_discharging = inverter['Paco'] - power_dc
    discharging_mask = dispatch > 0
    dispatch[discharging_mask] = np.min([dispatch, max_discharging], axis=0)[
        discharging_mask
    ]

    # Calculate the actual battery power flow
    final_state, battery_flow = battery_model(battery_parameters, dispatch)
    charge = -battery_flow['Power'].copy()
    charge.loc[charge < 0] = 0
    discharge = battery_flow['Power'].copy()
    discharge.loc[discharge < 0] = 0

    # Adjust the DC power
    ratios = [sum(power) / sum(power_dc) for power in p_dc]
    adjusted_p_dc = [
        power - ratio * charge for (power, ratio) in zip(p_dc, ratios)
    ]
    final_dc_power = sum(adjusted_p_dc) + discharge

    # PV-contributed AC power
    pv_ac_power = 0.0 * final_dc_power
    for vdc, pdc in zip(v_dc, adjusted_p_dc):
        array_contribution = (
            pdc / final_dc_power * _sandia_eff(vdc, final_dc_power, inverter)
        )
        array_contribution[np.isnan(array_contribution)] = 0.0
        pv_ac_power += array_contribution

    # Battery-contributed AC power
    vdc = inverter["Vdcmax"] / 2
    pdc = discharge
    battery_ac_power = (
        pdc / final_dc_power * _sandia_eff(vdc, final_dc_power, inverter)
    )
    battery_ac_power[np.isnan(battery_ac_power)] = 0.0

    # Total AC power
    total_ac_power = pv_ac_power + battery_ac_power

    # Limit output power (Sandia limits)
    clipping = total_ac_power - inverter["Paco"]
    clipping[clipping < 0] = 0
    limited_ac_power = total_ac_power - clipping
    battery_factor = battery_ac_power / limited_ac_power
    min_ac_power = -1.0 * abs(inverter["Pnt"])
    below_limit = final_dc_power < inverter["Pso"]
    limited_ac_power[below_limit] = min_ac_power

    result = DataFrame(index=dispatch.index)
    result["Battery power flow"] = battery_flow["Power"]
    result["AC power"] = limited_ac_power
    result["Clipping"] = clipping
    result["Battery factor"] = battery_factor
    return result
