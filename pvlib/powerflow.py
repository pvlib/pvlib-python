"""
This module contains functions for simulating power flow.
"""
from pandas import DataFrame


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


def self_consumption_ac_battery(generation, load, battery, model):
    """
    Calculate the power flow for a self-consumption use case with an
    AC-connected battery. It assumes the system is connected to the grid.

    Parameters
    ----------
    generation : Series
        The input generation profile. [W]
    load : Series
        The input load profile. [W]
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
    df = self_consumption(generation, load)
    charging = df["System to grid"]
    discharging = df["Grid to load"]
    dispatch = discharging - charging
    final_state, results = model(battery, dispatch)
    df["System to battery"] = -results["Power"].loc[results["Power"] < 0]
    df["System to battery"] = df["System to battery"].fillna(0.0)
    df["System to grid"] -= df["System to battery"]
    df["Battery to load"] = results["Power"].loc[results["Power"] > 0]
    df["Battery to load"] = df["Battery to load"].fillna(0.0)
    df["Grid to load"] -= df["Battery to load"]
    df["Grid"] = df[["Grid to system", "Grid to load"]].sum(
        axis=1, skipna=False
    )
    return final_state, df


def self_consumption_dc_battery(
    ac_generation,
    ac_clipping,
    inverter_efficiency,
    inverter_max_output_power_w,
    load,
    battery,
    model,
):
    """
    Calculate the power flow for a self-consumption use case with an
    AC-connected battery. It assumes the system is connected to the grid.

    Parameters
    ----------
    generation : Series
        The input generation profile. [W]
    load : Series
        The input load profile. [W]
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
    df = self_consumption(ac_generation, load)
    charging = (df["System to grid"] + ac_clipping) / inverter_efficiency
    discharging = df["Grid to load"]
    discharging = min(discharging, inverter_max_output_power_w - ac_generation)
    discharging /= inverter_efficiency
    dispatch = discharging - charging
    final_state, results = model(battery, dispatch)
    df["System to battery"] = -results["Power"].loc[results["Power"] < 0]
    df["System to battery"] = df["System to battery"].fillna(0.0)
    df["System to grid"] -= df["System to battery"] * inverter_efficiency
    df["Battery to load"] = results["Power"].loc[results["Power"] > 0]
    df["Battery to load"] = df["Battery to load"].fillna(0.0)
    df["Battery to load"] *= inverter_efficiency
    df["Grid to load"] -= df["Battery to load"]
    df["Grid"] = df[["Grid to system", "Grid to load"]].sum(
        axis=1, skipna=False
    )
    return final_state, df


def self_consumption_ac_battery_custom_dispatch(df, dispatch, battery, model):
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
    df["System to battery"].loc[df["System to battery"] < 0] = 0.0
    df["System to battery"] = df[["System to battery", "System to grid"]].min(
        axis=1
    )
    df["System to grid"] -= df["System to battery"]
    df["Battery to load"] = results["Power"]
    df["Battery to load"].loc[df["Battery to load"] < 0] = 0.0
    df["Battery to load"] = df[["Battery to load", "Grid to load"]].min(axis=1)
    df["Grid to load"] -= df["Battery to load"]
    df["Grid"] = df[["Grid to system", "Grid to load"]].sum(
        axis=1, skipna=False
    )
    return final_state, df


def sandia_multi_dc_battery(v_dc, p_dc, inverter, dispatch, battery, model):
    power_dc = sum(p_dc)
    power_ac = 0. * power_dc

    # First, limit charging to the available DC power
    max_charging = -power_dc
    charging_mask = dispatch < 0
    dispatch[charging_mask] = np.max([dispatch, max_charging], axis=0)[charging_mask]

    # Second, limit discharging to the inverter's maximum output power (approximately)
    # Note this can revert the dispatch and charge when there is too much DC power (prevents clipping)
    max_discharging = inverter['Paco'] - power_dc
    discharging_mask = dispatch > 0
    dispatch[discharging_mask] = np.min([dispatch, max_discharging], axis=0)[discharging_mask]

    # Calculate the actual battery power flow
    final_state, results = model(battery, dispatch)

    # Adjust the DC power
    power_dc += results['Power']
    adjust_ratio = power_dc / sum(p_dc)

    for vdc, pdc in zip(v_dc, p_dc):
        pdc *= adjust_ratio
        power_ac += pdc / power_dc * _sandia_eff(vdc, power_dc, inverter)

    return _sandia_limits(power_ac, power_dc, inverter['Paco'],
                          inverter['Pnt'], inverter['Pso'])


def self_consumption_dc_battery_custom_dispatch(v_dc, p_dc, inverter, dispatch, battery, model):
    """
    Calculate the power flow for a self-consumption use case with a
    DC-connected battery and a custom dispatch series. It assumes the system is
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
    df["System to battery"].loc[df["System to battery"] < 0] = 0.0
    df["System to battery"] = df[["System to battery", "System to grid"]].min(
        axis=1
    )
    df["System to grid"] -= df["System to battery"]
    df["Battery to load"] = results["Power"]
    df["Battery to load"].loc[df["Battery to load"] < 0] = 0.0
    df["Battery to load"] = df[["Battery to load", "Grid to load"]].min(axis=1)
    df["Grid to load"] -= df["Battery to load"]
    df["Grid"] = df[["Grid to system", "Grid to load"]].sum(
        axis=1, skipna=False
    )
    return final_state, df
