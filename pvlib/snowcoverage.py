"""
The ``snow`` module contains functions that model the effect of snow on
solar modules.
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind


def _snow_slide_amount(surface_tilt, sliding_coefficient=1.97,
                       time_step=1):
    '''
    Calculates the amount of snow that slides off in each time step.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).


    sliding_coefficient : numeric
        An empirically determined coefficient used in [1] to determine
        how much snow slides off if a sliding event occurs.

    time_step: float
        Period of the data. [hour]

    Returns
    ----------
    slide_amount : numeric
        The fraction of panel slant height from which snow slides off
        at each time step, in tenths of the panel's slant height.
    '''

    return sliding_coefficient / 10 * sind(surface_tilt) * time_step


def _snow_slide_event(poa_irradiance, temperature,
                      m=-80):
    '''
    Calculates when snow sliding events will occur.

    Parameters
    ----------
    poa_irradiance : numeric
        Total in-plane irradiance [W/m^2]

    temperature : numeric
        Ambient air temperature at the surface [C]

    m : numeric
        A coefficient used in the model described in [1]. [W/m^2 C]

    Returns
    ----------
    slide_event : boolean array
        True if the condiditions are suitable for a snow slide event.
        False elsewhere.
    '''

    return temperature > poa_irradiance / m


def fully_covered_panel(snow_data, time_step=1,
                        snow_data_type="snowfall"):
    '''
    Calculates the timesteps when the panel is presumed to be fully covered
    by snow.

    Parameters
    ----------
    snow_data : numeric
        Time series data on either snowfall (cm/hr) or ground snow depth (cm).
        The type of data should be specified in snow_data_type.

    time_step: float
        Period of the data. [hour]

    snow_data_type : string
        Defines what type of data is being passed as snow_data. Acceptable
        values are "snowfall" and "snow_depth".

    Returns
    ----------
    fully_covered_mask : boolean array
        True where the snowfall exceeds the defined threshold to fully cover
        the panel. False elsewhere.

    Notes
    -----
    Implements the model described in [1] with minor improvements in [2].

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       “Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
       of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
    '''
    if snow_data_type == "snow_depth":
        prev_data = snow_data.shift(1)
        prev_data[0] = 0
        snowfall = snow_data - prev_data
    elif snow_data_type == "snowfall":
        snowfall = snow_data
    else:
        raise ValueError('snow_data_type was not specified or was not set to a'
                         'valid option (snowfall, snow_depth).')

    time_adjusted = snowfall / time_step
    fully_covered_mask = time_adjusted >= 1
    return fully_covered_mask


def snow_coverage_model(snow_data, snow_data_type,
                        poa_irradiance, temperature, surface_tilt,
                        time_step=1, m=-80, sliding_coefficient=1.97):
    '''
    Calculates the fraction of the slant height of a row of modules covered by
    snow at every time step.

    Parameters
    ----------
    snow_data : Series
        Time series data on either snowfall or ground snow depth. The type of
        data should be specified in snow_data_type. The original model was
        designed for ground snowdepth only. (cm/hr or cm)

    snow_data_type : string
        Defines what type of data is being passed as snow_data. Acceptable
        values are "snowfall" and "snow_depth". "snowfall" will be in units of
        cm/hr. "snow_depth" is in units of cm.

    poa_irradiance : Series
        Total in-plane irradiance (W/m^2)

    temperature : Series
        Ambient air temperature at the surface (C)

    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).

    time_step: float
        Period of the data. [hour]

    sliding coefficient : numeric
        Empirically determined coefficient used in [1] to determine how much
        snow slides off if a sliding event occurs.

    m : numeric
        A coefficient used in the model described in [1]. [W/(m^2 C)]

    Returns
    -------
    snow_coverage : numeric
        The fraction of a the slant height of a row of modules that is covered
        by snow at each time step.

    Notes
    -----
    Implements the model described in [1] with minor improvements in [2].
    Currently only validated for fixed tilt systems.

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       “Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
       of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
    '''

    full_coverage_events = fully_covered_panel(snow_data,
                                               time_step=time_step,
                                               snow_data_type=snow_data_type)
    snow_coverage = pd.Series(np.full(len(snow_data), np.nan))
    snow_coverage = snow_coverage.reindex(snow_data.index)
    snow_coverage[full_coverage_events] = 1
    slide_events = _snow_slide_event(poa_irradiance, temperature)
    slide_amount = _snow_slide_amount(surface_tilt, sliding_coefficient,
                                      time_step)
    slidable_snow = ~np.isnan(snow_coverage)
    while(np.any(slidable_snow)):
        new_slides = np.logical_and(slide_events, slidable_snow)
        snow_coverage[new_slides] -= slide_amount
        new_snow_coverage = snow_coverage.fillna(method="ffill", limit=1)
        slidable_snow = np.logical_and(~np.isnan(new_snow_coverage),
                                       np.isnan(snow_coverage))
        slidable_snow = np.logical_and(slidable_snow, new_snow_coverage > 0)
        snow_coverage = new_snow_coverage

    new_slides = np.logical_and(slide_events, slidable_snow)
    snow_coverage[new_slides] -= slide_amount

    snow_coverage = snow_coverage.fillna(method="ffill")
    snow_coverage = snow_coverage.fillna(value=0)
    snow_coverage[snow_coverage < 0] = 0
    return snow_coverage


def DC_loss_factor(snow_coverage, num_strings):
    '''
    Calculates the DC loss due to snow coverage. Assumes that if a string is
    even partially covered by snow, it produces 0W.

    Parameters
    ----------
    snow_coverage : numeric
        The fraction of row slant height covered by snow at each time step.

    num_strings: int
        The number of parallel-connected strings along a row slant height.

    Returns
    -------
    loss : numeric
        DC loss due to snow coverage at each time step.
    '''
    return np.ceil(snow_coverage * num_strings) / num_strings
