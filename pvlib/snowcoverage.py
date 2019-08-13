"""
The ``snowcoverage`` module contains functions that model snow coverage on
solar modules. The model was first proposed in Marion et al. 2013 and was
later validated and implemented into NREL's SAM.
"""

import numpy as np
import pandas as pd


def snow_slide_amount(surface_tilt, sliding_coefficient=1.97):
    '''
    Calculates the amount of snow that slides off of the surface of a module
    following the model first described in [1] and later implemented with minor
    improvements in SAM [2].

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).


    sliding coefficient : numeric
        Another empirically determined coefficient used in [1]. It determines
        how much snow slides off of the panel if a sliding event occurs.

    Returns
    ----------
    slide_amount : numeric
        The amount of snow that slides off of the panel
        in tenths of the panel area at each time step.

    References
    ----------
    [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
    “Measured and modeled photovoltaic system energy losses from snow for
    Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
    [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
    of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
    '''

    tilt_radians = np.radians(surface_tilt)
    slide_amount = sliding_coefficient * np.sin(tilt_radians)
    return slide_amount


def snow_slide_event(poa_irradiance, temperature,
                     m=-80):
    '''
    Calculates when snow sliding events will occur following the model first
    described in [1] and later implemented in SAM [2].

    Parameters
    ----------
    poa_irradiance : numeric
        Total in-plane irradiance (W/m^2)

    temperature : numeric
        Ambient air temperature at the surface (C)

    m : numeric
        A coefficient used in the model described in [1]. It is an
        empirically determined value given in W/m^2.

    Returns
    ----------
    slide_event : boolean array
        True if the condiditions are suitable for a snow slide event.
        False elsewhere.

    References
    ----------
    [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
    “Measured and modeled photovoltaic system energy losses from snow for
    Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
    [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
    of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
    '''

    slide_event = temperature > poa_irradiance / m
    return slide_event


def fully_covered_panel(snow_data, time_step_hours=1,
                        snow_data_type="snowfall"):
    '''
    Calculates the timesteps where the panel is presumed to be fully covered
    by snow. Follows the same model first described in [1] and later
    implemented in SAM [2].

    Parameters
    ----------
    snow_data : numeric
        Time series data on either snowfall or ground snow depth. The type of
        data should be specified in snow_data_type. The original model was
        designed for ground snowdepth only. (cm/hr or cm)

    time_step_hours: float
        Period of the data in hours. (hours between data points)

    snow_data_type : string
        Defines what type of data is being passed as snow_data. Acceptable
        values are "snowfall" and "snow_depth". "snowfall" will be in units of
        cm/hr. "snow_depth" is in units of cm.

    Returns
    ----------
    fully_covered_mask : boolean array
        True where the snowfall exceeds the defined threshold to fully cover
        the panel. False elsewhere.

    References
    ----------
    [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
    “Measured and modeled photovoltaic system energy losses from snow for
    Colorado and Wisconsin locations.” Solar Energy 97; pp.112-121.
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

    time_adjusted = snowfall / time_step_hours
    fully_covered_mask = time_adjusted >= 1
    return fully_covered_mask


def snow_coverage_model(snow_data, snow_data_type,
                        poa_irradiance, temperature, surface_tilt,
                        time_step_hours=1, m=-80, sliding_coefficient=1.97):
    '''
    Calculates the fraction of a module covered by snow at every time step
    following the same model first described in [1] and later
    implemented in SAM [2]. Currently only implemented for fixed tilt systems
    but could be generalized for tracking systems.

    Parameters
    ----------
    snow_data : numeric
        Time series data on either snowfall or ground snow depth. The type of
        data should be specified in snow_data_type. The original model was
        designed for ground snowdepth only. (cm/hr or cm)

    snow_data_type : string
        Defines what type of data is being passed as snow_data. Acceptable
        values are "snowfall" and "snow_depth". "snowfall" will be in units of
        cm/hr. "snow_depth" is in units of cm.

    poa_irradiance : numeric
        Total in-plane irradiance (W/m^2)

    temperature : numeric
        Ambient air temperature at the surface (C)

    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).

    time_step_hours: float
        Period of the data in hours. (hours between data points)

    sliding coefficient : numeric
        Another empirically determined coefficient used in [1]. It determines
        how much snow slides off of the panel if a sliding event occurs.
    m : numeric
        A coefficient used in the model described in [1]. It is an
        empirically determined value given in W/m^2.

    Returns
    -------
    snow_coverage : numeric
        The fraction of a module covered by snow at each time step.
    '''

    full_coverage_events = fully_covered_panel(snow_data,
                                               time_step_hours=time_step_hours,
                                               snow_data_type=snow_data_type)
    snow_coverage = pd.Series(np.full(len(snow_data), np.nan))
    snow_coverage[full_coverage_events] = 1
    slide_events = snow_slide_event(poa_irradiance, temperature)
    slide_amount = snow_slide_amount(surface_tilt, sliding_coefficient)
    max_slides = int(np.ceil(10 / slide_amount))

    slidable_snow = ~np.isnan(snow_coverage)
    for i in range(max_slides - 1):
        new_slides = np.logical_and(slide_events, slidable_snow)
        snow_coverage[new_slides] -= slide_amount*.1
        new_snow_coverage = snow_coverage.fillna(method="ffill", limit=1)
        slidable_snow = np.logical_and(~np.isnan(new_snow_coverage),
                                       np.isnan(snow_coverage))
        snow_coverage = new_snow_coverage

    new_slides = np.logical_and(slide_events, slidable_snow)
    snow_coverage[new_slides] -= slide_amount
    snow_coverage = snow_coverage.fillna(method="ffill")
    snow_coverage = snow_coverage.fillna(value=0)
    snow_coverage[snow_coverage < 0] = 0
    return snow_coverage


def DC_loss_factor(snow_coverage, num_strings_per_row):
    '''
    Calculates the DC loss due to snow coverage. Assumes that if a string is
    even partially covered by snow, it produces 0W.

    Parameters
    ----------
    snow_coverage : numeric
        The fraction of a module covered by snow at each time step.

    num_strings_per_row: int
        The number of separate strings per module/row.

    Returns
    -------
    loss_factor : numeric
        DC loss due to snow coverage at each time step.
    '''
    loss_factor = np.ceil(snow_coverage *
                          num_strings_per_row) / num_strings_per_row
    return loss_factor
