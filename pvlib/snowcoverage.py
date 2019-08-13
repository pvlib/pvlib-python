
from __future__ import division

import datetime
import pytz

import numpy as np
import scipy as sp
import pandas as pd


import pvlib


def snow_slide_amount(surface_tilt, sliding_coefficient=1.97):
    '''
    Calculates the amount of snow that slides off of the surface of a module
    following the equations given in Marion et. al 2013 [1]

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
    '''

    tilt_radians = np.radians(surface_tilt)
    slide_amount = sliding_coefficient * np.sin(tilt_radians)
    return slide_amount


def snow_slide_event(poa_irradiance, temperature,
                     m=-80):
    '''
    Calculates when snow sliding events will occur
    following the equations given in Marion et. al 2013 [1]

    Parameters
    ----------
    poa_irradiance : numeric
        Total in-plane irradiance (W/m^2)

    temperature : numeric
        Ambient air temperature at the surface (C)

    m : numeric
        A coefficient used in the equations defined in [1]. It is empirically
        determined value given in W/m^2.

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
    '''

    slide_event = temperature > poa_irradiance / m
    return slide_event


def snow_covered_panel(snow_data, time_step_hours=1,
                       snow_data_type="snowfall"):
    '''


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
    panel_covered : numeric
        1 where the snowfall exceeds the defined threshold to fully cover the
        panel. NaN elsewhere.

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
    time_adjusted = snowfall / time_step_hours
    mask = time_adjusted >= 1
    panel_covered = np.full(snow_data.shape[0], np.nan)
    panel_covered[mask] = 1
    if isinstance(snow_data, pd.Series):
        panel_covered = pd.Series(panel_covered)
    return panel_covered


def snow_coverage(snow_data, snow_data_type,
                  poa_irradiance, temperature, surface_tilt,
                  time_step_hours=1, m=-80, sliding_coefficient=1.97):
    '''
    Calculates the fraction of a module covered by snow at one time point following Marion et. al

    inputs:
    snowfall: boolean. If True then snow_data is snowfall data (cm/hr). If False then snow_data is snow_depth data (cm).
    snow_data: either snowfall or snow_depth data. The original model was designed for snow_depth only.
    prev_data: snow_data for the previous time step. Used if snow_data is snow depth to calculate new snowfall.
    prev_coverage: fraction of module covered by snow at the previous time step.
    poa_irradiance: plane of array irradiance in W/m^2
    temperature: air temperature in C
    tilt: module tilt angle from horizontal in degrees
    time_step_minutes: period of the data in minutes. (minutes between data points)
    m=-80: an empirically determined value from Marion et al. 2013 in W/m^2
    sliding coefficient=1.97: also determined in Marion's paper for rooftop systems

    returns:
    the fraction of a module covered by snow
    '''
    snow_coverage = snow_covered_panel(snow_data,
                                       time_step_hours=time_step_hours,
                                       snow_data_type=snow_data_type)
    slide_events = snow_slide_event(poa_irradiance, temperature)
    slide_amount = snow_slide_amount(surface_tilt, sliding_coefficient)

    max_slides = int(np.ceil(10 / slide_amount))

    slidable_snow = ~np.isnan(snow_coverage)
    for i in range(max_slides - 1):
        new_slides = np.logical_and(slide_events, slidable_snow)
        snow_coverage[new_slides] -= slide_amount
        new_snow_coverage = snow_coverage.fillna(method="ffill", limit=1)
        slidable_snow = np.logical_and(~np.isnan(new_snow_coverage),
                                       np.isnan(snow_coverage))
        snow_coverage = new_snow_coverage

    new_slides = np.logical_and(slide_events, slidable_snow)
    snow_coverage[new_slides] -= slide_amount
    snow_coverage = snow_coverage.fillna(method="ffill")

    return snow_coverage

    # time_step_hours = time_step_minutes / 60.0

    # if snowfall:
    #     if snow_data / time_step_hours > 1:
    #         coverage = 1
    #     else:
    #         coverage = prev_coverage
    # else:
    #     if snow_data > 1 and (snow_data - prev_data) / time_step_hours > 1:
    #         coverage = 1
    #     elif snow_data == 0:
    #         coverage = 0
    #     else:
    #         coverage = prev_coverage

    # slide_amount = time_step_hours * 0.1 * snow_slide_amount(poa_irradiance, temperature, tilt, m, sliding_coefficient)
    # return max(0, coverage - slide_amount) 


def DC_loss_factor(snow_coverage, num_strings_per_row):
    '''
    Calculates the DC loss due to snow coverage.
    Assumes that if a string is even partially covered by snow, it produces 0W. 
   
    inputs:
    snow_coverage: the fraction of a module covered by snow
    num_strings_per_row: number of separate strings per module/row.
    
    returns:
    DC loss due to snow coverage. 
    '''
    loss_factor = np.ceil(snow_coverage * num_strings_per_row) / num_strings_per_row
    return loss_factor

def apply_snow_model(input_data, 
                     surface_tilt, 
                     time_step_minutes, 
                     num_strings_per_row, 
                     snow_data_type="None",
                     m=-80,
                     sliding_coefficient=1.97):
    '''
    Runs the snow model to generate DC losses from input data.
    Two parameters to the model can be specified.
    The type of snow data (snowfall or snow depth) can also be specified.
    Currently written for fixed-tilt systems but could be generalizd to tracking systems.

    inputs:
    input_data: datetime-indexed dataframe containing the columns listed below. 
    snow_data_type: must be either "snow_depth" or "snowfall" depending on the type of snow data in input_data.
    surface_tilt: module tilt angle from horizontal in degrees
    time_step_minutes: period of the data in minutes. (minutes between data points)
    m=-80: an empirically determined value from Marion et al. 2013 in W/m^2
    sliding coefficient=1.97: also determined in Marion's paper for rooftop systems

    returns:
    input_data with two appended columns: snow_coverage and DC_loss

    input_data required columns:
    poa_global: plan of array irradiance
    temp_air: air temperature
    snow_depth or snowfall: appropriate snow data
    ''' 
    output = input_data.copy()
    output["snow_coverage"] = np.nan
    output["DC_loss"] = np.nan
    prev_coverage = 0
    prev_data = 0
    for row in input_data.itertuples():
        if snow_data_type.lower() == "snow_depth":
            snow_data = row.snow_depth
            snowfall = False
        elif snow_data_type.lower() == "snowfall":
            snow_data = row.snowfall
            snowfall = True
        else:
            raise ValueError('snow_data_type was not specified or was not set to a valid option (snowfall, snow_depth).')
        timestamp = row.Index
        irradiance = row.poa_global
        temp = row.temp_air
        snow_coverage = snow_coverage_step(snowfall, snow_data, 
                                           prev_data, prev_coverage, 
                                           irradiance, temp, 
                                           surface_tilt, time_step_minutes,
                                           m)
        prev_data = snow_data
        prev_coverage = snow_coverage
        output.loc[timestamp,"snow_coverage"] = snow_coverage
        output.loc[timestamp,"DC_loss"] = DC_loss_factor(snow_coverage, num_strings_per_row)
    return output
