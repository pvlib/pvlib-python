import os
import inspect
import datetime
import random
import pytz

import numpy as np
import scipy as sp
import pandas as pd

import pvlib
import pvlib.pvsystem as pvsystem

import matplotlib
import matplotlib.pyplot as plt


def snow_slide_amount(poa_irradiance,
                      temperature,
                      surface_tilt,
                      m=-80,
                      sliding_coefficient=1.97):
    '''
    Calculates the amount of snow that slides off of a module following
    the equations given in Marion et. al 2013 [1]

    Parameters
    ----------
    poa_irradiance : numeric
        Total in-plane irradiance (W/m^2)

    temperature : numeric
        Ambient air temperature at the surface (C)

    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).

    m : numeric
        an empirically determined value from Marion et al. 2013 in W/m^2

    sliding coefficient : numeric
        also determined in Marion's paper for rooftop systems

    Returns
    ----------
    slide_amount : numeric
        The amount of snow that slides off of the panel
        in tenths of the panel area.
    '''

    tilt_radians = np.radians(surface_tilt)
    slide_amount = sliding_coefficient * np.sin(tilt_radians)
    mask = temperature <= poa_irradiance / m
    slide_amount[mask] = 0
    return slide_amount


def snow_coverage_step(snowfall, 
                       snow_data, 
                       prev_data, 
                       prev_coverage, 
                       poa_irradiance, 
                       temperature, 
                       tilt, 
                       time_step_minutes,
                       m=-80,
                       sliding_coefficient=1.97):
                      
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
    

    assert prev_coverage >= 0 and prev_coverage <= 1
    time_step_hours = time_step_minutes / 60.0

    if snowfall:
        if snow_data / time_step_hours > 1:
            coverage = 1
        else:
            coverage = prev_coverage
    else:
        if snow_data > 1 and (snow_data - prev_data) / time_step_hours > 1:
            coverage = 1
        elif snow_data == 0:
            coverage = 0
        else:
            coverage = prev_coverage
        
    slide_amount = time_step_hours * 0.1 * snow_slide_amount(poa_irradiance, temperature, tilt, m, sliding_coefficient)
    return max(0, coverage - slide_amount) 
    
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
    assert snow_coverage >= 0 and snow_coverage <= 1
    temp = np.ceil(snow_coverage * num_strings_per_row) / num_strings_per_row
    assert  temp >= 0 and temp <=1
    return temp

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
