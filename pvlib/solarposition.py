"""
Calculate the solar position using a variety of methods/packages.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd
    
try:
    from .spa_c_files.spa_py import spa_calc
except ImportError as e:
    pvl_logger.exception('Could not import built-in SPA calculator. You may need to recompile the SPA code.')

try:    
    import ephem
except ImportError as e:
    pvl_logger.warning('PyEphem not found.')



def spa(time, location, raw_spa_output=False):
    '''
    Calculate the solar position using the C implementation of the NREL 
    SPA code 

    The source files for this code are located in './spa_c_files/', along with
    a README file which describes how the C code is wrapped in Python. 

    Parameters
    ----------
    time : pandas.DatetimeIndex
    location : pvlib.Location object
    raw_spa_output : bool
        If true, returns the raw SPA output.

    Returns
    -------
    DataFrame
        The DataFrame will have the following columns:
        elevation, 
        azimuth,
        zenith.
    

    References
    ----------

    NREL SPA code: http://rredc.nrel.gov/solar/codesandalgorithms/spa/
    
    '''
    
    pvl_logger.debug('using built-in spa code to calculate solar position')
    
    time_utc = _localize_to_utc(time, location)
        
    spa_out = []
    
    for date in time_utc:
        spa_out.append(spa_calc(year=date.year,
                        month=date.month,
                        day=date.day,
                        hour=date.hour,
                        minute=date.minute,
                        second=date.second,
                        timezone=0, #timezone corrections handled above
                        latitude=location.latitude,
                        longitude=location.longitude,
                        elevation=location.altitude))
    
    spa_df = pd.DataFrame(spa_out, index=time_utc).tz_convert(location.tz)
    
    if raw_spa_output:
        return spa_df
    else:    
        dfout = spa_df[['zenith', 'azimuth']]
        dfout['elevation'] = 90 - dfout.zenith
    
        return dfout



def pyephem(time, location, pressure=101325, temperature=12):
    """
    Calculate the solar position using the PyEphem package.
    
    Parameters
    ----------
    time : pandas.DatetimeIndex
    location : pvlib.Location object
    pressure : int or float, optional
        air pressure in Pascals.
    temperature : int or float, optional
        air temperature in degrees C.
    
    Returns
    -------
    DataFrame
        The DataFrame will have the following columns:
        apparent_elevation, elevation, 
        apparent_azimuth, azimuth,
        apparent_zenith, zenith.
    """
    
    pvl_logger.debug('using PyEphem to calculate solar position')
    
    time_utc = _localize_to_utc(time, location)
    
    sun_coords = pd.DataFrame(index=time_utc)
    
    # initialize a PyEphem observer
    obs = ephem.Observer()
    obs.lat = str(location.latitude)
    obs.lon = str(location.longitude)
    obs.elevation = location.altitude
    obs.pressure = pressure / 100. # convert to mBar
    obs.temp = temperature
    
    # the PyEphem sun
    sun = ephem.Sun()
    
    # make and fill lists of the sun's altitude and azimuth
    # this is the pressure and temperature corrected apparent alt/az.
    alts = []
    azis = []
    for thetime in sun_coords.index:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        alts.append(sun.alt)
        azis.append(sun.az)
    
    sun_coords['apparent_elevation'] = alts
    sun_coords['apparent_azimuth'] = azis
    
    # redo it for p=0 to get no atmosphere alt/az
    obs.pressure = 0
    alts = []
    azis = []
    for thetime in sun_coords.index:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        alts.append(sun.alt)
        azis.append(sun.az)
    
    sun_coords['elevation'] = alts
    sun_coords['azimuth'] = azis
    
    # convert to degrees. add zenith
    sun_coords = np.rad2deg(sun_coords)
    sun_coords['apparent_zenith'] = 90 - sun_coords['apparent_elevation']
    sun_coords['zenith'] = 90 - sun_coords['elevation']
    
    try:
        return sun_coords.tz_convert(location.tz)
    except TypeError:
        return sun_coords.tz_localize(location.tz)
        
        
        
def _localize_to_utc(time, location):
    """
    Calculate the solar position using the PyEphem package.
    
    Parameters
    ----------
    time : pandas.DatetimeIndex
    location : pvlib.Location object
    
    Returns
    -------
    pandas.DatetimeIndex
        localized to UTC.
    """
    
    try:
        time_utc = time.tz_convert('UTC')
        pvl_logger.debug('tz_convert to UTC')
    except TypeError:
        time_utc = time.tz_localize(location.tz).tz_convert('UTC')
        pvl_logger.debug('tz_localize to {} and then tz_convert to UTC'.format(location.tz))
        
    return time_utc
    
