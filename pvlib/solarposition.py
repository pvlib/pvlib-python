"""
Calculate the solar position using a variety of methods/packages.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd

# optional imports.

try:    
    import ephem
except ImportError as e:
    pvl_logger.warning('PyEphem not found.')

from . import pvl_ephemeris



def get_solarposition(time, location, method='pyephem', pressure=101325, 
                      temperature=12):
    """
    Calculate the solar position using a variety of methods.
    * 'pvlib' uses the pvlib ephemeris code. Not recommended as of June 2014.
    * 'pyephem' uses the PyEphem package. Default.
    
    The returned DataFrame will be localized to location.tz.
    
    :param time: pandas.DatetimeIndex. 
    :param location: pvlib Location object.
    :param method: string. 'pvlib', 'pyephem'
    :param pressure: int or float. Pascals.
    :param temperature: int or float. Degrees C.
    
    :returns: pandas.DataFrame with columns 'elevation', 'zenith', 'azimuth'.
              Additional columns including 'solar_time', 'apparent_elevation',
              and 'apparent_zenith' are available with the 'pvlib' and 'pyephem' options.
    """
    
    method = method.lower()
    
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time.tz_localize(location.tz).tz_convert('UTC')
    
    if method == 'pvlib':
        # pvl_ephemeris needs local time, not utc time.
        ephem_df = _pvlib(time, location, pressure, temperature)
    elif method == 'pyephem':
        ephem_df = _ephem(time_utc, location, pressure, temperature)
    elif method == 'pysolar':
        ephem_df = _pysolar(time_utc, location)
    else:
        raise ValueError('Invalid method')
    
    try:
        return ephem_df.tz_convert(location.tz)
    except TypeError:
        return ephem_df.tz_localize(location.tz)



def _pvlib(time, location, pressure, temperature):
    """
    Calculate the solar position using PVLIB's ephemeris code.
    """
    # Will H: I suggest putting the ephemeris code inline here once it's fixed.
    
    pvl_logger.debug('using pvlib internal ephemeris code to calculate solar position')
    
    return pvl_ephemeris.pvl_ephemeris(time, location, pressure, temperature)



def _ephem(time, location, pressure, temperature):
    """
    Calculate the solar position using the PyEphem package.
    """
    
    pvl_logger.debug('using PyEphem to calculate solar position')
    
    sun_coords = pd.DataFrame(index=time)
    
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
    
    return sun_coords
    
