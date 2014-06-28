import logging
pvl_logger = logging.getLogger('pvlib')

import datetime
from collections import namedtuple

import numpy as np
import pandas as pd

# optional imports.
try:
    import Pysolar 
except ImportError as e:
    pvl_logger.warning('Pysolar not found.')

try:    
    import ephem
except ImportError as e:
    pvl_logger.warning('PyEphem not found.')

import pvl_ephemeris



def get_solarposition(time, location, method='pyephem', pressure=101325, 
                      temperature=12):
    """
    Calculate the solar position using a variety of methods.
    * 'pvlib' uses the pvlib ephemeris code. Not recommended as of June 2014.
    * 'pyephem' uses the PyEphem package. Default.
    * 'pysolar' uses the Pysolar package.
    
    The returned DataFrame will be localized to location.tz.
    
    :param time: pandas.DatetimeIndex. 
    :param location: namedtuple with latitude, longitude, and tz.
    :param method: string. 'pvlib', 'pyephem', 'pysolar'
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
    
    

def _pysolar(time, location):
    '''
    Calculate the solar position using the PySolar package

    The PySolar Package is developed by Brandon Stafford, and is
    found here: https://github.com/pingswept/pysolar/tree/master

    This function will map the standard time and location structures
    onto the Pysolar function

    Parameters
    ----------

    Time: Dataframe.index

        A pandas datatime object

    Location: struct

        Standard location structure, containting:

        *Location.latitude* - vector or scalar latitude in decimal degrees (positive is
                              northern hemisphere)
        *Location.longitude* - vector or scalar longitude in decimal degrees (positive is 
                              east of prime meridian)
        *Location.altitude* - an optional component of the Location struct, not
                              used in the ephemeris code directly, but it may be used to calculate
                              standard site pressure (see pvl_alt2pres function)

    Returns
    -------
    
    DataFrame with the following columns:
    
    azimuth: 
        Azimuth of the sun in decimal degrees from North. 0 = North to 270 = West
  
    elevation:
        Actual elevation (not accounting for refraction) of the sun 
        in decimal degrees, 0 = on horizon. The complement of the True Zenith
        Angle.
        
    zenith: 90 - elevation.

    References
    ----------

    PySolar Documentation: https://github.com/pingswept/pysolar/tree/master
    '''
    
    #pdb.set_trace()

    pvl_logger.debug('using Pysolar to calculate solar position')
    
    sun_az = map(lambda x: Pysolar.GetAzimuth(location.latitude, location.longitude, x), time)
    sun_el = map(lambda x: Pysolar.GetAltitude(location.latitude, location.longitude, x), time)
    
    #sun_el[sun_el < 0] = 0 # Will H: is there a reason this is here?
    zen = 90 - np.array(sun_el)
    
    # Pysolar sets azimuth=0 when facing south. This is inconsistent with
    # most SPA conventions, including ours. So, we fix it below.
    sun_az  = (np.array(sun_az) + 360) * -1
    sun_az[sun_az < -180] = sun_az[sun_az < -180] + 360
    sun_az += 180

    df_out = pd.DataFrame({'azimuth':sun_az, 'elevation':sun_el, 'zenith':zen}, index=time)

    return df_out



# only a skeleton for now 
def test_get_solarposition():
    times = pd.date_range(start=datetime.datetime(2014,6,24), end=datetime.datetime(2014,6,26), freq='1Min')
    
    Location = namedtuple('Location', ['latitude', 'longitude', 'altitude', 'tz'])
    tus = Location(32.2, -111, 700, 'US/Arizona')
    times_localized = times.tz_localize(tus.tz)
    
    ephem_data = get_solarposition(times, tus)
    
    ephem_data = get_solarposition(times, tus, method='pvlib')
    ephem_data = get_solarposition(times, tus, method='pyephem')
    ephem_data = get_solarposition(times, tus, method='pysolar')
    
    ephem_data = get_solarposition(times_localized, tus, method='pvlib')
    ephem_data = get_solarposition(times_localized, tus, method='pyephem')
    ephem_data = get_solarposition(times_localized, tus, method='pysolar')
    
    try:
        get_solarposition(times, tus, method='invalid')
    except ValueError:
        pvl_logger.debug('invalid method properly caught')
