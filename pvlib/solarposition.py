"""
Calculate the solar position using a variety of methods/packages.
"""

# Contributors:
# Rob Andrews (@Calama-Consulting), Calama Consulting, 2014 
# Will Holmgren (@wholmgren), University of Arizona, 2014 

from __future__ import division
import logging
pvl_logger = logging.getLogger('pvlib')
import datetime as dt


import numpy as np
import pandas as pd
import pytz
    

from pvlib.tools import localize_to_utc, datetime_to_djd, djd_to_datetime


def get_solarposition(time, location, method='pyephem', pressure=101325, 
                      temperature=12):
    """
    A convenience wrapper for the solar position calculators.
    
    Parameters
    ----------
    time : pandas.DatetimeIndex
    location : pvlib.Location object
    method : string
        'pyephem' uses the PyEphem package. Default.
        'spa' uses the pvlib ephemeris code.
    pressure : float
        Pascals.
    temperature : float 
        Degrees C.
    """
    
    method = method.lower()
    if isinstance(time, dt.datetime):
        time = pd.DatetimeIndex([time,])
    
    if method == 'spa':
        ephem_df = spa(time, location)
    elif method == 'pyephem':
        ephem_df = pyephem(time, location, pressure, temperature)
    else:
        raise ValueError('Invalid solar position method')
        
    return ephem_df


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
    
    # Added by Rob Andrews (@Calama-Consulting), Calama Consulting, 2014 
    # Edited by Will Holmgren (@wholmgren), University of Arizona, 2014 

    try:
        from pvlib.spa_c_files.spa_py import spa_calc
    except ImportError as e:
        raise ImportError('Could not import built-in SPA calculator. '+
                          'You may need to recompile the SPA code.')
    
    pvl_logger.debug('using built-in spa code to calculate solar position')
    
    pvl_logger.warning('Using solarposition.spa is currently liscenced for ' \
                       'non-commercial purposes only, according to the NREL SPA' \
                       'documentation at http://www.nrel.gov/midc/spa/#agree')
    time_utc = localize_to_utc(time, location)
        
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


def _ephem_setup(location, pressure, temperature):
    import ephem
    # initialize a PyEphem observer
    obs = ephem.Observer()
    obs.lat = str(location.latitude)
    obs.lon = str(location.longitude)
    obs.elevation = location.altitude
    obs.pressure = pressure / 100. # convert to mBar
    obs.temp = temperature
    
    # the PyEphem sun
    sun = ephem.Sun()
    return obs, sun


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
    
    # Written by Will Holmgren (@wholmgren), University of Arizona, 2014 

    import ephem

    pvl_logger.debug('using PyEphem to calculate solar position')
    
    time_utc = localize_to_utc(time, location)
    
    sun_coords = pd.DataFrame(index=time_utc)
    
    obs, sun = _ephem_setup(location, pressure, temperature)

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

        
def ephemeris(time, location, pressure=101325, temperature=12):
    ''' 
    Python-native solar position calculator.
    The accuracy of this code is not guaranteed. 
    Consider using the built-in spa_c code or the PyEphem library.

    Parameters
    ----------
    time : pandas.DatetimeIndex
    location : pvlib.Location

    Other Parameters
    ----------------

    pressure : float or DataFrame
          Ambient pressure (Pascals)

    temperature : float or DataFrame
          Ambient temperature (C)
      
    Returns
    -------
    
    DataFrame with the following columns
    
    SunEl : float of DataFrame
        Actual elevation (not accounting for refraction)of the sun 
        in decimal degrees, 0 = on horizon. The complement of the True Zenith
        Angle.
        
    SunAz : Azimuth of the sun in decimal degrees from North. 0 = North to 270 = West
    
    SunZen : Solar zenith angle

    ApparentSunEl : float or DataFrame

        Apparent sun elevation accounting for atmospheric 
        refraction. This is the complement of the Apparent Zenith Angle.

    SolarTime : fload or DataFrame
        Solar time in decimal hours (solar noon is 12.00).
        

    References
    -----------

    Grover Hughes' class and related class materials on Engineering 
    Astronomy at Sandia National Laboratories, 1985.

    See also
    --------
    pvl_makelocationstruct
    pvl_alt2pres
    pvl_getaoi 
    pvl_spa

    '''
    
    # Added by Rob Andrews (@Calama-Consulting), Calama Consulting, 2014 
    # Edited by Will Holmgren (@wholmgren), University of Arizona, 2014 
    
    pvl_logger.warning('Using solarposition.ephemeris is discouraged. ' \
                       'solarposition.pyephem and solarposition.spa are ' \
                       'faster and more accurate.')
    
    pvl_logger.debug('location={}, temperature={}, pressure={}'.format(location, temperature, pressure))

    Latitude = location.latitude
    ''' the inversion of longitude is due to the fact that this code was
    originally written for the convention that positive longitude were for
    locations west of the prime meridian. However, the correct convention (as
    of 2009) is to use negative longitudes for locations west of the prime
    meridian. Therefore, the user should input longitude values under the
    correct convention (e.g. Albuquerque is at -106 longitude), but it needs
    to be inverted for use in the code.
    '''
    Latitude = location.latitude
    Longitude=1 * location.longitude
    Year = time.year
    Month = time.month
    Day = time.day
    Hour = time.hour
    Minute = time.minute
    Second = time.second
    DayOfYear = time.dayofyear

    DecHours = Hour + Minute / float(60) + Second / float(3600)

    Abber=20 / float(3600)
    LatR=np.radians(Latitude)
    
    # this code is needed to handle the new location.tz format string
    try:
        utc_offset = pytz.timezone(location.tz).utcoffset(time[0]).total_seconds() / 3600.
    except ValueError:
        utc_offset = time.tzinfo.utcoffset(time[0]).total_seconds() / 3600.
    pvl_logger.debug('utc_offset={}'.format(utc_offset))
    
    UnivDate=DayOfYear 
    UnivHr = DecHours + utc_offset - .5 # Will H: surprised to see the 0.5 here, but moving on...
    #+60/float(60)/2

    Yr=Year - 1900

    YrBegin=365 * Yr + np.floor((Yr - 1) / float(4)) - 0.5

    Ezero=YrBegin + UnivDate
    T=Ezero / float(36525)
    GMST0=6 / float(24) + 38 / float(1440) + (45.836 + 8640184.542 * T + 0.0929 * T ** 2) / float(86400)
    GMST0=360 * (GMST0 - np.floor(GMST0))
    GMSTi=np.mod(GMST0 + 360 * (1.0027379093 * UnivHr / float(24)),360)

    LocAST=np.mod((360 + GMSTi - Longitude),360)
    EpochDate=Ezero + UnivHr / float(24)
    T1=EpochDate / float(36525)
    ObliquityR=np.radians(23.452294 - 0.0130125 * T1 - 1.64e-06 * T1 ** 2 + 5.03e-07 * T1 ** 3)
    MlPerigee=281.22083 + 4.70684e-05 * EpochDate + 0.000453 * T1 ** 2 + 3e-06 * T1 ** 3
    MeanAnom=np.mod((358.47583 + 0.985600267 * EpochDate - 0.00015 * T1 ** 2 - 3e-06 * T1 ** 3),360)
    Eccen=0.01675104 - 4.18e-05 * T1 - 1.26e-07 * T1 ** 2
    EccenAnom=MeanAnom
    E=0

    while np.max(abs(EccenAnom - E)) > 0.0001:
        E=EccenAnom
        EccenAnom=MeanAnom + np.degrees(Eccen)*(np.sin(np.radians(E)))
    
    #pdb.set_trace()     
    TrueAnom=2 * np.mod(np.degrees(np.arctan2(((1 + Eccen) / (1 - Eccen)) ** 0.5*np.tan(np.radians(EccenAnom) / float(2)),1)),360)
    EcLon=np.mod(MlPerigee + TrueAnom,360) - Abber
    EcLonR=np.radians(EcLon)
    DecR=np.arcsin(np.sin(ObliquityR)*(np.sin(EcLonR)))
    Dec=np.degrees(DecR)
    #pdb.set_trace()

    RtAscen=np.degrees(np.arctan2(np.cos(ObliquityR)*((np.sin(EcLonR))),np.cos(EcLonR)))

    HrAngle=LocAST - RtAscen
    HrAngleR=np.radians(HrAngle)

    HrAngle=HrAngle - (360*((abs(HrAngle) > 180)))
    SunAz=np.degrees(np.arctan2(- 1 * np.sin(HrAngleR),np.cos(LatR)*(np.tan(DecR)) - np.sin(LatR)*(np.cos(HrAngleR))))
    SunAz=SunAz + (SunAz < 0) * 360
    SunEl=np.degrees(np.arcsin((np.cos(LatR)*(np.cos(DecR))*(np.cos(HrAngleR)) + np.sin(LatR)*(np.sin(DecR))))) #potential error
    SolarTime=(180 + HrAngle) / float(15)

    Refract=[]

    for Elevation in SunEl:
        TanEl=np.tan(np.radians(Elevation))
        if Elevation>5 and Elevation<=85:
            Refract.append((58.1 / float(TanEl) - 0.07 / float(TanEl ** 3) + 8.6e-05 / float(TanEl ** 5)))
        elif Elevation > -0.575 and Elevation <=5:
            Refract.append((Elevation*((- 518.2 + Elevation*((103.4 + Elevation*((- 12.79 + Elevation*(0.711))))))) + 1735))
        elif Elevation> -1 and Elevation<= -0.575:
            Refract.append(- 20.774 / float(TanEl))
        else:
            Refract.append(0)


    Refract=np.array(Refract)*((283 / float(273 + temperature)))*(pressure) / float(101325) / float(3600)

    SunZen = 90 - SunEl
    #SunZen[SunZen >= 90] = 90 
    
    ApparentSunEl = SunEl + Refract

    DFOut = pd.DataFrame({'elevation':SunEl}, index=time)
    DFOut['azimuth'] = SunAz-180  #Changed RA Feb 18,2014 to match Duffe
    DFOut['zenith'] = SunZen
    DFOut['apparent_elevation'] = ApparentSunEl
    DFOut['apparent_zenith'] = 90 - ApparentSunEl
    DFOut['solar_time'] = SolarTime

    return DFOut


def calc_time(lower_bound, upper_bound, location, attribute, value, 
              pressure=101325, temperature=12, xtol=1.0e-12):
    """
    Calculate the time between lower_bound and upper_bound
    where the attribute is equal to value. Uses PyEphem for
    solar position calculations.
    
    Parameters
    ----------
    lower_bound : datetime.datetime 
    upper_bound : datetime.datetime 
    location : pvlib.Location object
    attribute : str
        The attribute of a pyephem.Sun object that
        you want to solve for. Likely options are 'alt'
        and 'az' (which must be given in radians).
    value : int or float
        The value of the attribute to solve for
    pressure : int or float, optional
        Air pressure in Pascals. Set to 0 for no
        atmospheric correction.
    temperature : int or float, optional
        Air temperature in degrees C.
    xtol : float, optional
        The allowed error in the result from value
    
    Returns
    -------
    datetime.datetime

    Raises
    ------
    ValueError
        If the value is not contained between the bounds.
    AttributeError
        If the given attribute is not an attribute of a
        PyEphem.Sun object.
    """

    try:
        import scipy.optimize as so
    except ImportError as e:
        raise ImportError('The calc_time function requires scipy')

    obs, sun = _ephem_setup(location, pressure, temperature)

    def compute_attr(thetime, target, attr):
        obs.date = thetime
        sun.compute(obs)
        return getattr(sun, attr) - target
        
    lb = datetime_to_djd(lower_bound)
    ub = datetime_to_djd(upper_bound)

    djd_root = so.brentq(compute_attr, lb, ub, 
                         (value, attribute), xtol=xtol)

    return djd_to_datetime(djd_root, location.tz)
    
    
    
def pyephem_earthsun_distance(time):
    """
    Calculates the distance from the earth to the sun using pyephem.
    
    Parameters
    ----------
    time : pd.DatetimeIndex
    
    Returns
    -------
    pd.Series. Earth-sun distance in AU.
    """
    pvl_logger.debug('solarposition.pyephem_earthsun_distance()')
    
    import ephem
    
    sun = ephem.Sun()
    earthsun = []
    for thetime in time:
        sun.compute(ephem.Date(thetime))
        earthsun.append(sun.earth_distance)
        
    return pd.Series(earthsun, index=time)




