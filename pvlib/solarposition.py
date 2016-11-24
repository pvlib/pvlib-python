"""
Calculate the solar position using a variety of methods/packages.
"""

# Contributors:
# Rob Andrews (@Calama-Consulting), Calama Consulting, 2014
# Will Holmgren (@wholmgren), University of Arizona, 2014
# Tony Lorenzo (@alorenzo175), University of Arizona, 2015

from __future__ import division
import os
import datetime as dt
try:
    from importlib import reload
except ImportError:
    try:
        from imp import reload
    except ImportError:
        pass

import numpy as np
import pandas as pd

from pvlib import atmosphere
from pvlib.tools import datetime_to_djd, djd_to_datetime

import logging
pvl_logger = logging.getLogger('pvlib')


def get_solarposition(time, latitude, longitude,
                      altitude=None, pressure=None,
                      method='nrel_numpy',
                      temperature=12, **kwargs):
    """
    A convenience wrapper for the solar position calculators.

    Parameters
    ----------
    time : pandas.DatetimeIndex
    latitude : float
    longitude : float
    altitude : None or float
        If None, computed from pressure. Assumed to be 0 m
        if pressure is also None.
    pressure : None or float
        If None, computed from altitude. Assumed to be 101325 Pa
        if altitude is also None.
    method : string
        'pyephem' uses the PyEphem package: :func:`pyephem`

        'nrel_c' uses the NREL SPA C code [3]: :func:`spa_c`

        'nrel_numpy' uses an implementation of the NREL SPA algorithm
        described in [1] (default): :func:`spa_python`

        'nrel_numba' uses an implementation of the NREL SPA algorithm
        described in [1], but also compiles the code first: :func:`spa_python`

        'ephemeris' uses the pvlib ephemeris code: :func:`ephemeris`
    temperature : float
        Degrees C.

    Other keywords are passed to the underlying solar position function.

    References
    ----------
    [1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
    applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.

    [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.

    [3] NREL SPA code: http://rredc.nrel.gov/solar/codesandalgorithms/spa/
    """

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    method = method.lower()
    if isinstance(time, dt.datetime):
        time = pd.DatetimeIndex([time, ])

    if method == 'nrel_c':
        ephem_df = spa_c(time, latitude, longitude, pressure, temperature,
                         **kwargs)
    elif method == 'nrel_numba':
        ephem_df = spa_python(time, latitude, longitude, altitude,
                              pressure, temperature,
                              how='numba', **kwargs)
    elif method == 'nrel_numpy':
        ephem_df = spa_python(time, latitude, longitude, altitude,
                              pressure, temperature,
                              how='numpy', **kwargs)
    elif method == 'pyephem':
        ephem_df = pyephem(time, latitude, longitude, pressure, temperature,
                           **kwargs)
    elif method == 'ephemeris':
        ephem_df = ephemeris(time, latitude, longitude, pressure, temperature,
                             **kwargs)
    else:
        raise ValueError('Invalid solar position method')

    return ephem_df


def spa_c(time, latitude, longitude, pressure=101325, altitude=0,
          temperature=12, delta_t=67.0,
          raw_spa_output=False):
    """
    Calculate the solar position using the C implementation of the NREL
    SPA code

    The source files for this code are located in './spa_c_files/', along with
    a README file which describes how the C code is wrapped in Python.
    Due to license restrictions, the C code must be downloaded seperately
    and used in accordance with it's license.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Localized or UTC.
    latitude : float
    longitude : float
    pressure : float
        Pressure in Pascals
    altitude : float
        Elevation above sea level.
    temperature : float
        Temperature in C
    delta_t : float
        Difference between terrestrial time and UT1.
        USNO has previous values and predictions.
    raw_spa_output : bool
        If true, returns the raw SPA output.

    Returns
    -------
    DataFrame
        The DataFrame will have the following columns:
        elevation,
        azimuth,
        zenith,
        apparent_elevation,
        apparent_zenith.

    References
    ----------
    NREL SPA code: http://rredc.nrel.gov/solar/codesandalgorithms/spa/

    USNO delta T:
    http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term

    See also
    --------
    pyephem, spa_python, ephemeris
    """

    # Added by Rob Andrews (@Calama-Consulting), Calama Consulting, 2014
    # Edited by Will Holmgren (@wholmgren), University of Arizona, 2014
    # Edited by Tony Lorenzo (@alorenzo175), University of Arizona, 2015

    try:
        from pvlib.spa_c_files.spa_py import spa_calc
    except ImportError:
        raise ImportError('Could not import built-in SPA calculator. ' +
                          'You may need to recompile the SPA code.')

    pvl_logger.debug('using built-in spa code to calculate solar position')

    time_utc = time

    spa_out = []

    for date in time_utc:
        spa_out.append(spa_calc(year=date.year,
                                month=date.month,
                                day=date.day,
                                hour=date.hour,
                                minute=date.minute,
                                second=date.second,
                                timezone=0,  # must input localized or utc time
                                latitude=latitude,
                                longitude=longitude,
                                elevation=altitude,
                                pressure=pressure / 100,
                                temperature=temperature,
                                delta_t=delta_t
                                ))

    spa_df = pd.DataFrame(spa_out, index=time_utc)

    if raw_spa_output:
        return spa_df
    else:
        dfout = pd.DataFrame({'azimuth': spa_df['azimuth'],
                              'apparent_zenith': spa_df['zenith'],
                              'apparent_elevation': spa_df['e'],
                              'elevation': spa_df['e0'],
                              'zenith': 90 - spa_df['e0']})

        return dfout


def _spa_python_import(how):
    """Compile spa.py appropriately"""

    from pvlib import spa

    # check to see if the spa module was compiled with numba
    using_numba = spa.USE_NUMBA

    if how == 'numpy' and using_numba:
        # the spa module was compiled to numba code, so we need to
        # reload the module without compiling
        # the PVLIB_USE_NUMBA env variable is used to tell the module
        # to not compile with numba
        os.environ['PVLIB_USE_NUMBA'] = '0'
        pvl_logger.debug('Reloading spa module without compiling')
        spa = reload(spa)
        del os.environ['PVLIB_USE_NUMBA']
    elif how == 'numba' and not using_numba:
        # The spa module was not compiled to numba code, so set
        # PVLIB_USE_NUMBA so it does compile to numba on reload.
        os.environ['PVLIB_USE_NUMBA'] = '1'
        pvl_logger.debug('Reloading spa module, compiling with numba')
        spa = reload(spa)
        del os.environ['PVLIB_USE_NUMBA']
    elif how != 'numba' and how != 'numpy':
        raise ValueError("how must be either 'numba' or 'numpy'")

    return spa


def spa_python(time, latitude, longitude,
               altitude=0, pressure=101325, temperature=12, delta_t=67.0,
               atmos_refract=None, how='numpy', numthreads=4, **kwargs):
    """
    Calculate the solar position using a python implementation of the
    NREL SPA algorithm described in [1].

    If numba is installed, the functions can be compiled to
    machine code and the function can be multithreaded.
    Without numba, the function evaluates via numpy with
    a slight performance hit.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Localized or UTC.
    latitude : float
    longitude : float
    altitude : float
    pressure : int or float, optional
        avg. yearly air pressure in Pascals.
    temperature : int or float, optional
        avg. yearly air temperature in degrees C.
    delta_t : float, optional
        If delta_t is None, uses spa.calculate_deltat
        using time.year and time.month from pandas.DatetimeIndex.
        For most simulations specifing delta_t is sufficient.
        Difference between terrestrial time and UT1.
        *Note: delta_t = None will break code using nrel_numba,
        this will be fixed in a future version.
        The USNO has historical and forecasted delta_t [3].
    atmos_refrac : float, optional
        The approximate atmospheric refraction (in degrees)
        at sunrise and sunset.
    how : str, optional
        Options are 'numpy' or 'numba'. If numba >= 0.17.0
        is installed, how='numba' will compile the spa functions
        to machine code and run them multithreaded.
    numthreads : int, optional
        Number of threads to use if how == 'numba'.

    Returns
    -------
    DataFrame
        The DataFrame will have the following columns:
        apparent_zenith (degrees),
        zenith (degrees),
        apparent_elevation (degrees),
        elevation (degrees),
        azimuth (degrees),
        equation_of_time (minutes).


    References
    ----------
    [1] I. Reda and A. Andreas, Solar position algorithm for solar
    radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.

    [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.

    [3] USNO delta T:
    http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term

    See also
    --------
    pyephem, spa_c, ephemeris
    """

    # Added by Tony Lorenzo (@alorenzo175), University of Arizona, 2015

    pvl_logger.debug('Calculating solar position with spa_python code')

    lat = latitude
    lon = longitude
    elev = altitude
    pressure = pressure / 100  # pressure must be in millibars for calculation

    atmos_refract = atmos_refract or 0.5667

    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    unixtime = np.array(time.astype(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    app_zenith, zenith, app_elevation, elevation, azimuth, eot = \
        spa.solar_position(unixtime, lat, lon, elev, pressure, temperature,
                           delta_t, atmos_refract, numthreads)

    result = pd.DataFrame({'apparent_zenith': app_zenith, 'zenith': zenith,
                           'apparent_elevation': app_elevation,
                           'elevation': elevation, 'azimuth': azimuth,
                           'equation_of_time': eot},
                          index=time)

    return result


def get_sun_rise_set_transit(time, latitude, longitude, how='numpy',
                             delta_t=67.0,
                             numthreads=4):
    """
    Calculate the sunrise, sunset, and sun transit times using the
    NREL SPA algorithm described in [1].

    If numba is installed, the functions can be compiled to
    machine code and the function can be multithreaded.
    Without numba, the function evaluates via numpy with
    a slight performance hit.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Only the date part is used
    latitude : float
    longitude : float
    delta_t : float, optional
        If delta_t is None, uses spa.calculate_deltat
        using time.year and time.month from pandas.DatetimeIndex.
        For most simulations specifing delta_t is sufficient.
        Difference between terrestrial time and UT1.
        *Note: delta_t = None will break code using nrel_numba,
        this will be fixed in a future version.
        By default, use USNO historical data and predictions
    how : str, optional
        Options are 'numpy' or 'numba'. If numba >= 0.17.0
        is installed, how='numba' will compile the spa functions
        to machine code and run them multithreaded.
    numthreads : int, optional
        Number of threads to use if how == 'numba'.

    Returns
    -------
    DataFrame
        The DataFrame will have the following columns:
        sunrise, sunset, transit

    References
    ----------
    [1] Reda, I., Andreas, A., 2003. Solar position algorithm for solar
    radiation applications. Technical report: NREL/TP-560- 34302. Golden,
    USA, http://www.nrel.gov.
    """
    # Added by Tony Lorenzo (@alorenzo175), University of Arizona, 2015

    pvl_logger.debug('Calculating sunrise, set, transit with spa_python code')

    lat = latitude
    lon = longitude

    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    # must convert to midnight UTC on day of interest
    utcday = pd.DatetimeIndex(time.date).tz_localize('UTC')
    unixtime = np.array(utcday.astype(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    transit, sunrise, sunset = spa.transit_sunrise_sunset(
        unixtime, lat, lon, delta_t, numthreads)

    # arrays are in seconds since epoch format, need to conver to timestamps
    transit = pd.to_datetime(transit*1e9, unit='ns', utc=True).tz_convert(
        time.tz).tolist()
    sunrise = pd.to_datetime(sunrise*1e9, unit='ns', utc=True).tz_convert(
        time.tz).tolist()
    sunset = pd.to_datetime(sunset*1e9, unit='ns', utc=True).tz_convert(
        time.tz).tolist()

    result = pd.DataFrame({'transit': transit,
                           'sunrise': sunrise,
                           'sunset': sunset}, index=time)

    return result


def _ephem_setup(latitude, longitude, altitude, pressure, temperature):
    import ephem
    # initialize a PyEphem observer
    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.elevation = altitude
    obs.pressure = pressure / 100.  # convert to mBar
    obs.temp = temperature

    # the PyEphem sun
    sun = ephem.Sun()
    return obs, sun


def pyephem(time, latitude, longitude, altitude=0, pressure=101325,
            temperature=12):
    """
    Calculate the solar position using the PyEphem package.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Localized or UTC.
    latitude : float
    longitude : float
    altitude : float
        distance above sea level.
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

    See also
    --------
    spa_python, spa_c, ephemeris
    """

    # Written by Will Holmgren (@wholmgren), University of Arizona, 2014
    try:
        import ephem
    except ImportError:
        raise ImportError('PyEphem must be installed')

    pvl_logger.debug('using PyEphem to calculate solar position')

    # if localized, convert to UTC. otherwise, assume UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    sun_coords = pd.DataFrame(index=time)

    obs, sun = _ephem_setup(latitude, longitude, altitude,
                            pressure, temperature)

    # make and fill lists of the sun's altitude and azimuth
    # this is the pressure and temperature corrected apparent alt/az.
    alts = []
    azis = []
    for thetime in time_utc:
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
    for thetime in time_utc:
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


def ephemeris(time, latitude, longitude, pressure=101325, temperature=12):
    """
    Python-native solar position calculator.
    The accuracy of this code is not guaranteed.
    Consider using the built-in spa_c code or the PyEphem library.

    Parameters
    ----------
    time : pandas.DatetimeIndex
    latitude : float
    longitude : float
    pressure : float or Series
        Ambient pressure (Pascals)
    temperature : float or Series
        Ambient temperature (C)

    Returns
    -------

    DataFrame with the following columns:

        * apparent_elevation : apparent sun elevation accounting for
          atmospheric refraction.
        * elevation : actual elevation (not accounting for refraction)
          of the sun in decimal degrees, 0 = on horizon.
          The complement of the zenith angle.
        * azimuth : Azimuth of the sun in decimal degrees East of North.
          This is the complement of the apparent zenith angle.
        * apparent_zenith : apparent sun zenith accounting for atmospheric
          refraction.
        * zenith : Solar zenith angle
        * solar_time : Solar time in decimal hours (solar noon is 12.00).

    References
    -----------

    Grover Hughes' class and related class materials on Engineering
    Astronomy at Sandia National Laboratories, 1985.

    See also
    --------
    pyephem, spa_c, spa_python

    """

    # Added by Rob Andrews (@Calama-Consulting), Calama Consulting, 2014
    # Edited by Will Holmgren (@wholmgren), University of Arizona, 2014

    # Most comments in this function are from PVLIB_MATLAB or from
    # pvlib-python's attempt to understand and fix problems with the
    # algorithm. The comments are *not* based on the reference material.
    # This helps a little bit:
    # http://www.cv.nrao.edu/~rfisher/Ephemerides/times.html

    # the inversion of longitude is due to the fact that this code was
    # originally written for the convention that positive longitude were for
    # locations west of the prime meridian. However, the correct convention (as
    # of 2009) is to use negative longitudes for locations west of the prime
    # meridian. Therefore, the user should input longitude values under the
    # correct convention (e.g. Albuquerque is at -106 longitude), but it needs
    # to be inverted for use in the code.

    Latitude = latitude
    Longitude = -1 * longitude

    Abber = 20 / 3600.
    LatR = np.radians(Latitude)

    # the SPA algorithm needs time to be expressed in terms of
    # decimal UTC hours of the day of the year.

    # if localized, convert to UTC. otherwise, assume UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    # strip out the day of the year and calculate the decimal hour
    DayOfYear = time_utc.dayofyear
    DecHours = (time_utc.hour + time_utc.minute/60. + time_utc.second/3600. +
                time_utc.microsecond/3600.e6)

    UnivDate = DayOfYear
    UnivHr = DecHours

    Yr = time_utc.year - 1900
    YrBegin = 365 * Yr + np.floor((Yr - 1) / 4.) - 0.5

    Ezero = YrBegin + UnivDate
    T = Ezero / 36525.

    # Calculate Greenwich Mean Sidereal Time (GMST)
    GMST0 = 6 / 24. + 38 / 1440. + (
        45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400.
    GMST0 = 360 * (GMST0 - np.floor(GMST0))
    GMSTi = np.mod(GMST0 + 360 * (1.0027379093 * UnivHr / 24.), 360)

    # Local apparent sidereal time
    LocAST = np.mod((360 + GMSTi - Longitude), 360)

    EpochDate = Ezero + UnivHr / 24.
    T1 = EpochDate / 36525.

    ObliquityR = np.radians(
        23.452294 - 0.0130125 * T1 - 1.64e-06 * T1 ** 2 + 5.03e-07 * T1 ** 3)
    MlPerigee = 281.22083 + 4.70684e-05 * EpochDate + 0.000453 * T1 ** 2 + (
        3e-06 * T1 ** 3)
    MeanAnom = np.mod((358.47583 + 0.985600267 * EpochDate - 0.00015 *
                       T1 ** 2 - 3e-06 * T1 ** 3), 360)
    Eccen = 0.01675104 - 4.18e-05 * T1 - 1.26e-07 * T1 ** 2
    EccenAnom = MeanAnom
    E = 0

    while np.max(abs(EccenAnom - E)) > 0.0001:
        E = EccenAnom
        EccenAnom = MeanAnom + np.degrees(Eccen)*np.sin(np.radians(E))

    TrueAnom = (
        2 * np.mod(np.degrees(np.arctan2(((1 + Eccen) / (1 - Eccen)) ** 0.5 *
                   np.tan(np.radians(EccenAnom) / 2.), 1)), 360))
    EcLon = np.mod(MlPerigee + TrueAnom, 360) - Abber
    EcLonR = np.radians(EcLon)
    DecR = np.arcsin(np.sin(ObliquityR)*np.sin(EcLonR))

    RtAscen = np.degrees(np.arctan2(np.cos(ObliquityR)*np.sin(EcLonR),
                                    np.cos(EcLonR)))

    HrAngle = LocAST - RtAscen
    HrAngleR = np.radians(HrAngle)
    HrAngle = HrAngle - (360 * ((abs(HrAngle) > 180)))

    SunAz = np.degrees(np.arctan2(-np.sin(HrAngleR),
                                  np.cos(LatR)*np.tan(DecR) -
                                  np.sin(LatR)*np.cos(HrAngleR)))
    SunAz[SunAz < 0] += 360

    SunEl = np.degrees(np.arcsin(
        np.cos(LatR) * np.cos(DecR) * np.cos(HrAngleR) +
        np.sin(LatR) * np.sin(DecR)))

    SolarTime = (180 + HrAngle) / 15.

    # Calculate refraction correction
    Elevation = SunEl
    TanEl = pd.Series(np.tan(np.radians(Elevation)), index=time_utc)
    Refract = pd.Series(0, index=time_utc)

    Refract[(Elevation > 5) & (Elevation <= 85)] = (
        58.1/TanEl - 0.07/(TanEl**3) + 8.6e-05/(TanEl**5))

    Refract[(Elevation > -0.575) & (Elevation <= 5)] = (
        Elevation *
        (-518.2 + Elevation*(103.4 + Elevation*(-12.79 + Elevation*0.711))) +
        1735)

    Refract[(Elevation > -1) & (Elevation <= -0.575)] = -20.774 / TanEl

    Refract *= (283/(273. + temperature)) * (pressure/101325.) / 3600.

    ApparentSunEl = SunEl + Refract

    # make output DataFrame
    DFOut = pd.DataFrame(index=time)
    DFOut['apparent_elevation'] = ApparentSunEl
    DFOut['elevation'] = SunEl
    DFOut['azimuth'] = SunAz
    DFOut['apparent_zenith'] = 90 - ApparentSunEl
    DFOut['zenith'] = 90 - SunEl
    DFOut['solar_time'] = SolarTime

    return DFOut


def calc_time(lower_bound, upper_bound, latitude, longitude, attribute, value,
              altitude=0, pressure=101325, temperature=12, xtol=1.0e-12):
    """
    Calculate the time between lower_bound and upper_bound
    where the attribute is equal to value. Uses PyEphem for
    solar position calculations.

    Parameters
    ----------
    lower_bound : datetime.datetime
    upper_bound : datetime.datetime
    latitude : float
    longitude : float
    attribute : str
        The attribute of a pyephem.Sun object that
        you want to solve for. Likely options are 'alt'
        and 'az' (which must be given in radians).
    value : int or float
        The value of the attribute to solve for
    altitude : float
        Distance above sea level.
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
    except ImportError:
        raise ImportError('The calc_time function requires scipy')

    obs, sun = _ephem_setup(latitude, longitude, altitude,
                            pressure, temperature)

    def compute_attr(thetime, target, attr):
        obs.date = thetime
        sun.compute(obs)
        return getattr(sun, attr) - target

    lb = datetime_to_djd(lower_bound)
    ub = datetime_to_djd(upper_bound)

    djd_root = so.brentq(compute_attr, lb, ub,
                         (value, attribute), xtol=xtol)

    return djd_to_datetime(djd_root)


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


def nrel_earthsun_distance(time, how='numpy', delta_t=67.0, numthreads=4):
    """
    Calculates the distance from the earth to the sun using the
    NREL SPA algorithm described in [1].

    Parameters
    ----------
    time : pd.DatetimeIndex

    how : str, optional
        Options are 'numpy' or 'numba'. If numba >= 0.17.0
        is installed, how='numba' will compile the spa functions
        to machine code and run them multithreaded.

    delta_t : float, optional
        If delta_t is None, uses spa.calculate_deltat
        using time.year and time.month from pandas.DatetimeIndex.
        For most simulations specifing delta_t is sufficient.
        Difference between terrestrial time and UT1.
        *Note: delta_t = None will break code using nrel_numba,
        this will be fixed in a future version.
        By default, use USNO historical data and predictions

    numthreads : int, optional
        Number of threads to use if how == 'numba'.

    Returns
    -------
    dist : pd.Series
        Earth-sun distance in AU.

    References
    ----------
    [1] Reda, I., Andreas, A., 2003. Solar position algorithm for solar
    radiation applications. Technical report: NREL/TP-560- 34302. Golden,
    USA, http://www.nrel.gov.
    """

    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    unixtime = np.array(time.astype(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    dist = spa.earthsun_distance(unixtime, delta_t, numthreads)

    dist = pd.Series(dist, index=time)

    return dist


def _calculate_simple_day_angle(dayofyear):
    """
    Calculates the day angle for the Earth's orbit around the Sun.

    Parameters
    ----------
    dayofyear : numeric

    Returns
    -------
    day_angle : numeric
    """
    return (2. * np.pi / 365.) * (dayofyear - 1)
