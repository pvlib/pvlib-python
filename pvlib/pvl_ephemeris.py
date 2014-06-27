import logging
pvl_logger = logging.getLogger('pvlib')

import pdb

import pytz
import numpy as np
import pandas as pd

import pvl_tools



def pvl_ephemeris(time, location, pressure=101325, temperature=12):
    ''' 
    Calculates the position of the sun given time, location, and optionally pressure and temperature


    Parameters
    ----------
    Time :  `pandas.Index <http://pandas.pydata.org/pandas-docs/version/0.13.1/generated/pandas.Index.html>`_

    Location: struct

        *Location.latitude* - vector or scalar latitude in decimal degrees (positive is
                              northern hemisphere)
        *Location.longitude* - vector or scalar longitude in decimal degrees (positive is 
                              east of prime meridian)
        *Location.altitude* - an optional component of the Location struct, not
                              used in the ephemeris code directly, but it may be used to calculate
                              standard site pressure (see pvl_alt2pres function)
        *location.TZ*     - Time Zone offset from UTC 

    Other Parameters
    ----------------

    pressure : float or DataFrame
          Ambient pressure (Pascals)

    tempreature: float or DataFrame
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
    #SunZen[SunZen >= 90] = 90 # Will H: This is not the appropriate place to put this restriction. Maybe I want this information.

    ApparentSunEl = SunEl + Refract

    DFOut = pd.DataFrame({'elevation':SunEl}, index=time)
    DFOut['azimuth'] = SunAz-180  #Changed RA Feb 18,2014 to match Duffe
    DFOut['zenith'] = SunZen
    DFOut['apparent_elevation'] = ApparentSunEl
    DFOut['apparent_zenith'] = 90 - ApparentSunEl
    DFOut['solar_time'] = SolarTime

    return DFOut
