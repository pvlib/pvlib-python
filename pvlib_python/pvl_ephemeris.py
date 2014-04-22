''' 
 PVL_EPHEMERIS Calculates the position of the sun given time, location, and optionally pressure and temperature

 Syntax
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location)
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location, Pressure)
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location, Pressure, Temperature)
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location, 'temperature', Temperature)

 Description  
  [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location)
      Uses the given time and location structs to give sun positions with
      the pressure assumed to be 1 atm (101325 Pa) and the temperature
      assumed to be 12 C.
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location, Pressure)
      Uses the given time and location structs with the given pressure to
      determine sun positions. The temperature is assumed to be 12C.
      Pressure must be given in Pascals (1atm = 101325 Pa). If site pressure
      is unknown but site altitude is known, a conversion function may be
      used.
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location, Pressure, Temperature)
      Uses the given time and location structs with the given pressure and
      temperature to determine sun positions. Pressure must be given in
      Pascals, and temperature must be given in C.
   [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location, 'temperature', Temperature)
      Uses the given time and location structs with the given temperature
      (in C) to determine sun positions. Default pressure is 101325 Pa.

 Input Parameters:
   Time is a pandas index object with the following elements, 

   Time.year = The year in the gregorian calendar
   Time.month = the month of the year (January = 1 to December = 12)
   Time.day = the day of the month
   Time.hour = the hour of the day
   Time.minute = the minute of the hour
   Time.second = the second of the minute
   Time.UTCOffset = the UTC offset code, using the convention
     that a positive UTC offset is for time zones east of the prime meridian
     (e.g. EST = -5)
 
   Location is a struct with the following elements 
   Location.latitude = vector or scalar latitude in decimal degrees (positive is
     northern hemisphere)
   Location.longitude = vector or scalar longitude in decimal degrees (positive is 
     east of prime meridian)
   Location.altitude = an optional component of the Location struct, not
     used in the ephemeris code directly, but it may be used to calculate
     standard site pressure (see pvl_alt2pres function)
 
 Output Parameters:
   SunAz = Azimuth of the sun in decimal degrees from North. 0 = North to 270 = West
   SunEl = Actual elevation (not accounting for refraction)of the sun 
     in decimal degrees, 0 = on horizon. The complement of the True Zenith
     Angle.
   ApparentSunEl = Apparent sun elevation accounting for atmospheric 
     refraction. This is the complement of the Apparent Zenith Angle.
   SolarTime = solar time in decimal hours (solar noon is 12.00).

 References
   Grover Hughes' class and related class materials on Engineering 
   Astronomy at Sandia National Laboratories, 1985.

 See also PVL_MAKETIMESTRUCT PVL_MAKELOCATIONSTRUCT PVL_ALT2PRES
          PVL_GETAOI PVL_SPA
'''

import numpy as np
import pvl_tools
import pandas as pd

def pvl_ephemeris(Time,Location,pressure=101325,temperature=12):

    Vars=locals()
    Expect={'pressure': ('default','default=101325','array','num','x>=0'),
            'temperature': ('default','default=12','array', 'num', 'x>=-273.15'),
            'Time':'',
            'Location':''
            }
    var=pvl_tools.Parse(Vars,Expect)


    Latitude=var.Location.latitude
    Longitude=- 1 * var.Location.longitude
    Year=var.Time.year
    Month=var.Time.month
    Day=var.Time.day
    Hour=var.Time.hour
    Minute=var.Time.minute
    Second=var.Time.second
    TZone=- 1 * var.Location.TZ
 

    DayOfYear=var.Time.dayofyear
    DecHours=Hour + Minute / 60 + Second / 3600
    Abber=20 / 3600
    LatR=np.radians(Latitude)
  

    UnivDate=DayOfYear + np.floor((DecHours + TZone) / 24)
    UnivHr=np.mod((DecHours + TZone),24)
    Yr=Year - 1900
    YrBegin=365 * Yr + np.floor((Yr - 1) / 4) - 0.5
    Ezero=YrBegin + UnivDate
    T=Ezero / 36525
    GMST0=6 / 24 + 38 / 1440 + (45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400
    GMST0=360 * (GMST0 - np.floor(GMST0))
    GMSTi=np.mod(GMST0 + 360 * (1.0027379093 * UnivHr / 24),360)
    LocAST=np.mod((360 + GMSTi - Longitude),360)
    EpochDate=Ezero + UnivHr / 24
    T1=EpochDate / 36525
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
    TrueAnom=2 * np.mod(np.degrees(np.arctan2(((1 + Eccen) / (1 - Eccen)) ** 0.5*np.tan(np.radians(EccenAnom) / 2),1)),360)
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
    SolarTime=(180 + HrAngle) / 15
    
    Refract=[]

    for Elevation in SunEl:
        TanEl=np.tan(np.radians(Elevation))
        if Elevation>5 and Elevation<=85:
            Refract.append((58.1 / TanEl - 0.07 / (TanEl ** 3) + 8.6e-05 / (TanEl ** 5)))
        elif Elevation > -0.575 and Elevation <=5:
            Refract.append((Elevation*((- 518.2 + Elevation*((103.4 + Elevation*((- 12.79 + Elevation*(0.711))))))) + 1735))
        elif Elevation> -1 and Elevation<= -0.575:
            Refract.append(- 20.774 / TanEl)
        else:
            Refract.append(0)


    Refract=np.array(Refract)*((283 / (273 + var.temperature)))*(var.pressure) / 101325 / 3600


    SunZen=90-SunEl
    SunZen[SunZen >= 90 ] = 0 

    ApparentSunEl=SunEl + Refract

    DFOut=pd.DataFrame({'SunEl':SunEl}, index=var.Time)
    DFOut['SunAz']=SunAz-180  #Changed RA Feb 18,2014 to match Duffe
    DFOut['SunZen']=SunZen
    DFOut['ApparentSunEl']=ApparentSunEl
    DFOut['SolarTime']=SolarTime

    return DFOut
    