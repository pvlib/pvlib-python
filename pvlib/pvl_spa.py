import numpy as np
from scipy.io import loadmat,savemat
import os
import sys
from pvl_tools import * #load all of pvl_tools into namespace
import pandas as pd



def pvl_spa(Time,Location):
    '''
    Calculate the solar position using the C implementation of the NREL 
    SPA code 

    The source files for this code are located in './spa_c_files/', along with
    a README file which describes how the C code is wrapped in Python. 

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

    SunAz : DataFrame 
        Azimuth of the sun in decimal degrees from North. 0 = North to 270 = West
  
    SunEl : DataFrame
        Actual elevation (not accounting for refraction)of the sun 
        in decimal degrees, 0 = on horizon. The complement of the True Zenith
        Angle.
    

    References
    ----------

    NREL SPA code: http://rredc.nrel.gov/solar/codesandalgorithms/spa/
    
    '''
    
    sys.path.append(os.path.abspath('spa_c_files/'))
    
    import spa_py #import the Cython version of the C compiled NREL SPA algorithm

    try: 
        Timeshifted=Time.tz_convert('UTC') #This will work with a timezone aware dataset
    except:
        Timeshifted=Time.shift(abs(Location['TZ']),freq='H') #This will work with a timezone unaware dataset

    spa_out=[]
    
    for date in Timeshifted:
        spa_out.append(spa_py.spa_calc(year=date.year,
                        month=date.month,
                        day=date.day,
                        hour=date.hour,
                        minute=date.minute,
                        second=date.second,
                        timezone=0, #timezone corrections handled above
                        latitude=Location['latitude'],
                        longitude=Location['longitude'],
                        elevation=Location['altitude']))
    
    DFOut=pd.DataFrame(spa_out,index=Time)
    
    DFOut['SunEl']=90-DFOut.zenith
    

    return DFOut['azimuth180'],DFOut['SunEl'],DFOut['zenith']





