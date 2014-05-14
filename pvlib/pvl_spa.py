import numpy as np
from scipy.io import loadmat,savemat
import os
from pvl_tools import * #load all of pvl_tools into namespace
import Pysolar 
import pandas as pd



def pvl_spa(Time,Location):
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

    SunAz : DataFrame 
        Azimuth of the sun in decimal degrees from North. 0 = North to 270 = West
  
    SunEl : DataFrame
        Actual elevation (not accounting for refraction)of the sun 
        in decimal degrees, 0 = on horizon. The complement of the True Zenith
        Angle.
    

    References
    ----------

    PySolar Documentation: https://github.com/pingswept/pysolar/tree/master
    '''
    pdb.set_trace()
    try: 
        Timeshifted=Time.tz_convert('UTC')
    except:
        Timeshifted=Time.shift(abs(Location.TZ),freq='H')

    SunAz=map(lambda x: Pysolar.GetAzimuth(Location.latitude,Location.longitude,x),Timeshifted)#.tz_convert('UTC'))
    SunEl=map(lambda x: Pysolar.GetAltitude(Location.latitude,Location.longitude,x),Timeshifted)#.tz_convert('UTC'))
    SunEl[SunEl<0]=0
    Zen=90-np.array(SunEl)

    SunAz=(SunAz+360)*-1
    SunAz[SunAz<-180]=SunAz+360

    DFOut=pd.DataFrame({'SunAz':SunAz,'SunEl':SunEl,'SunZen':Zen},index=Time)

    return DFOut['SunAz'],DFOut['SunEl'],DFOut['SunZen']





