import numpy as np
from scipy.io import loadmat,savemat
import os
from pvl_tools import * #load all of pvl_tools into namespace
import Pysolar 
import pandas as pd



def pvl_spa(time, location):
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
    try: 
        # This will work with a timezone aware dataset
        timeshifted = time.tz_convert('UTC') 
    except TypeError:
        # This will work with a timezone unaware dataset
        # Replaced shift method since it throws away data
        # and it didn't know the difference between plus and minus offsets
        ns_offset = location.TZ*3600*1e9
        timeshifted = pd.DatetimeIndex(time.values.astype(int) - ns_offset).tz_localize('UTC')

    sun_az = map(lambda x: Pysolar.GetAzimuth(location.latitude, location.longitude, x), timeshifted)#.tz_convert('UTC'))
    sun_el = map(lambda x: Pysolar.GetAltitude(location.latitude, location.longitude, x), timeshifted)#.tz_convert('UTC'))
    
    #sun_el[sun_el < 0] = 0 # Will H: is there a reason this is here?
    zen = 90 - np.array(sun_el)
    
    # Pysolar sets azimuth=0 when facing south. This is inconsistent with
    # most SPA conventions, including ours. So, we fix it below.
    sun_az  = (np.array(sun_az) + 360) * -1
    sun_az[sun_az < -180] = sun_az[sun_az < -180] + 360

    df_out = pd.DataFrame({'azimuth':sun_az, 'elevation':sun_el, 'zenith':zen}, index=time)

    return df_out





