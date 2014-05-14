

import numpy as np
import pvl_tools as pvt
import pdb

def pvl_makelocationstruct(latitude,longitude,TZ,altitude=100):
  '''
  Create a struct to define a site location

  Parameters
  ----------

  Latitude : float
            Positive north of equator, decimal notation
  Longitude : float 
            Positive east of prime meridian, decimal notation
  TZ  : int
            Timezone in GMT offset

  Other Parameters
  ----------------

  altitude : float (optional, default=100)
            Altitude from sea level. Set to 100m if none input

  Returns
  -------

  Location : struct

          *Location.latitude*

          *Location.longitude*

          *Location.TZ*

          *Location.altitude*


  See Also
  --------

  pvl_ephemeris
  pvl_alt2pres
  pvl_pres2alt

  '''

  Vars=locals()
  Expect={'latitude':('num','x>=-90','x<=90'),
          'longitude': ('num','x<=180','x>=-180'),
          'altitude':('num','default','default=100'),
          'TZ':('num')
          }
  Location=pvt.Parse(Vars,Expect)


  return Location
