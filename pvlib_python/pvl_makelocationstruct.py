'''
 PVL_MAKELOCATIONSTRUCT Create a struct to define a site location

 Syntax
   Location = pvl_makelocationstruct(latitude, longitude, TZ)
   Location = pvl_makelocationstruct(latitude, longitude, TZ, altitude)

 Description
   Creates a location struct for use with some PVLib functions. Site
   information includes latitude, longitude, TimeZone,and optionally altitude.

   Inputs latitude and longitude should be latitude and longitude
   coordinates in decimal degrees. Latitude convention is positive north 
   of the equator. Longitude convention is positive east of the prime meridian.
   Input altitude is optional, and should be given in meters above sea
   level.
   All inputs must be scalars.

   Output is a struct, Location, consisting of Location.latitude,
   Location.longitude, Location.TZ and Location.altitude (if altitude provided). 
   All output struct components are scalars.
   

 See also PVL_EPHEMERIS  PVL_ALT2PRES PVL_PRES2ALT

'''

import numpy as np
import pvl_tools as pvt
import pdb

def pvl_makelocationstruct(latitude,longitude,TZ,altitude=100):
	Vars=locals()
	Expect={'latitude':('num','x>=-90','x<=90'),
            'longitude': ('num','x<=180','x>=-180'),
            'altitude':('num','default','default=100'),
            'TZ':('num')
            }
    Location=pvt.Parse(Vars,Expect)
    

    return Location
