"""
This module contains the Location class.
"""

# Will Holmgren, University of Arizona, 2014.

import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pytz

from pvlib import solarposition

class Location(object):
    """
    Location objects are convenient containers for latitude, longitude,
    timezone, and altitude data associated with a particular 
    geographic location. You can also assign a name to a location object.
    
    Location objects have two timezone attributes: 
    
        * ``tz`` is a IANA timezone string.
        * ``pytz`` is a pytz timezone object.
        
    Location objects support the print method.
    
    Parameters
    ----------
    latitude : float.
        Positive is north of the equator.
        Use decimal degrees notation.
    longitude : float. 
        Positive is east of the prime meridian.
        Use decimal degrees notation.
    tz : string or pytz.timezone. 
        See 
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        for a list of valid time zones.
        pytz.timezone objects will be converted to strings.
    alitude : float. 
        Altitude from sea level in meters.
    name : None or string. 
        Sets the name attribute of the Location object.
        
    See also
    --------
    pvsystem.PVSystem
    """
    
    def __init__(self, latitude, longitude, tz='UTC', altitude=0,
                 name=None):

        pvl_logger.debug('creating Location object')
        
        self.latitude = latitude
        self.longitude = longitude
        
        if isinstance(tz, str):
            self.tz = tz
            self.pytz = pytz.timezone(tz)
        elif isinstance(tz, datetime.tzinfo):
            self.tz = tz.zone
            self.pytz = tz
        else:
            raise TypeError('Invalid tz specification')
        
        self.altitude = altitude
        
        self.name = name
        
        
        
    def __str__(self):
        return ('{}: latitude={}, longitude={}, tz={}, altitude={}'
                .format(self.name, self.latitude, self.longitude, 
                        self.tz, self.altitude))
    
    
    @classmethod
    def from_tmy(cls, tmy_metadata):
        """
        Create an object based on a metadata 
        dictionary from tmy2 or tmy3 data readers.
        """
        return cls(**tmy_metadata)
    
    
    def get_solarposition(self, times, **kwargs):
        return solarposition.get_solarposition(times, latitude=self.latitude,
                                               longitude=self.longitude,
                                               **kwargs)
                                               