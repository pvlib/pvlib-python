"""
This module contains the Location class.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pytz

class Location(object):
    """
    Location objects are convenient containers for latitude, longitude,
    timezone, and altitude data associated with a particular 
    geographic location. You can also assign a name to a location object.
    
    Location objects have two timezone attributes: 
        * ``location.tz`` is a IANA timezone string.
        * ``location.pytz`` is a pytz timezone object.
        
    Location objects support the print method.
    """
    
    def __init__(self, latitude, longitude, tz='US/Mountain', altitude=100,
                 name=None):
        """
        :param latitude: float. Positive is north of the equator.
                         Use decimal degrees notation.
        :param longitude: float. Positive is east of the prime meridian.
                          Use decimal degrees notation.
        :param tz: string or pytz.timezone. See 
                   http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
                   for a list of valid time zones.
                   pytz.timezone objects will be converted to strings.
        :param alitude: float. Altitude from sea level in meters.
        :param name: None or string. 
        """
        
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