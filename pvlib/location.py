"""
This module contains the Location class.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import pytz

class Location(object):
    """
    The Location class.
    """
    
    def __init__(self, latitude, longitude, tz='US/Mountain', altitude=100):
        """
        :param latitude: float. Positive is north of the equator.
                         Use decimal degrees notation.
        :param longitude: float. Positive is east of the prime meridian.
                          Use decimal degrees notation.
        :param tz: string. See 
                   http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
                   for a list of valid time zones.
        :param alitude: float. Altitude from sea level in meters.
        """
        
        pvl_logger.debug('creating Location object')
        
        self.latitude = latitude
        self.longitude = longitude
        
        self.tz = tz
        self.pytz = pytz.timezone(tz)
        
        self.altitude = altitude
        