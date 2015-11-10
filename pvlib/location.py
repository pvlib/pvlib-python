"""
This module contains the Location class.
"""

# Will Holmgren, University of Arizona, 2014.

import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pytz

from pvlib import solarposition
from pvlib import clearsky


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
    def from_tmy(cls, tmy_metadata, tmy_data=None, **kwargs):
        """
        Create an object based on a metadata 
        dictionary from tmy2 or tmy3 data readers.
        
        Parameters
        ----------
        tmy_metadata : dict
            Returned from tmy.readtmy2 or tmy.readtmy3
        tmy_data : None or DataFrame
            Optionally attach the TMY data to this object.
        
        Returns
        -------
        Location object (or the child class of Location that you
        called this method from).
        """
        # not complete, but hopefully you get the idea.
        # might need code to handle the difference between tmy2 and tmy3
        
        # determine if we're dealing with TMY2 or TMY3 data
        tmy2 = tmy_metadata.get('StationName', False)
        
        latitude = tmy_metadata['latitude']
        longitude = tmy_metadata['longitude']
        
        if tmy2:
            altitude = tmy_metadata['SiteElevation']
            name = tmy_metadata['StationName']
            tz = tmy_metadata['SiteTimeZone']
        else:
            altitude = tmy_metadata['alititude']
            name = tmy_metadata['Name']
            tz = tmy_metadata['TZ']
        
        new_object = cls(latitude, longitude, tz, altitude, name, **kwargs)
        
        # not sure if this should be assigned regardless of input.
        if tmy_data is not None:
            new_object.tmy_data = tmy_data
        
        return new_object


    def get_solarposition(self, times, **kwargs):
        """
        Uses the :func:`solarposition.get_solarposition` function
        to calculate the solar zenith, azimuth, etc. at this location.
        
        Parameters
        ----------
        times : DatetimeIndex
        
        kwargs passed to :func:`solarposition.get_solarposition`
        
        Returns
        -------
        solarposition : DataFrame
            Columns depend on the ``method`` kwarg, but always include
            ``zenith`` and ``azimuth``. 
        """
        return solarposition.get_solarposition(times, latitude=self.latitude,
                                               longitude=self.longitude,
                                               **kwargs)


    def get_clearsky(self, times, **kwargs):
        """
        Uses the :func:`clearsky.ineichen` function to calculate
        the clear sky estimates of GHI, DNI, and DHI at this location.
        
        Parameters
        ----------
        times : DatetimeIndex
        
        kwargs passed to :func:`clearsky.ineichen`
        
        Returns
        -------
        clearsky : DataFrame
            Column names are: ``ghi, dni, dhi``.
        """
        return clearsky.ineichen(times, latitude=self.latitude,
                                 longitude=self.longitude,
                                 **kwargs)

                                      