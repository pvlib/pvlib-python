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
from pvlib import atmosphere


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
    
    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.
        
    See also
    --------
    pvsystem.PVSystem
    """
    
    def __init__(self, latitude, longitude, tz='UTC', altitude=0,
                 name=None, **kwargs):
        
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
        
        # needed for tying together Location and PVSystem in LocalizedPVSystem
        # if LocalizedPVSystem signature is reversed
        # super(Location, self).__init__(**kwargs)
        
        
        
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
                                               altitude=self.altitude,
                                               **kwargs)


    def get_clearsky(self, times, model='ineichen', **kwargs):
        """
        Calculate the clear sky estimates of GHI, DNI, and/or DHI
        at this location.
        
        Parameters
        ----------
        times : DatetimeIndex
        
        model : str
            The clear sky model to use.
        
        kwargs passed to the relevant function(s).
        
        Returns
        -------
        clearsky : Series or DataFrame
            Column names are: ``ghi, dni, dhi``.
        """
        
        if model == 'ineichen':
            cs = clearsky.ineichen(times, latitude=self.latitude,
                                   longitude=self.longitude,
                                   altitude=self.altitude,
                                   **kwargs)
        elif model == 'haurwitz':
            solpos = self.get_solarposition(times, **kwargs)
            cs = clearsky.haurwitz(solpos['apparent_zenith'])
        else:
            raise ValueError('%s is not a valid clear sky model', model)

        return cs
    
    
    def get_airmass(self, times=None, solar_position=None,
                    model='kastenyoung1998'):
        """
        Calculate the relative and absolute airmass.
        
        Function will calculate the solar zenith and apparent zenith angles,
        and choose the appropriate one.
        
        Parameters
        ----------
        times : None or DatetimeIndex
            Only used if solar_position is not provided.
        solar_position : None or DataFrame
            DataFrame with with columns 'apparent_zenith', 'zenith'.
        model : str
            Relative airmass model
        
        Returns
        -------
        airmass : DataFrame
            Columns are 'airmass_relative', 'airmass_absolute'
        """

        if solar_position is None:
            solar_position = self.get_solarposition(times)

        apparents = ['simple', 'kasten1966', 'kastenyoung1989',
                     'gueymard1993', 'pickering2002']

        trues = ['youngirvine1967', 'young1994']

        if model in apparents:
            zenith = solar_position['apparent_zenith']
        elif model in trues:
            zenith = solar_position['zenith']
        else
            raise ValueError('invalid model %s', model)

        airmass_relative = atmosphere.relativeairmass(zenith, model)

        pressure = atmosphere.alt2pres(self.altitude)
        airmass_absolute = atmosphere.absoluteairmass(airmass_relative,
                                                      pressure)

        airmass = pd.DataFrame()
        airmass['airmass_relative'] = airmass_relative
        airmass['airmass_absolute'] = airmass_absolute

        return airmass
                                      