"""
This module contains the Location class.
"""

# Will Holmgren, University of Arizona, 2014-2016.

import datetime

import pandas as pd
import pytz

from pvlib import solarposition, clearsky, atmosphere, irradiance


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

    tz : str, int, float, or pytz.timezone.
        See
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        for a list of valid time zones.
        pytz.timezone objects will be converted to strings.
        ints and floats must be in hours from UTC.

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
        elif isinstance(tz, (int, float)):
            self.tz = tz
            self.pytz = pytz.FixedOffset(tz*60)
        else:
            raise TypeError('Invalid tz specification')

        self.altitude = altitude

        self.name = name

        # needed for tying together Location and PVSystem in LocalizedPVSystem
        # if LocalizedPVSystem signature is reversed
        # super(Location, self).__init__(**kwargs)

    def __repr__(self):
        attrs = ['name', 'latitude', 'longitude', 'altitude', 'tz']
        return ('Location: \n  ' + '\n  '.join(
            (attr + ': ' + str(getattr(self, attr)) for attr in attrs)))

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
        tmy2 = tmy_metadata.get('City', False)

        latitude = tmy_metadata['latitude']
        longitude = tmy_metadata['longitude']

        if tmy2:
            name = tmy_metadata['City']
        else:
            name = tmy_metadata['Name']

        tz = tmy_metadata['TZ']
        altitude = tmy_metadata['altitude']

        new_object = cls(latitude, longitude, tz=tz, altitude=altitude,
                         name=name, **kwargs)

        # not sure if this should be assigned regardless of input.
        if tmy_data is not None:
            new_object.tmy_data = tmy_data

        return new_object

    def get_solarposition(self, times, pressure=None, temperature=12,
                          **kwargs):
        """
        Uses the :py:func:`solarposition.get_solarposition` function
        to calculate the solar zenith, azimuth, etc. at this location.

        Parameters
        ----------
        times : DatetimeIndex
        pressure : None, float, or array-like
            If None, pressure will be calculated using
            :py:func:`atmosphere.alt2pres` and ``self.altitude``.
        temperature : None, float, or array-like

        kwargs passed to :py:func:`solarposition.get_solarposition`

        Returns
        -------
        solar_position : DataFrame
            Columns depend on the ``method`` kwarg, but always include
            ``zenith`` and ``azimuth``.
        """
        if pressure is None:
            pressure = atmosphere.alt2pres(self.altitude)

        return solarposition.get_solarposition(times, latitude=self.latitude,
                                               longitude=self.longitude,
                                               altitude=self.altitude,
                                               pressure=pressure,
                                               temperature=temperature,
                                               **kwargs)

    def get_clearsky(self, times, model='ineichen', solar_position=None,
                     dni_extra=None, **kwargs):
        """
        Calculate the clear sky estimates of GHI, DNI, and/or DHI
        at this location.

        Parameters
        ----------
        times: DatetimeIndex
        model: str
            The clear sky model to use. Must be one of
            'ineichen', 'haurwitz', 'simplified_solis'.
        solar_position : None or DataFrame
            DataFrame with with columns 'apparent_zenith', 'zenith',
            'apparent_elevation'.
        dni_extra: None or numeric
            If None, will be calculated from times.

        kwargs passed to the relevant functions. Climatological values
        are assumed in many cases. See source code for details!

        Returns
        -------
        clearsky : DataFrame
            Column names are: ``ghi, dni, dhi``.
        """
        if dni_extra is None:
            dni_extra = irradiance.extraradiation(times.dayofyear)

        try:
            pressure = kwargs.pop('pressure')
        except KeyError:
            pressure = atmosphere.alt2pres(self.altitude)

        if solar_position is None:
            solar_position = self.get_solarposition(times, pressure=pressure,
                                                    **kwargs)

        apparent_zenith = solar_position['apparent_zenith']
        apparent_elevation = solar_position['apparent_elevation']

        if model == 'ineichen':
            try:
                linke_turbidity = kwargs.pop('linke_turbidity')
            except KeyError:
                interp_turbidity = kwargs.pop('interp_turbidity', True)
                linke_turbidity = clearsky.lookup_linke_turbidity(
                    times, self.latitude, self.longitude,
                    interp_turbidity=interp_turbidity)

            try:
                airmass_absolute = kwargs.pop('airmass_absolute')
            except KeyError:
                airmass_absolute = self.get_airmass(
                    times, solar_position=solar_position)['airmass_absolute']

            cs = clearsky.ineichen(apparent_zenith, airmass_absolute,
                                   linke_turbidity, altitude=self.altitude,
                                   dni_extra=dni_extra)
        elif model == 'haurwitz':
            cs = clearsky.haurwitz(apparent_zenith)
        elif model == 'simplified_solis':
            cs = clearsky.simplified_solis(
                apparent_elevation, pressure=pressure, dni_extra=dni_extra,
                **kwargs)
        else:
            raise ValueError(('{} is not a valid clear sky model. Must be ' +
                              'one of ineichen, simplified_solis, haurwitz')
                             .format(model))

        return cs

    def get_airmass(self, times=None, solar_position=None,
                    model='kastenyoung1989'):
        """
        Calculate the relative and absolute airmass.

        Automatically chooses zenith or apparant zenith
        depending on the selected model.

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

        if model in atmosphere.APPARENT_ZENITH_MODELS:
            zenith = solar_position['apparent_zenith']
        elif model in atmosphere.TRUE_ZENITH_MODELS:
            zenith = solar_position['zenith']
        else:
            raise ValueError('{} is not a valid airmass model'.format(model))

        airmass_relative = atmosphere.relativeairmass(zenith, model)

        pressure = atmosphere.alt2pres(self.altitude)
        airmass_absolute = atmosphere.absoluteairmass(airmass_relative,
                                                      pressure)

        airmass = pd.DataFrame()
        airmass['airmass_relative'] = airmass_relative
        airmass['airmass_absolute'] = airmass_absolute

        return airmass
