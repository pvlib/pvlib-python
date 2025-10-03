.. currentmodule:: pvlib

Location
========

Methods for information about locations.

.. autosummary::
   :toctree: generated/

   location.lookup_altitude

A :py:class:`~pvlib.location.Location` object may be created from the
metadata returned by some file types.

.. autosummary::
   :toctree: generated/

   location.Location.from_tmy
   location.Location.from_epw

Methods for calculating time series of certain variables for a given
location.

.. autosummary::
   :toctree: generated/

   location.Location.get_airmass
   location.Location.get_solarposition
   location.Location.get_sun_rise_set_transit
   location.Location.get_clearsky
