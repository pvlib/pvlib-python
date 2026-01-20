.. currentmodule:: pvlib

Location
========

Methods for information about locations.

.. autosummary::
   :toctree: generated/

   location.lookup_altitude

Classes
-------
.. autosummary::
   :toctree: generated/
   
   location.Location

A :py:class:`~pvlib.location.Location` object may be created from the
metadata returned by these file types:

.. autosummary::
   :toctree: generated/

   location.Location.from_tmy
   location.Location.from_epw

Methods for calculating time series of certain variables for a given
location are available through this class.

