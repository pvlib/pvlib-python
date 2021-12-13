.. currentmodule:: pvlib

Solar Position
==============

Functions and methods for calculating solar position.

The :py:meth:`location.Location.get_solarposition` method and the
:py:func:`solarposition.get_solarposition` function with default
parameters are fast and accurate. We recommend using these functions
unless you know that you need a different function.

.. autosummary::
   :toctree: generated/

   location.Location.get_solarposition
   solarposition.get_solarposition
   solarposition.spa_python
   solarposition.ephemeris
   solarposition.pyephem
   solarposition.spa_c


Additional functions for quantities closely related to solar position.

.. autosummary::
   :toctree: generated/

   solarposition.calc_time
   solarposition.pyephem_earthsun_distance
   solarposition.nrel_earthsun_distance
   spa.calculate_deltat


Functions for calculating sunrise, sunset and transit times.

.. autosummary::
   :toctree: generated/

   location.Location.get_sun_rise_set_transit
   solarposition.sun_rise_set_transit_ephem
   solarposition.sun_rise_set_transit_spa
   solarposition.sun_rise_set_transit_geometric


The spa module contains the implementation of the built-in NREL SPA
algorithm.

.. autosummary::
   :toctree: generated/

   spa

Correlations and analytical expressions for low precision solar position
calculations.

.. autosummary::
   :toctree: generated/

   solarposition.solar_zenith_analytical
   solarposition.solar_azimuth_analytical
   solarposition.declination_spencer71
   solarposition.declination_cooper69
   solarposition.equation_of_time_spencer71
   solarposition.equation_of_time_pvcdrom
   solarposition.hour_angle
