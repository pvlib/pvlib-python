.. currentmodule:: pvlib

#############
API reference
#############


Classes
=======

pvlib-python provides a collection of classes
for users that prefer object-oriented programming.
These classes can help users keep track of data in a more organized way,
and can help to simplify the modeling process.
The classes do not add any functionality beyond the procedural code.
Most of the object methods are simple wrappers around the
corresponding procedural code.

.. autosummary::
   :toctree: generated/

   pvlib.location.Location
   pvlib.pvsystem.PVSystem
   pvlib.tracking.SingleAxisTracker
   pvlib.modelchain.ModelChain
   pvlib.pvsystem.LocalizedPVSystem
   pvlib.tracking.LocalizedSingleAxisTracker


Solar Position
==============

Functions and methods for calculating solar position.

.. autosummary::
   :toctree: generated/

   pvlib.location.Location.get_solarposition
   pvlib.solarposition.get_solarposition
   pvlib.solarposition.spa_python
   pvlib.solarposition.ephemeris
   pvlib.solarposition.pyephem
   pvlib.solarposition.spa_c

Additional functions for quantities closely related to solar position.

.. autosummary::
   :toctree: generated/

   pvlib.solarposition.calc_time
   pvlib.solarposition.pyephem_earthsun_distance
   pvlib.solarposition.nrel_earthsun_distance


Clear sky
=========

.. autosummary::
   :toctree: generated/

   pvlib.location.Location.get_clearsky
   pvlib.clearsky.ineichen
   pvlib.clearsky.lookup_linke_turbidity
   pvlib.clearsky.simplfied_solis
   pvlib.clearsky.haurwitz


Airmass and atmospheric models
==============================

.. autosummary::
   :toctree: generated/

   pvlib.location.Location.get_airmass
   pvlib.atmosphere.absoluteairmass
   pvlib.atmosphere.relativeairmass
   pvlib.atmosphere.pres2alt
   pvlib.atmosphere.alt2pres
   pvlib.atmosphere.gueymard94_pw
   pvlib.atmosphere.first_solar_spectral_correction


Irradiance
==========

Functions for irradiance calculations.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.PVSystem.get_irradiance
   pvlib.pvsystem.PVSystem.get_aoi
   pvlib.tracking.SingleAxisTracker.get_irradiance
   pvlib.pvsystem.SingleAxisTracker.get_aoi

Functions for irradiance calculations.

.. autosummary::
   :toctree: generated/

   pvlib.irradiance.extraradiation
   pvlib.irradiance.aoi
   pvlib.irradiance.aoi_projection
   pvlib.irradiance.poa_horizontal_ratio
   pvlib.irradiance.beam_component
   pvlib.irradiance.globalinplane
   pvlib.irradiance.grounddiffuse

Transposition models.

.. autosummary::
   :toctree: generated/

   pvlib.irradiance.total_irrad
   pvlib.irradiance.isotropic
   pvlib.irradiance.perez
   pvlib.irradiance.haydavies
   pvlib.irradiance.klucher
   pvlib.irradiance.reindl
   pvlib.irradiance.king

DNI estimation models.

.. autosummary::
   :toctree: generated/

   pvlib.irradiance.disc
   pvlib.irradiance.dirint
   pvlib.irradiance.erbs
   pvlib.irradiance.liujordan


PV Modeling
===========

Classes.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.PVSystem
   pvlib.pvsystem.LocalizedPVSystem

Angle of incidence modifiers.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.physicaliam
   pvlib.pvsystem.ashraeiam
   pvlib.pvsystem.sapm_aoi_loss

Functions relevant for the single diode model.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.singlediode
   pvlib.pvsystem.calcparams_desoto
   pvlib.pvsystem.v_from_i
   pvlib.pvsystem.i_from_v

Functions relevant for the SAPM model.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.sapm
   pvlib.pvsystem.sapm_effective_irradiance
   pvlib.pvsystem.sapm_celltemp
   pvlib.pvsystem.sapm_spectral_loss
   pvlib.pvsystem.sapm_aoi_loss
   pvlib.pvsystem.snlinverter

PVWatts model.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.pvwatts_dc
   pvlib.pvsystem.pvwatts_ac
   pvlib.pvsystem.pvwatts_losses


Other.

.. autosummary::
   :toctree: generated/

   pvlib.pvsystem.retrieve_sam
   pvlib.pvsystem.systemdef
   pvlib.pvsystem.scale_voltage_current_power


Tracking
========

.. autosummary::
   :toctree: generated/

   pvlib.tracking.SingleAxisTracker
   pvlib.tracking


TMY
===

.. autosummary::
   :toctree: generated/

   pvlib.location.Location.from_tmy
   pvlib.tmy.readtmy2
   pvlib.tmy.readtmy3


Forecasting
===========

.. autosummary::
   :toctree: generated/

   pvlib.forecast


ModelChain
==========

.. autosummary::
   :toctree: generated/

   pvlib.modelchain.ModelChain

