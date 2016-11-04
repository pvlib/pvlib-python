.. currentmodule:: pvlib

#############
API reference
#############


Classes
=======

pvlib-python provides a collection of classes for users that prefer
object-oriented programming. These classes can help users keep track of
data in a more organized way, and can help to simplify the modeling
process. The classes do not add any functionality beyond the procedural
code. Most of the object methods are simple wrappers around the
corresponding procedural code.

.. autosummary::
   :toctree: generated/

   location.Location
   pvsystem.PVSystem
   tracking.SingleAxisTracker
   modelchain.ModelChain
   pvsystem.LocalizedPVSystem
   tracking.LocalizedSingleAxisTracker


Solar Position
==============

Functions and methods for calculating solar position.

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

The spa module contains the implementation of the built-in NREL SPA
algorithm.

.. autosummary::
   :toctree: generated/

   spa


Clear sky
=========

.. autosummary::
   :toctree: generated/

   location.Location.get_clearsky
   clearsky.ineichen
   clearsky.lookup_linke_turbidity
   clearsky.simplified_solis
   clearsky.haurwitz


Airmass and atmospheric models
==============================

.. autosummary::
   :toctree: generated/

   location.Location.get_airmass
   atmosphere.absoluteairmass
   atmosphere.relativeairmass
   atmosphere.pres2alt
   atmosphere.alt2pres
   atmosphere.gueymard94_pw
   atmosphere.first_solar_spectral_correction


Irradiance
==========

Methods for irradiance calculations
-----------------------------------

.. autosummary::
   :toctree: generated/

   pvsystem.PVSystem.get_irradiance
   pvsystem.PVSystem.get_aoi
   tracking.SingleAxisTracker.get_irradiance

Decomposing and combining irradiance
------------------------------------

.. autosummary::
   :toctree: generated/

   irradiance.extraradiation
   irradiance.aoi
   irradiance.aoi_projection
   irradiance.poa_horizontal_ratio
   irradiance.beam_component
   irradiance.globalinplane
   irradiance.grounddiffuse

Transposition models
--------------------

.. autosummary::
   :toctree: generated/

   irradiance.total_irrad
   irradiance.isotropic
   irradiance.perez
   irradiance.haydavies
   irradiance.klucher
   irradiance.reindl
   irradiance.king

DNI estimation models
---------------------

.. autosummary::
   :toctree: generated/

   irradiance.disc
   irradiance.dirint
   irradiance.erbs
   irradiance.liujordan


PV Modeling
===========

Classes
-------

The :py:class:`~pvsystem.PVSystem` class provides many methods that
wrap the functions listed below. See its documentation for details.

.. autosummary::
   :toctree: generated/

   pvsystem.PVSystem
   pvsystem.LocalizedPVSystem

AOI modifiers
-------------

.. autosummary::
   :toctree: generated/

   pvsystem.physicaliam
   pvsystem.ashraeiam
   pvsystem.sapm_aoi_loss

Single diode model
------------------

Functions relevant for the single diode model.

.. autosummary::
   :toctree: generated/

   pvsystem.singlediode
   pvsystem.calcparams_desoto
   pvsystem.v_from_i
   pvsystem.i_from_v

SAPM model
----------

Functions relevant for the SAPM model.

.. autosummary::
   :toctree: generated/

   pvsystem.sapm
   pvsystem.sapm_effective_irradiance
   pvsystem.sapm_celltemp
   pvsystem.sapm_spectral_loss
   pvsystem.sapm_aoi_loss
   pvsystem.snlinverter

PVWatts model
-------------

.. autosummary::
   :toctree: generated/

   pvsystem.pvwatts_dc
   pvsystem.pvwatts_ac
   pvsystem.pvwatts_losses


Other
-----

.. autosummary::
   :toctree: generated/

   pvsystem.retrieve_sam
   pvsystem.systemdef
   pvsystem.scale_voltage_current_power


Tracking
========

SingleAxisTracker
-----------------

The :py:class:`~tracking.SingleAxisTracker` inherits from
:py:class:`~pvsystem.PVSystem`.

.. autosummary::
   :toctree: generated/

   tracking.SingleAxisTracker
   tracking.SingleAxisTracker.singleaxis
   tracking.SingleAxisTracker.get_irradiance
   tracking.SingleAxisTracker.localize
   tracking.LocalizedSingleAxisTracker

Functions
---------

.. autosummary::
   :toctree: generated/

   tracking.singleaxis


TMY
===

Methods and functions for reading data from TMY files.

.. autosummary::
   :toctree: generated/

   location.Location.from_tmy
   tmy.readtmy2
   tmy.readtmy3


Forecasting
===========

Forecast models
---------------

.. autosummary::
   :toctree: generated/

   forecast.GFS
   forecast.NAM
   forecast.RAP
   forecast.HRRR
   forecast.HRRR_ESRL
   forecast.NDFD

Getting data
------------

.. autosummary::
   :toctree: generated/

   forecast.ForecastModel.get_data
   forecast.ForecastModel.get_processed_data

Processing data
---------------

.. autosummary::
   :toctree: generated/

   forecast.ForecastModel.process_data
   forecast.ForecastModel.rename
   forecast.ForecastModel.cloud_cover_to_ghi_linear
   forecast.ForecastModel.cloud_cover_to_irradiance_clearsky_scaling
   forecast.ForecastModel.cloud_cover_to_transmittance_linear
   forecast.ForecastModel.cloud_cover_to_irradiance_liujordan
   forecast.ForecastModel.cloud_cover_to_irradiance
   forecast.ForecastModel.kelvin_to_celsius
   forecast.ForecastModel.isobaric_to_ambient_temperature
   forecast.ForecastModel.uv_to_speed
   forecast.ForecastModel.gust_to_speed

IO support
----------

These are public for now, but use at your own risk.

.. autosummary::
   :toctree: generated/

   forecast.ForecastModel.set_dataset
   forecast.ForecastModel.set_query_latlon
   forecast.ForecastModel.set_location
   forecast.ForecastModel.set_time


ModelChain
==========

Creating a ModelChain object.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain

Running
-------

Running a ModelChain.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.run_model
   modelchain.ModelChain.complete_irradiance
   modelchain.ModelChain.prepare_inputs

Attributes
----------

Simple ModelChain attributes:

``system, location, clearsky_model, transposition_model,
solar_position_method, airmass_model``

Properties
----------

ModelChain properties that are aliases for your specific modeling functions.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.orientation_strategy
   modelchain.ModelChain.dc_model
   modelchain.ModelChain.ac_model
   modelchain.ModelChain.aoi_model
   modelchain.ModelChain.spectral_model
   modelchain.ModelChain.temp_model
   modelchain.ModelChain.losses_model
   modelchain.ModelChain.effective_irradiance_model

Model definitions
-----------------

ModelChain model definitions.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.sapm
   modelchain.ModelChain.singlediode
   modelchain.ModelChain.pvwatts_dc
   modelchain.ModelChain.snlinverter
   modelchain.ModelChain.adrinverter
   modelchain.ModelChain.pvwatts_inverter
   modelchain.ModelChain.ashrae_aoi_loss
   modelchain.ModelChain.physical_aoi_loss
   modelchain.ModelChain.sapm_aoi_loss
   modelchain.ModelChain.no_aoi_loss
   modelchain.ModelChain.first_solar_spectral_loss
   modelchain.ModelChain.sapm_spectral_loss
   modelchain.ModelChain.no_spectral_loss
   modelchain.ModelChain.sapm_temp
   modelchain.ModelChain.pvwatts_losses
   modelchain.ModelChain.no_extra_losses

Inference methods
-----------------

Methods that automatically determine which models should be used based
on the information in the associated :py:class:`~pvsystem.PVSystem` object.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.infer_dc_model
   modelchain.ModelChain.infer_ac_model
   modelchain.ModelChain.infer_aoi_model
   modelchain.ModelChain.infer_spectral_model
   modelchain.ModelChain.infer_temp_model
   modelchain.ModelChain.infer_losses_model

Functions
---------

Functions for power modeling.

.. autosummary::
   :toctree: generated/

   modelchain.basic_chain
   modelchain.get_orientation
