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
   pvsystem.Array
   tracking.SingleAxisTracker
   modelchain.ModelChain
   modelchain.ModelChainResult

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
   solarposition.sun_rise_set_transit_geometric


Clear sky
=========

.. autosummary::
   :toctree: generated/

   location.Location.get_clearsky
   clearsky.ineichen
   clearsky.lookup_linke_turbidity
   clearsky.simplified_solis
   clearsky.haurwitz
   clearsky.detect_clearsky
   clearsky.bird


Airmass and atmospheric models
==============================

.. autosummary::
   :toctree: generated/

   location.Location.get_airmass
   atmosphere.get_absolute_airmass
   atmosphere.get_relative_airmass
   atmosphere.pres2alt
   atmosphere.alt2pres
   atmosphere.gueymard94_pw
   atmosphere.first_solar_spectral_correction
   atmosphere.bird_hulstrom80_aod_bb
   atmosphere.kasten96_lt
   atmosphere.angstrom_aod_at_lambda
   atmosphere.angstrom_alpha


Irradiance
==========

Methods for irradiance calculations
-----------------------------------

.. autosummary::
   :toctree: generated/

   pvsystem.PVSystem.get_irradiance
   pvsystem.PVSystem.get_aoi
   pvsystem.PVSystem.get_iam
   tracking.SingleAxisTracker.get_irradiance

Decomposing and combining irradiance
------------------------------------

.. autosummary::
   :toctree: generated/

   irradiance.get_extra_radiation
   irradiance.aoi
   irradiance.aoi_projection
   irradiance.poa_horizontal_ratio
   irradiance.beam_component
   irradiance.poa_components
   irradiance.get_ground_diffuse
   irradiance.dni

Transposition models
--------------------

.. autosummary::
   :toctree: generated/

   irradiance.get_total_irradiance
   irradiance.get_sky_diffuse
   irradiance.isotropic
   irradiance.perez
   irradiance.haydavies
   irradiance.klucher
   irradiance.reindl
   irradiance.king

.. _dniestmodels:

DNI estimation models
---------------------

.. autosummary::
   :toctree: generated/

   irradiance.disc
   irradiance.dirint
   irradiance.dirindex
   irradiance.erbs
   irradiance.campbell_norman
   irradiance.gti_dirint

Clearness index models
----------------------

.. autosummary::
   :toctree: generated/

   irradiance.clearness_index
   irradiance.clearness_index_zenith_independent
   irradiance.clearsky_index


PV Modeling
===========

Classes
-------

The :py:class:`~pvsystem.PVSystem` class provides many methods that
wrap the functions listed below. See its documentation for details.

.. autosummary::
   :toctree: generated/

   pvsystem.PVSystem

Incident angle modifiers
------------------------

.. autosummary::
   :toctree: generated/

   iam.physical
   iam.ashrae
   iam.martin_ruiz
   iam.martin_ruiz_diffuse
   iam.sapm
   iam.interp
   iam.marion_diffuse
   iam.marion_integrate

PV temperature models
---------------------

.. autosummary::
   :toctree: generated/

   temperature.sapm_cell
   temperature.sapm_module
   temperature.sapm_cell_from_module
   temperature.pvsyst_cell
   temperature.faiman
   temperature.fuentes
   temperature.ross
   pvsystem.PVSystem.sapm_celltemp
   pvsystem.PVSystem.pvsyst_celltemp
   pvsystem.PVSystem.faiman_celltemp

Temperature Model Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: pvlib.temperature
.. autodata:: TEMPERATURE_MODEL_PARAMETERS
   :annotation:

.. currentmodule:: pvlib

Single diode models
-------------------

Functions relevant for single diode models.

.. autosummary::
   :toctree: generated/

   pvsystem.calcparams_cec
   pvsystem.calcparams_desoto
   pvsystem.calcparams_pvsyst
   pvsystem.i_from_v
   pvsystem.singlediode
   pvsystem.v_from_i
   pvsystem.max_power_point

Low-level functions for solving the single diode equation.

.. autosummary::
   :toctree: generated/

   singlediode.estimate_voc
   singlediode.bishop88
   singlediode.bishop88_i_from_v
   singlediode.bishop88_v_from_i
   singlediode.bishop88_mpp

Functions for fitting diode models

.. autosummary::
   :toctree: generated/

    ivtools.sde.fit_sandia_simple
    ivtools.sdm.fit_cec_sam
    ivtools.sdm.fit_desoto

Inverter models (DC to AC conversion)
-------------------------------------

.. autosummary::
   :toctree: generated/

   inverter.sandia
   inverter.sandia_multi
   inverter.adr
   inverter.pvwatts
   inverter.pvwatts_multi

Functions for fitting inverter models

.. autosummary::
   :toctree: generated/

   inverter.fit_sandia


PV System Models
----------------

Sandia array performance model (SAPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   pvsystem.sapm
   pvsystem.sapm_effective_irradiance
   pvsystem.sapm_spectral_loss
   inverter.sandia
   temperature.sapm_cell

Pvsyst model
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   temperature.pvsyst_cell
   pvsystem.calcparams_pvsyst
   pvsystem.singlediode

PVWatts model
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   pvsystem.pvwatts_dc
   inverter.pvwatts
   pvsystem.pvwatts_losses

Estimating PV model parameters
------------------------------

Functions for fitting single diode models

.. autosummary::
   :toctree: generated/

    ivtools.sdm.fit_cec_sam
    ivtools.sdm.fit_desoto
    ivtools.sdm.fit_pvsyst_sandia
    ivtools.sdm.fit_desoto_sandia

Functions for fitting the single diode equation

.. autosummary::
   :toctree: generated/

    ivtools.sde.fit_sandia_simple

Utilities for working with IV curve data

.. autosummary::
   :toctree: generated/

    ivtools.utils.rectify_iv_curve

Other
-----

.. autosummary::
   :toctree: generated/

   pvsystem.retrieve_sam
   pvsystem.scale_voltage_current_power


Effects on PV System Output
===========================

Loss models
-----------

.. autosummary::
   :toctree: generated/

   pvsystem.combine_loss_factors

Snow
----

.. autosummary::
   :toctree: generated/

   snow.coverage_nrel
   snow.fully_covered_nrel
   snow.dc_loss_nrel

Soiling
-------

.. autosummary::
   :toctree: generated/

   soiling.hsu
   soiling.kimber

Shading
-------

.. autosummary::
   :toctree: generated/

   shading.masking_angle
   shading.masking_angle_passias
   shading.sky_diffuse_passias

Spectrum
--------

.. autosummary::
   :toctree: generated/

   spectrum.spectrl2

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

Functions
---------

.. autosummary::
   :toctree: generated/

   tracking.singleaxis
   tracking.calc_axis_tilt
   tracking.calc_cross_axis_tilt


.. _iotools:

IO Tools
========

Functions for reading and writing data from a variety of file formats
relevant to solar energy modeling.

.. autosummary::
   :toctree: generated/

   iotools.read_tmy2
   iotools.read_tmy3
   iotools.read_epw
   iotools.parse_epw
   iotools.read_srml
   iotools.read_srml_month_from_solardat
   iotools.read_surfrad
   iotools.read_midc
   iotools.read_midc_raw_data_from_nrel
   iotools.read_ecmwf_macc
   iotools.get_ecmwf_macc
   iotools.read_crn
   iotools.read_solrad
   iotools.get_psm3
   iotools.read_psm3
   iotools.parse_psm3
   iotools.get_pvgis_tmy
   iotools.read_pvgis_tmy

A :py:class:`~pvlib.location.Location` object may be created from metadata
in some files.

.. autosummary::
   :toctree: generated/

   location.Location.from_tmy
   location.Location.from_epw


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
   forecast.ForecastModel.cloud_cover_to_irradiance_campbell_norman
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
   modelchain.ModelChain.with_pvwatts
   modelchain.ModelChain.with_sapm

Running
-------

A ModelChain can be run from a number of starting points, depending on the
input data available.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.run_model
   modelchain.ModelChain.run_model_from_poa
   modelchain.ModelChain.run_model_from_effective_irradiance

Functions to assist with setting up ModelChains to run

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.complete_irradiance
   modelchain.ModelChain.prepare_inputs
   modelchain.ModelChain.prepare_inputs_from_poa

Results
-------

Output from the running the ModelChain is stored in the
:py:attr:`modelchain.ModelChain.results` attribute. For more
information see :py:class:`modelchain.ModelChainResult`.

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
   modelchain.ModelChain.temperature_model
   modelchain.ModelChain.losses_model
   modelchain.ModelChain.effective_irradiance_model

Model definitions
-----------------

ModelChain model definitions.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain.sapm
   modelchain.ModelChain.cec
   modelchain.ModelChain.desoto
   modelchain.ModelChain.pvsyst
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
   modelchain.ModelChain.pvsyst_temp
   modelchain.ModelChain.faiman_temp
   modelchain.ModelChain.fuentes_temp
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
   modelchain.ModelChain.infer_temperature_model
   modelchain.ModelChain.infer_losses_model

Functions
---------

Functions for power modeling.

.. autosummary::
   :toctree: generated/

   modelchain.basic_chain
   modelchain.get_orientation


Bifacial
========

Methods for calculating back surface irradiance

.. autosummary::
   :toctree: generated/

   bifacial.pvfactors_timeseries


Scaling
=======

Methods for manipulating irradiance for temporal or spatial considerations

.. autosummary::
   :toctree: generated/

   scaling.wvm
