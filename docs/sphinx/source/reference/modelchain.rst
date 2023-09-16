.. currentmodule:: pvlib

ModelChain
==========

Creating a ModelChain object.

.. autosummary::
   :toctree: generated/

   modelchain.ModelChain
   modelchain.ModelChain.with_pvwatts
   modelchain.ModelChain.with_sapm

.. _modelchain_runmodel:

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

   modelchain.ModelChain.dc_model
   modelchain.ModelChain.ac_model
   modelchain.ModelChain.aoi_model
   modelchain.ModelChain.spectral_model
   modelchain.ModelChain.temperature_model
   modelchain.ModelChain.dc_ohmic_model
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
   modelchain.ModelChain.sandia_inverter
   modelchain.ModelChain.adr_inverter
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
   modelchain.ModelChain.dc_ohmic_model
   modelchain.ModelChain.no_dc_ohmic_loss
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

   modelchain.get_orientation
