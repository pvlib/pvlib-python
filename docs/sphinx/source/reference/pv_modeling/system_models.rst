.. currentmodule:: pvlib


PV System Models
----------------

Sandia array performance model (SAPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/

   pvsystem.sapm
   pvsystem.sapm_effective_irradiance
   pvsystem.sapm_spectral_loss
   spectrum.spectral_factor_sapm
   inverter.sandia
   temperature.sapm_cell

PVsyst model
^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/

   temperature.pvsyst_cell
   pvsystem.calcparams_pvsyst
   pvsystem.singlediode
   ivtools.sdm.pvsyst_temperature_coeff
   pvsystem.dc_ohms_from_percent
   pvsystem.dc_ohmic_losses

PVWatts model
^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/

   pvsystem.pvwatts_dc
   inverter.pvwatts
   pvsystem.pvwatts_losses

ADR model
^^^^^^^^^

.. autosummary::
   :toctree: ../generated/

   pvarray.pvefficiency_adr
   pvarray.fit_pvefficiency_adr

PVGIS model
^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/

    pvarray.huld
