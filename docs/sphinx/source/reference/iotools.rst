
.. py:module:: pvlib.iotools

.. currentmodule:: pvlib

.. _iotools:

IO Tools
========

Functions for retrieving, reading, and writing data from a variety
of sources and file formats relevant to solar energy modeling.

.. contents:: Currently supported APIs
    :depth: 3
    :local:

Public datasets
---------------

PVGIS
^^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_pvgis_tmy
   iotools.read_pvgis_tmy
   iotools.get_pvgis_hourly
   iotools.read_pvgis_hourly
   iotools.get_pvgis_horizon


CAMS
^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_cams
   iotools.read_cams
   iotools.parse_cams


NSRDB
^^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_nsrdb_psm4_aggregated
   iotools.get_nsrdb_psm4_tmy
   iotools.get_nsrdb_psm4_conus
   iotools.get_nsrdb_psm4_full_disc
   iotools.read_nsrdb_psm4
   iotools.parse_nsrdb_psm4
   iotools.get_psm3
   iotools.read_psm3
   iotools.parse_psm3


BSRN
^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_bsrn
   iotools.read_bsrn
   iotools.parse_bsrn


SOLRAD
^^^^^^

.. autosummary::
   :toctree: generated/

   iotools.read_solrad
   iotools.get_solrad


SURFRAD
^^^^^^^

.. autosummary::
   :toctree: generated/

   iotools.read_surfrad


MIDC
^^^^

.. autosummary::
   :toctree: generated/

   iotools.read_midc
   iotools.read_midc_raw_data_from_nrel


SRML
^^^^

.. autosummary::
   :toctree: generated/

   iotools.read_srml
   iotools.get_srml


ACIS
^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_acis_prism
   iotools.get_acis_nrcc
   iotools.get_acis_mpe
   iotools.get_acis_station_data
   iotools.get_acis_available_stations


CRN
^^^

.. autosummary::
   :toctree: generated/

   iotools.read_crn



Commercial datasets
-------------------


SolarAnywhere
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_solaranywhere
   iotools.read_solaranywhere


Solcast
^^^^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_solcast_tmy
   iotools.get_solcast_historic
   iotools.get_solcast_forecast
   iotools.get_solcast_live


Solargis
^^^^^^^^

.. autosummary::
   :toctree: generated/

   iotools.get_solargis


Generic data file readers
-------------------------

.. autosummary::
   :toctree: generated/

   iotools.read_tmy2
   iotools.read_tmy3
   iotools.read_epw
   iotools.parse_epw
   iotools.read_panond


A :py:class:`~pvlib.location.Location` object may be created from metadata
in some files.

.. autosummary::
   :toctree: generated/

   location.Location.from_tmy
   location.Location.from_epw
