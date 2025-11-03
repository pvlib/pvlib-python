
.. py:module:: pvlib.iotools

.. currentmodule:: pvlib

.. _iotools:

IO Tools
========

Functions for retrieving, reading, and writing data from a variety
of sources and file formats relevant to solar energy modeling.  See also
:ref:`weatherdata`.

.. contents:: Currently supported APIs
    :depth: 3
    :local:


Gridded solar resource data
---------------------------

Modeled irradiance and meteorological data, available for large regions of
the world.

Public datasets
^^^^^^^^^^^^^^^

These APIs are free to access, although some require registration.

PVGIS
*****

Multiple gridded irradiance datasets and weather data with global coverage.

.. autosummary::
   :toctree: generated/

   iotools.get_pvgis_tmy
   iotools.read_pvgis_tmy
   iotools.get_pvgis_hourly
   iotools.read_pvgis_hourly
   iotools.get_pvgis_horizon


CAMS
****

Satellite-derived irradiance data for Europe, Africa, and Asia, and
clear-sky irradiance globally.

.. autosummary::
   :toctree: generated/

   iotools.get_cams
   iotools.read_cams
   iotools.parse_cams


NASA POWER
**********

Satellite-derived irradiance and weather data with global coverage.

.. autosummary::
   :toctree: generated/

   iotools.get_nasa_power


NSRDB
*****

Satellite-derived irradiance and weather data for the Americas.

.. autosummary::
   :toctree: generated/

   iotools.get_nsrdb_psm4_aggregated
   iotools.get_nsrdb_psm4_tmy
   iotools.get_nsrdb_psm4_conus
   iotools.get_nsrdb_psm4_full_disc
   iotools.read_nsrdb_psm4


Commercial datasets
^^^^^^^^^^^^^^^^^^^

Accessing these APIs typically requires payment.
Datasets provide near-global coverage.

Meteonorm
*********

.. autosummary::
   :toctree: generated/

   iotools.get_meteonorm_forecast_basic
   iotools.get_meteonorm_forecast_precision
   iotools.get_meteonorm_observation_training
   iotools.get_meteonorm_observation_realtime
   iotools.get_meteonorm_tmy


SolarAnywhere
*************

.. autosummary::
   :toctree: generated/

   iotools.get_solaranywhere
   iotools.read_solaranywhere


Solcast
*******

.. autosummary::
   :toctree: generated/

   iotools.get_solcast_tmy
   iotools.get_solcast_historic
   iotools.get_solcast_forecast
   iotools.get_solcast_live


Solargis
********

.. autosummary::
   :toctree: generated/

   iotools.get_solargis


Ground station solar data
-------------------------

Measurements from various networks of ground stations
focusing on solar irradiance.

BSRN
^^^^

A global network dedicated to high-quality monitoring of solar and
long-wave radiation.

.. autosummary::
   :toctree: generated/

   iotools.get_bsrn
   iotools.read_bsrn
   iotools.parse_bsrn


SOLRAD
^^^^^^

A solar radiation network in the USA, run by NOAA.

.. autosummary::
   :toctree: generated/

   iotools.read_solrad
   iotools.get_solrad


SURFRAD
^^^^^^^

A solar radiation network in the USA, run by NOAA.

.. autosummary::
   :toctree: generated/

   iotools.read_surfrad


MIDC
^^^^

A solar radiation network in the USA, run by NREL.

.. autosummary::
   :toctree: generated/

   iotools.read_midc
   iotools.read_midc_raw_data_from_nrel


SRML
^^^^

A solar radiation network in the northwestern USA, run by
the University of Oregon.

.. autosummary::
   :toctree: generated/

   iotools.read_srml
   iotools.get_srml


Weather data
------------

Meteorological data from a variety of modeled and measured datasets.

ACIS
^^^^

A combination of many meteorological datasets providing temperature,
precipitation, wind speed, and other weather measurements.

.. autosummary::
   :toctree: generated/

   iotools.get_acis_prism
   iotools.get_acis_nrcc
   iotools.get_acis_mpe
   iotools.get_acis_station_data
   iotools.get_acis_available_stations


CRN
^^^

A network of ground stations from NOAA.  Irradiance measurements are of
lower quality.

.. autosummary::
   :toctree: generated/

   iotools.read_crn

ECMWF ERA5
^^^^^^^^^^

A global reanalysis dataset providing weather and solar resource data.

.. autosummary::
   :toctree: generated/

   iotools.get_era5

MERRA-2
^^^^^^^

A global reanalysis dataset providing weather, aerosol, and solar irradiance
data.

.. autosummary::
   :toctree: generated/

   iotools.get_merra2


Generic data file readers
-------------------------

Functions for reading irradiance/weather data files.

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
