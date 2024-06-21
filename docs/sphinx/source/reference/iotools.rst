
.. py:module:: pvlib.iotools

.. currentmodule:: pvlib

.. _iotools:

IO Tools
========

Functions for retrieving, reading, and writing data from a variety
of sources and file formats relevant to solar energy modeling.

.. autosummary::
   :toctree: generated/

   iotools.read_tmy2
   iotools.read_tmy3
   iotools.read_epw
   iotools.parse_epw
   iotools.read_srml
   iotools.get_srml
   iotools.read_surfrad
   iotools.read_midc
   iotools.read_midc_raw_data_from_nrel
   iotools.read_crn
   iotools.read_solrad
   iotools.get_solrad
   iotools.get_psm3
   iotools.read_psm3
   iotools.parse_psm3
   iotools.get_pvgis_tmy
   iotools.read_pvgis_tmy
   iotools.get_pvgis_hourly
   iotools.read_pvgis_hourly
   iotools.get_pvgis_horizon
   iotools.get_bsrn
   iotools.read_bsrn
   iotools.parse_bsrn
   iotools.get_cams
   iotools.read_cams
   iotools.parse_cams
   iotools.get_acis_prism
   iotools.get_acis_nrcc
   iotools.get_acis_mpe
   iotools.get_acis_station_data
   iotools.get_acis_available_stations
   iotools.read_panond
   iotools.get_solaranywhere
   iotools.read_solaranywhere
   iotools.get_solcast_tmy
   iotools.get_solcast_historic
   iotools.get_solcast_forecast
   iotools.get_solcast_live
   iotools.get_solargis


A :py:class:`~pvlib.location.Location` object may be created from metadata
in some files.

.. autosummary::
   :toctree: generated/

   location.Location.from_tmy
   location.Location.from_epw
