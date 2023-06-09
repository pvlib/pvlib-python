.. currentmodule:: pvlib

Forecasting
===========

.. warning::

    All functionality in the ``pvlib.forecast`` module is deprecated as of
    pvlib v0.9.1. For details, see :ref:`forecasts`.


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
