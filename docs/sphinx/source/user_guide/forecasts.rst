.. _forecasts:

***********
Forecasting
***********

pvlib python provides a set of functions and classes that make it easy
to obtain weather forecast data and convert that data into a PV power
forecast. Users can retrieve standardized weather forecast data relevant
to PV power modeling from NOAA/NCEP/NWS models including the GFS, NAM,
RAP, HRRR, and the NDFD. A PV power forecast can then be obtained using
the weather data as inputs to the comprehensive modeling capabilities of
pvlib python. Standardized, open source, reference implementations of
forecast methods using publicly available data may help advance the
state-of-the-art of solar power forecasting.

pvlib python uses Unidata's `Siphon
<http://siphon.readthedocs.org/en/latest/>`_ library to simplify access
to real-time forecast data hosted on the Unidata `THREDDS catalog
<http://thredds.ucar.edu/thredds/catalog.html>`_. Siphon is great for
programatic access of THREDDS data, but we also recommend using tools
such as `Panoply <http://www.giss.nasa.gov/tools/panoply/>`_
to easily browse the catalog and become more familiar with its contents.

We do not know of a similarly easy way to access archives of forecast data.

This document demonstrates how to use pvlib python to create a PV power
forecast using these tools. The `forecast
<http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/
master/docs/tutorials/forecast.ipynb>`_ and `forecast_to_power
<http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/
master/docs/tutorials/forecast_to_power.ipynb>`_ Jupyter notebooks
provide additional example code.

.. warning::

    The forecast module algorithms and features are highly experimental.
    The API may change, the functionality may be consolidated into an io
    module, or the module may be separated into its own package.

.. note::

    This documentation is difficult to reliably build on readthedocs.
    If you do not see images, try building the documentation on your
    own machine or see the notebooks linked to above.


Accessing Forecast Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Siphon library provides access to, among others, forecasts from the
Global Forecast System (GFS), North American Model (NAM), High
Resolution Rapid Refresh (HRRR), Rapid Refresh (RAP), and National
Digital Forecast Database (NDFD) on a Unidata THREDDS server.
Unfortunately, many of these models use different names to describe the
same quantity (or a very similar one), and not all variables are present
in all models. For example, on the THREDDS server, the GFS has a field
named
``Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average``,
while the NAM has a field named
``Total_cloud_cover_entire_atmosphere_single_layer``, and a
similar field in the HRRR is named
``Total_cloud_cover_entire_atmosphere``.

pvlib python aims to simplify the access of the model fields relevant
for solar power forecasts. Model data accessed with pvlib python is
returned as a pandas DataFrame with consistent column names:
``temp_air, wind_speed, total_clouds, low_clouds, mid_clouds,
high_clouds, dni, dhi, ghi``. To accomplish this, we use an
object-oriented framework in which each weather model is represented by
a class that inherits from a parent
:py:class:`~pvlib.forecast.ForecastModel` class.
The parent :py:class:`~pvlib.forecast.ForecastModel` class contains the
common code for accessing and parsing the data using Siphon, while the
child model-specific classes (:py:class:`~pvlib.forecast.GFS`,
:py:class:`~pvlib.forecast.HRRR`, etc.) contain the code necessary to
map and process that specific model's data to the standardized fields.

The code below demonstrates how simple it is to access and plot forecast
data using pvlib python. First, we set up make the basic imports and
then set the location and time range data.

.. ipython:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime

    # import pvlib forecast models
    from pvlib.forecast import GFS, NAM, NDFD, HRRR, RAP

    # specify location (Tucson, AZ)
    latitude, longitude, tz = 32.2, -110.9, 'US/Arizona'

    # specify time range.
    start = pd.Timestamp(datetime.date.today(), tz=tz)
    end = start + pd.Timedelta(days=7)

    irrad_vars = ['ghi', 'dni', 'dhi']


Next, we instantiate a GFS model object and get the forecast data
from Unidata.

.. ipython:: python

    # GFS model, defaults to 0.5 degree resolution
    # 0.25 deg available
    model = GFS()

    # retrieve data. returns pandas.DataFrame object
    raw_data = model.get_data(latitude, longitude, start, end)

    print(raw_data.head())

It will be useful to process this data before using it with pvlib. For
example, the column names are non-standard, the temperature is in
Kelvin, the wind speed is broken into east/west and north/south
components, and most importantly, most of the irradiance data is
missing. The forecast module provides a number of methods to fix these
problems.

.. ipython:: python

    data = raw_data

    # rename the columns according the key/value pairs in model.variables.
    data = model.rename(data)

    # convert temperature
    data['temp_air'] = model.kelvin_to_celsius(data['temp_air'])

    # convert wind components to wind speed
    data['wind_speed'] = model.uv_to_speed(data)

    # calculate irradiance estimates from cloud cover.
    # uses a cloud_cover to ghi to dni model or a
    # uses a cloud cover to transmittance to irradiance model.
    # this step is discussed in more detail in the next section
    irrad_data = model.cloud_cover_to_irradiance(data['total_clouds'])
    data = data.join(irrad_data, how='outer')

    # keep only the final data
    data = data[model.output_variables]

    print(data.head())

Much better.

The GFS class's
:py:func:`~pvlib.forecast.GFS.process_data` method combines these steps
in a single function. In fact, each forecast model class
implements its own ``process_data`` method since the data from each
weather model is slightly different. The ``process_data`` functions are
designed to be explicit about how the data is being processed, and users
are **strongly** encouraged to read the source code of these methods.

.. ipython:: python

    data = model.process_data(raw_data)

    print(data.head())

Users can easily implement their own ``process_data`` methods on
inherited classes or implement similar stand-alone functions.

The forecast model classes also implement a
:py:func:`~pvlib.forecast.ForecastModel.get_processed_data` method that
combines the :py:func:`~pvlib.forecast.ForecastModel.get_data` and
:py:func:`~pvlib.forecast.ForecastModel.process_data` calls.

.. ipython:: python

    data = model.get_processed_data(latitude, longitude, start, end)

    print(data.head())


Cloud cover and radiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the weather models currently accessible by pvlib include one or
more cloud cover forecasts. For example, below we plot the GFS cloud
cover forecasts.

.. ipython:: python

    # plot cloud cover percentages
    cloud_vars = ['total_clouds', 'low_clouds',
                  'mid_clouds', 'high_clouds']
    data[cloud_vars].plot();
    plt.ylabel('Cloud cover %');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('GFS 0.5 deg forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig gfs_cloud_cover.png width=6in
    plt.legend();
    @suppress
    plt.close();

However, many of forecast models do not include radiation components in
their output fields, or if they do then the radiation fields suffer from
poor solar position or radiative transfer algorithms. It is often more
accurate to create empirically derived radiation forecasts from the
weather models' cloud cover forecasts.

pvlib python provides two basic ways to convert cloud cover forecasts to
irradiance forecasts. One method assumes a linear relationship between
cloud cover and GHI, applies the scaling to a clear sky climatology, and
then uses the DISC model to calculate DNI. The second method assumes a
linear relationship between cloud cover and atmospheric transmittance,
and then uses the Campbell-Norman model to calculate GHI, DNI, and
DHI [Cam98]_. Campbell-Norman is an approximation of Liu-Jordan [Liu60]_.

*Caveat emptor*: these algorithms are not rigorously verified! The
purpose of the forecast module is to provide a few exceedingly simple
options for users to play with before they develop their own models. We
strongly encourage pvlib users first read the source code and second
to implement new cloud cover to irradiance algorithms.

The essential parts of the clear sky scaling algorithm are as follows.
Clear sky scaling of climatological GHI is also used in Larson et. al.
[Lar16]_.

.. code-block:: python

    solpos = location.get_solarposition(cloud_cover.index)
    cs = location.get_clearsky(cloud_cover.index, model='ineichen')
    # offset and cloud cover in decimal units here
    # larson et. al. use offset = 0.35
    ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
    dni = disc(ghi, solpos['zenith'], cloud_cover.index)['dni']
    dhi = ghi - dni * np.cos(np.radians(solpos['zenith']))

The figure below shows the result of the total cloud cover to
irradiance conversion using the clear sky scaling algorithm.

.. ipython:: python

    # plot irradiance data
    data = model.rename(raw_data)
    irrads = model.cloud_cover_to_irradiance(data['total_clouds'], how='clearsky_scaling')
    irrads.plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('GFS 0.5 deg forecast for lat={}, lon={} using "clearsky_scaling"'
              .format(latitude, longitude));
    @savefig gfs_irrad_cs.png width=6in
    plt.legend();
    @suppress
    plt.close();


The essential parts of the Campbell-Norman cloud cover to irradiance algorithm
are as follows.

.. code-block:: python

    # cloud cover in percentage units here
    transmittance = ((100.0 - cloud_cover) / 100.0) * 0.75
    # irrads is a DataFrame containing ghi, dni, dhi
    irrads = campbell_norman(apparent_zenith, transmittance)

The figure below shows the result of the Campbell-Norman total cloud cover to
irradiance conversion.

.. ipython:: python

    # plot irradiance data
    irrads = model.cloud_cover_to_irradiance(data['total_clouds'], how='campbell_norman')
    irrads.plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('GFS 0.5 deg forecast for lat={}, lon={} using "campbell_norman"'
              .format(latitude, longitude));
    @savefig gfs_irrad_lj.png width=6in
    plt.legend();
    @suppress
    plt.close();


Most weather model output has a fairly coarse time resolution, at least
an hour. The irradiance forecasts have the same time resolution as the
weather data. However, it is straightforward to interpolate the cloud
cover forecasts onto a higher resolution time domain, and then
recalculate the irradiance.

.. ipython:: python

    resampled_data = data.resample('5min').interpolate()
    resampled_irrads = model.cloud_cover_to_irradiance(resampled_data['total_clouds'], how='clearsky_scaling')
    resampled_irrads.plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('GFS 0.5 deg forecast for lat={}, lon={} resampled'
              .format(latitude, longitude));
    @savefig gfs_irrad_high_res.png width=6in
    plt.legend();
    @suppress
    plt.close();

Users may then recombine resampled_irrads and resampled_data using
slicing :py:func:`pandas.concat` or :py:meth:`pandas.DataFrame.join`.

We reiterate that the open source code enables users to customize the
model processing to their liking.

.. [Lar16] Larson et. al. "Day-ahead forecasting of solar power output
    from photovoltaic plants in the American Southwest" Renewable
    Energy 91, 11-20 (2016).

.. [Cam98] Campbell, G. S., J. M. Norman (1998) An Introduction to
    Environmental Biophysics. 2nd Ed. New York: Springer.

.. [Liu60] B. Y. Liu and R. C. Jordan, The interrelationship and
    characteristic distribution of direct, diffuse, and total solar
    radiation, *Solar Energy* **4**, 1 (1960).


Weather Models
~~~~~~~~~~~~~~

Next, we provide a brief description of the weather models available to
pvlib users. Note that the figures are generated when this documentation
is compiled so they will vary over time.

GFS
---
The Global Forecast System (GFS) is the US model that provides forecasts
for the entire globe. The GFS is updated every 6 hours. The GFS is run
at two resolutions, 0.25 deg and 0.5 deg, and is available with 3 hour
time resolution. Forecasts from GFS model were shown above. Use the GFS,
among others, if you want forecasts for 1-7 days or if you want forecasts
for anywhere on Earth.


HRRR
----
The High Resolution Rapid Refresh (HRRR) model is perhaps the most
accurate model, however, it is only available for ~15 hours. It is
updated every hour and runs at 3 km resolution. The HRRR excels in
severe weather situations. See the `NOAA ESRL HRRR page
<http://rapidrefresh.noaa.gov/hrrr/>`_ for more information. Use the
HRRR, among others, if you want forecasts for less than 24 hours.
The HRRR model covers the continental United States.

.. ipython:: python

    model = HRRR()
    data = model.get_processed_data(latitude, longitude, start, end)

    data[irrad_vars].plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('HRRR 3 km forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig hrrr_irrad.png width=6in
    plt.legend();
    @suppress
    plt.close();


RAP
---
The Rapid Refresh (RAP) model is the parent model for the HRRR. It is
updated every hour and runs at 40, 20, and 13 km resolutions. Only the
20 and 40 km resolutions are currently available in pvlib. It is also
excels in severe weather situations. See the `NOAA ESRL HRRR page
<http://rapidrefresh.noaa.gov/hrrr/>`_ for more information. Use the
RAP, among others, if you want forecasts for less than 24 hours.
The RAP model covers most of North America.

.. ipython:: python

    model = RAP()
    data = model.get_processed_data(latitude, longitude, start, end)

    data[irrad_vars].plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('RAP 13 km forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig rap_irrad.png width=6in
    plt.legend();
    @suppress
    plt.close();


NAM
---
The North American Mesoscale model covers, not surprisingly, North
America. It is updated every 6 hours. pvlib provides access to 20 km
resolution NAM data with a time horizon of up to 4 days.

.. ipython:: python

    model = NAM()
    data = model.get_processed_data(latitude, longitude, start, end)

    data[irrad_vars].plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('NAM 20 km forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig nam_irrad.png width=6in
    plt.legend();
    @suppress
    plt.close();


NDFD
----
The National Digital Forecast Database is not a model, but rather a
collection of forecasts made by National Weather Service offices
across the country. It is updated every 6 hours.
Use the NDFD, among others, for forecasts at all time horizons.
The NDFD is available for the United States.

.. ipython:: python
   :okexcept:

    model = NDFD()
    data = model.get_processed_data(latitude, longitude, start, end)

    data[irrad_vars].plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('NDFD forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig ndfd_irrad.png width=6in
    plt.legend();
    @suppress
    plt.close();


PV Power Forecast
~~~~~~~~~~~~~~~~~

Finally, we demonstrate the application of the weather forecast data to
a PV power forecast. Please see the remainder of the pvlib documentation
for details.

.. ipython:: python

    from pvlib.pvsystem import PVSystem, retrieve_sam
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    from pvlib.tracking import SingleAxisTracker
    from pvlib.modelchain import ModelChain

    sandia_modules = retrieve_sam('sandiamod')
    cec_inverters = retrieve_sam('cecinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = cec_inverters['SMA_America__SC630CP_US__with_ABB_EcoDry_Ultra_transformer_']
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # model a big tracker for more fun
    system = SingleAxisTracker(module_parameters=module, inverter_parameters=inverter, temperature_model_parameters=temperature_model_parameters, modules_per_string=15, strings_per_inverter=300)

    # fx is a common abbreviation for forecast
    fx_model = GFS()
    fx_data = fx_model.get_processed_data(latitude, longitude, start, end)

    # use a ModelChain object to calculate modeling intermediates
    mc = ModelChain(system, fx_model.location)

    # extract relevant data for model chain
    mc.run_model(fx_data);

Now we plot a couple of modeling intermediates and the forecast power.
Here's the forecast plane of array irradiance...

.. ipython:: python

    mc.results.total_irrad.plot();
    @savefig poa_irrad.png width=6in
    plt.ylabel('Plane of array irradiance ($W/m^2$)');
    plt.legend(loc='best');
    @suppress
    plt.close();

...the cell and module temperature...

.. ipython:: python

    mc.results.cell_temperature.plot();
    @savefig pv_temps.png width=6in
    plt.ylabel('Cell Temperature (C)');
    @suppress
    plt.close();

...and finally AC power...

.. ipython:: python

    mc.results.ac.fillna(0).plot();
    plt.ylim(0, None);
    @savefig ac_power.png width=6in
    plt.ylabel('AC Power (W)');
    @suppress
    plt.close();

