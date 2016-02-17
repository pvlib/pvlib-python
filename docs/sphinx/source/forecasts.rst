.. _forecasts:

***********
Forecasting
***********

pvlib-python provides a set of functions and classes that make it easy to obtain
weather forecast data and convert that data into a PV power forecast.
Users can retrieve standardized weather forecast data relevant to PV
power modeling from NOAA/NCEP/NWS models including the GFS, NAM, RAP,
HRRR, and the NDFD. A PV power forecast can then be obtained using the
weather data as inputs to the comprehensive modeling capabilities of
PVLIB-Python. Standardized, open source, reference implementations of
forecast methods using publicly available data may help advance the
state-of-the-art of solar power forecasting.

pvlib-python uses Unidata's
`Siphon <http://siphon.readthedocs.org/en/latest/>`_ library to simplify
access to forecast data hosted on the Unidata
`THREDDS catalog <http://thredds.ucar.edu/thredds/catalog.html>`_.

This document demonstrates how to use pvlib-python to create
a PV power forecast using these tools.


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
``Total\_cloud\_cover\_entire\_atmosphere\_Mixed\_intervals\_Average``,
while the RAP has a field named
``Total\_cloud\_cover\_entire\_atmosphere\_single\_layer``, and a similar
field in the HRRR is named ``Total\_cloud\_cover\_entire\_atmosphere``.

PVLIB-Python aims to simplify the access of the model fields relevant
for solar power forecasts. All models accessed via PVLIB-Python are
returned with uniform field names: ``temperature, wind\_speed,
total\_clouds, low\_clouds, mid\_clouds, high\_clouds, dni, dhi, ghi``. To
accomplish this, we use an object-oriented framework in which each
weather model is represented by a class that inherits from a parent
:py:ref:`~pvlib.forecast.ForecastModel` class.
The parent :py:ref:`~pvlib.forecast.ForecastModel` class
contains the common code for accessing and parsing the data using
Siphon, while the child model-specific classes contain the code
necessary to map and process that specific model's data to the
standardized fields.

The code below demonstrates how simple it is to access
and plot forecast data using PVLIB-Python.

.. ipython:: python

    import pandas as pd
    import matplotlib.pyplot as plt

    # seaborn makes the plots look nicer
    import seaborn as sns; sns.set_color_codes()

    # import forecast models
    from pvlib.forecast import GFS, NAM, NDFD, HRRR

    # specify location (Tucson, AZ)
    latitude, longitude, tz = 32.2, -110.9, 'US/Arizona'

    # specify time range
    start = pd.Timestamp.now(tz=tz)
    end = start + pd.Timedelta(days=7)

    # GFS model, defaults to 0.5 degree resolution
    # 0.25 deg available
    model = GFS()

    # retrieve data. returns pandas.DataFrame object
    data = model.get_query_data(latitude, longitude, start, end)

    # plot cloud cover percentages
    cloud_vars = ['total_clouds', 'low_clouds',
                  'mid_clouds', 'high_clouds']
    data[cloud_vars].plot();
    plt.ylabel('Cloud cover %');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('GFS 0.5 deg forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig gfs_cloud_cover.png width=6in
    plt.legend()


Cloud cover and radiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many of forecast models do not include radiation components in their output
fields, or if they do they suffer from poor solar
position calculations or radiative transfer algorithms.
It is often more accurate to create empirically derived
radiation forecasts from the weather models' cloud cover
forecasts. pvlib-python currently uses the Liu Jordan model
to convert cloud cover to radiation, however, we encourage
developers to explore alternatives.

PVLIB-Python currently uses the Liu-Jordan [Liu60] model to convert
cloud cover forecasts to irradiance forecasts, though it is fairly
simple to implement new models and provide additional options.
The figure below shows the result of the cloud cover to irradiance
conversion.

.. ipython: python

    # plot irradiance data
    irrad_vars = ['dni', 'ghi', 'dhi']
    data[irrad_vars].plot();
    plt.ylabel('Irradiance ($W/m^2$)');
    plt.xlabel('Forecast Time ({})'.format(tz));
    plt.title('GFS 0.5 deg forecast for lat={}, lon={}'
              .format(latitude, longitude));
    @savefig gfs_irrad.png width=6in
    plt.legend()

Note that the GFS data is hourly resolution, thus the
default irradiance forecasts are also hourly resolution. However, it is
straightforward to interpolate the cloud cover forecasts onto a higher
resolution time domain, and then recalculate the irradiance. We
reiterate that the open source, permissively licensed, and accessible
code enables users to customize the model processing to their liking.

.. [Liu60] B. Y. Liu and R. C. Jordan, The interrelationship and
    characteristic distribution of direct, diffuse, and total solar
    radiation, *Solar Energy* **4**, 1 (1960).

pvlib-python Forecast Module Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasts in pvlib-python aim to be:

* Simple and easy to use
* Comprehensive
* Flexible
* Integrated
* Standardized


pvlib-python's forecasting tools address a number of common issues with
weather model data:

* Data format dissimilarities between forecast models
	* Forecast period
		Many of the forecasts come at different intervals
		and span different lengths of time.
	* Variables provided
		The model share many of the same quantities,
		however they are labeled using different terms
		or need to be converted into useful values.
	* Data availability
		The models are updated a different intervals and
		also are sometimes missing data.

* Irradiance
	* Cloud cover and radiation
		Many of the forecast models do not have radiation
		fields, or if they do they suffer from poor solar
		position calculations or radiative transfer algorithms.
		It is often more accurate to create empirically derived
		radiation forecasts from the weather models' cloud cover
		forecasts. pvlib-python currently uses the Liu Jordan model
		to convert cloud cover to radiation, however, we encourage
		developers to explore alternatives.

.. math::

	DNI &= {\tau} ^m DNI_{ET} \\
	DHI &= 0.3(1 - {\tau} ^m)cos{\psi}DNI_{ET}



Forecast Module Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Model subclass
~~~~~~~~~~~~~~

Each forecast model has its own subclass.
These subclasses belong to a more comprehensive parent
class that holds many of the methods used by every model.

Within each subclass model specific variables are
assigned to common variable labels that are
available from each forecast model.

Here are the subclasses for two models.

.. image:: images/gfs.jpg
.. image:: images/ndfd.jpg


ForecastModel class
~~~~~~~~~~~~~~~~~~~

The following code is part of the parent class that
each forecast model belongs to.

.. image:: images/forecastmodel.jpg

Upon instatiation of a forecast model, several assignments are
made and functions called to initialize
values and objects within the class.

.. image:: images/fm_init.jpg

The query function is responsible for completing the retrieval
of data from the Unidata THREDDS server using
the Unidata siphon THREDDS server API.

.. image:: images/query.jpg

The ForecastModel class also contains miscellaneous functions
that process raw NetCDF data from the THREDDS
server and create a DataFrame for all the processed data.

.. image:: images/tempconvert.jpg

