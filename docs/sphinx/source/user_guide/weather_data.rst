.. _weatherdata:

Weather data
============

Simulating the performance of a PV system requires irradiance and meteorological data
as the inputs to a PV system model.  Weather datasets are available
from many sources and in many formats.  The :py:mod:`pvlib.iotools` module
contains functions to easily retrieve and import such datasets in a standardized
form that is convenient to use with the rest of pvlib.  

The primary focus of :py:mod:`pvlib.iotools` is time series solar resource
data like the irradiance datasets from PVGIS and the NSRDB, but it also provides
functionality for other types of data useful for certain aspects of PV modeling.
For example, precipitation data is available from :py:func:`~pvlib.iotools.get_acis_prism`
for soiling modeling and horizon profiles can be retrieved from
:py:func:`~pvlib.iotools.get_pvgis_horizon` for horizon shade modeling.

For a complete list of functions related to retrieving and importing weather
data, see :ref:`iotools`, and for a detailed comparison of the weather datasets
available through pvlib, see [1]_.


Types of weather data sources
-----------------------------

Weather data can be grouped into a few fundamental categories.  Which
type is most useful depends on the application.  Here we provide a high-level
overview of different types of weather data, and when you might want to use
them.

1. **Ground station measurements**:
   From in-situ monitoring equipment. If properly maintained and
   quality-controlled, these are the highest quality
   source of weather information. However, the coverage depends on
   a weather station having been set up in advance for the location and
   times of interest. Some ground station networks like the BSRN and SURFRAD
   make their measurement data publicly available. An global overview of ground
   stations is available at https://SolarStations.org.
   
   Data from public ground station measurement networks are useful if you
   want accurate, high-resolution data but have flexibility around the
   specific measurement location.

2. **Satellite data**: 
   These sources process satellite imagery (typically from geostationary
   satellites) to identify and classify clouds, and combine this with solar
   irradiance models and aerosol data to produce irradiance estimates. The
   quality is generally much higher than NWP, but still not as good as a well-maintained
   weather station. They have high spatial and temporal resolution
   corresponding to the source satellite imagery, and are generally
   optimised to estimate solar irradiance for PV applications. Free sources
   such as PVGIS, NSRDB, and CAMS are available, and commerical sources such
   as SolarAnywhere, Solcast, and Solargis provide paid options though often
   have free trials.
   
   Satellite data is useful when suitable ground measurements are
   not available for the location and/or times of interest.

3. **Numerical Weather Prediction (NWP)**:
   These are mathematical simulations of weather systems.
   The data quality is much lower than that of measurements and
   satellite data, owing in part to coarser spatial and temporal
   resolution, as well as many models not being optimised for solar
   irradiance for PV applications. On the plus side, these models typically
   have worldwide coverage, with some regional models (e.g. HRRR) sacrifice
   global coverage for somewhat higher spatial and temporal resolution.
   Various forecast (e.g. GFS, ECMWF, ICON) and reanalysis sources (ERA5,
   MERRA2) exist.
   
   NWP datasets are primarily useful for parts of the world not covered
   by satellite-based datasets (e.g. polar regions) or if extremely long time
   ranges are needed.


Usage
-----

With some exceptions, the :py:mod:`pvlib.iotools` functions
provide a uniform interface for accessing data across many formats.
Specifically, :py:mod:`pvlib.iotools` functions usually return two objects:
a :py:class:`pandas.DataFrame` of the actual dataset, plus a metadata
dictionary.  Most :py:mod:`pvlib.iotools` functions also have
a ``map_variables`` parameter to automatically translate
the column names used in the data file (which vary widely from dataset to dataset)
into standard pvlib names (see :ref:`variables_style_rules`).  

Typical usage looks something like this:

.. code-block:: python

    # get_pvgis_tmy returns two additional values besides df and metadata
    df, _, _, metadata = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)

This code will fetch a Typical Meteorological Year (TMY) dataset from PVGIS,
returning a :py:class:`pandas.DataFrame` containing the hourly weather data
and a python dict with information about the dataset.

Most :py:mod:`pvlib.iotools` functions work with time series datasets.
In that case, the returned ``df`` DataFrame has a datetime index, localized
to the appropriate time zone where possible.  Make sure to understand each
dataset's timestamping convention (e.g. center versus end of interval), as
pvlib will use these timestamps for solar position calculations.

The content of the metadata dictionary varies for each function/dataset.


Data retrieval
**************

Several :py:mod:`pvlib.iotools` functions access the internet to fetch data from
online web APIs.  For example, :py:func:`~pvlib.iotools.get_pvgis_hourly`
downloads data from PVGIS's webservers and returns it as a python variable.
Functions that retrieve data from the internet are named ``get_``, followed
by the name of the data source: :py:func:`~pvlib.iotools.get_bsrn`,
:py:func:`~pvlib.iotools.get_psm3`, :py:func:`~pvlib.iotools.get_pvgis_tmy`,
and so on.

For satellite/reanalysis datasets, the location is specified by latitude and
longitude in decimal degrees:

.. code-block:: python

    latitude, longitude = 33.75, -84.39  # Atlanta, Georgia, United States
    df, metadata = pvlib.iotools.get_psm3(latitude, longitude, map_variables=True, ...)


For ground station networks, the location identifier is the station ID:

.. code-block:: python

    df, metadata = pvlib.iotools.get_bsrn(station='cab', start='2020-01-01', end='2020-01-31', ...)

Some of these data providers require registration.  In those cases, your
access credentials must be passed as parameters to the function.  See the
individual function documentation pages for details.


Reading local files
*******************

:py:mod:`pvlib.iotools` also provides functions for parsing data files
stored locally on your computer.
Functions that read and parse local data files are named ``read_``, followed by
the name of the file format they parse: :py:func:`~pvlib.iotools.read_tmy3`,
:py:func:`~pvlib.iotools.read_epw`, and so on.

For example, here is how to read a file in the TMY3 file format:

.. code-block:: python

    df, metadata = pvlib.iotools.read_tmy3(r"C:\path\to\file.csv", map_variables=True)


References
----------
.. [1] Jensen et al. "pvlib iotools—Open-source Python functions for seamless
   access to solar irradiance data". Solar Energy, 2023.
   :doi:`10.1016/j.solener.2023.112092`.
