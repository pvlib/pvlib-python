.. _weatherdata:

Weather data
============

Simulating the performance of a PV system requires irradiance and meteorological data
as the inputs to a PV system model.  Weather datasets are available
from many sources and in many formats.  The :py:mod:`pvlib.iotools` module
contains functions to easily retrieve and import such datasets in a standardized
form that is convenient to use with the rest of pvlib.  For a complete list
of functions related to retrieving and importing weather data, see :ref:`iotools`.


Types of weather data sources
-----------------------------

Ground station measurements
***************************

From in-situ monitoring equipment. If properly maintained and quality-controlled,
these are the highest quality source of weather information. However, the coverage
depends on a weather station having been set up in advance for the location and
times of interest. There are datasets such as BSRN and SURFRAD which make their
measurement data publicly available.


Numerical Weather Prediction (NWP)
**********************************

These are mathematical simulations of weather systems. The data quality is much
lower than that of measurements, owing in part to coarser spatial and temporal
resolution, as well as many models not being optimised for solar irradiance for
PV applications. On the plus side, these models typically have worldwide coverage,
with some regional models (e.g. HRRR) sacrifice global coverage for somewhat higher
spatial and temporal resolution. Various forecast (e.g. GFS, ECMWF, ICON) and
reanalysis sources (ERA5, MERRA2) exist.


Satellite Data
**************

These sources process satellite imagery (typically from geostationary satellites)
to identify and classify clouds, and combine this with solar irradiance models to
produce irradiance estimates. The quality is generally much higher than NWP, but
still not as good as a well-maintained weather station. They have high spatial
and temporal resolution corresponding to the source satellite imagery, and are
generally optimised to estimate solar irradiance for PV applications. Free sources
such as PVGIS are available, and commerical sources such as SolarAnywhere,
Solcast and Solargis provide paid options though often have free trials.


:py:mod:`pvlib.iotools` usage
-----------------------------

With some exceptions, the :py:mod:`pvlib.iotools` functions
provide a uniform interface for accessing data across many formats.
Specifically, :py:mod:`pvlib.iotools` functions usually return two objects:
a :py:class:`pandas.DataFrame` of the actual dataset, plus a metadata
dictionary.  Most :py:mod:`pvlib.iotools` functions also have
a ``map_variables`` parameter to automatically translate
the column names used in the data file (which vary widely from dataset to dataset)
into standard pvlib names (see :ref:`variables_style_rules`).  Typical usage
looks like this:

.. code-block:: python

    # reading a local data file:
    df, metadata = pvlib.iotools.read_XYZ(filepath, map_variables=True, ...)
    
    # retrieving data from an online service
    df, metadata = pvlib.iotools.get_XYZ(location, date_range, map_variables=True, ...)


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
Functions that retrieve data from the internet have names that begin with
``get_``: :py:func:`~pvlib.iotools.get_bsrn`, :py:func:`~pvlib.iotools.get_psm3`,
:py:func:`~pvlib.iotools.get_pvgis_tmy`, and so on.

Some of these data providers require registration.  In those cases, your
access credentials must be passed as parameters to the function.  See the
individual function documentation pages for details.


Reading local files
*******************

:py:mod:`pvlib.iotools` also provides functions for parsing data files
stored locally on your computer.
Functions that read and parse files in a particular format have names
that begin with ``read_``: :py:func:`~pvlib.iotools.read_tmy3`,
:py:func:`~pvlib.iotools.read_epw`, and so on.


References
----------
.. [1] Jensen et al. "pvlib iotools—Open-source Python functions for seamless
   access to solar irradiance data". Solar Energy, 2023.
   :doi:`10.1016/j.solener.2023.112092`.
