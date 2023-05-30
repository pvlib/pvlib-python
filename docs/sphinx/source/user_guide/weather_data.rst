.. _weatherdata:

Importing weather data
======================

Many PV modeling applications require irradiance and meteorological datasets
as the inputs to a PV system model.  These weather datasets are available
from many sources and in many formats.  The :py:mod:`pvlib.iotools` module
contains functions to retrieve and import these datasets in a form
that is convenient to use with the rest of pvlib.


Usage
-----

Although there are some exceptions, the :py:mod:`pvlib.iotools` functions
provide a uniform interface for reading data files in many common formats.
Specifically, :py:mod:`pvlib.iotools` functions usually return two objects:
a :py:class:`pandas.DataFrame` of the actual dataset and, plus a metadata
dictionary.  Most :py:mod:`pvlib.iotools` functions also have
a ``map_variables`` parameter to automatically translate
the column names used in the data file (which vary widely across datasets)
into standard pvlib names (see :ref:`variables_style_rules`).  Typical usage
looks like this:

.. code-block:: python

    df, metadata = pvlib.iotools.function(..., map_variables=True)


Most :py:mod:`pvlib.iotools` functions work with time series datasets.
In that case, the returned ``df`` DataFrame has a datetime index, localized
to the appropriate time zone where possible.  The metadata dictionary
varies based on the function/dataset being used.

For the full list of available :py:mod:`pvlib.iotools` functions, see
:ref:`iotools`.


File readers
------------

Some weather data file formats have internal structure that requires
more than just a call to :py:func:`pandas.read_csv`.  pvlib provides
functions for reading files in many of these formats.  Functions that
read and parse files in a particular format have names that begin with ``read_``:
:py:func:`~pvlib.iotools.read_tmy3`, :py:func:`~pvlib.iotools.read_epw`, and so on.


Online APIs
-----------

Several :py:mod:`pvlib.iotools` functions access the internet to fetch data from
external web APIs.  For example, :py:func:`~pvlib.iotools.get_pvgis_hourly`
downloads data from PVGIS's webservers and returns it as a python variable.
Functions that retrieve data from the internet have names that begin with
``get_``: :py:func:`~pvlib.iotools.get_bsrn`, :py:func:`~pvlib.iotools.get_psm3`,
:py:func:`~pvlib.iotools.get_pvgis_tmy`, and so on.

Some of these data providers require registration.  In those cases, your
access credentials must be passed as parameters to the function.  See the
individual function documentation pages for details.

