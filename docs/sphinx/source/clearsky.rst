.. _clearsky:

Clear sky
=========

This section reviews the clear sky modeling capabilities of
pvlib-python.

pvlib-python supports two ways to generate clear sky irradiance:

1. A :py:class:`~pvlib.location.Location` object's
   :py:meth:`~pvlib.location.Location.get_clearsky` method.
2. The functions contained within the :py:mod:`~pvlib.clearsky` module,
   including :py:func:`~pvlib.clearsky.ineichen` and
   :py:func:`~pvlib.clearsky.simplified_solis`.

Users that work with simple time series data may prefer to use
:py:meth:`~pvlib.location.Location.get_clearsky`, while users
that want finer control, more explicit code, or work with
multidimensional data may prefer to use the basic functions.

The :ref:`location` subsection demonstrates the easiest
way to obtain a time series of clear sky data for a location.
The :ref:`ineichen` and :ref:`simplified_solis` subsections detail the
clear sky algorithms and input data.

We'll need these imports for the examples below.

.. ipython::

    In [1]: import itertools

    In [1]: import matplotlib.pyplot as plt

    In [1]: import pandas as pd

    # seaborn makes the plots look nicer
    In [1]: import seaborn as sns

    In [1]: sns.set_color_codes()

    In [1]: import pvlib

    In [1]: from pvlib import clearsky, atmosphere

    In [1]: from pvlib.location import Location


.. _location:

Location
--------

The easiest way to obtain a time series of clear sky irradiance is to use a
:py:class:`~pvlib.location.Location` object's
:py:meth:`~pvlib.location.Location.get_clearsky` method. The
:py:meth:`~pvlib.location.Location.get_clearsky` method does the dirty
work of calculating solar position, extraterrestrial irradiance,
airmass, and atmospheric pressure, as appropriate, leaving the user to
only specify the most important parameters: time and atmospheric
attenuation. The time input must be a :py:class:`pandas.DatetimeIndex`,
while the atmospheric attenuation inputs may be constants or arrays.

.. ipython::

    In [1]: tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')

    In [1]: times = pd.DatetimeIndex(start='2016-07-01', end='2016-07-04', freq='1min', tz=tus.tz)

    In [1]: cs = tus.get_clearsky(times)  # ineichen with climatology table by default

    In [1]: cs.plot();

    In [1]: plt.ylabel('Irradiance $W/m^2$');

    @savefig location-basic.png width=6in
    In [1]: plt.title('Ineichen, climatological turbidity');

The :py:meth:`~pvlib.location.Location.get_clearsky` method accepts a
model keyword argument and propagates additional arguments to the
functions that do the computation.

.. ipython::

    In [1]: cs = tus.get_clearsky(times, model='ineichen', linke_turbidity=3)

    In [1]: cs.plot();

    In [1]: plt.title('Ineichen, linke_turbidity=3');

    @savefig location-ineichen.png width=6in
    In [1]: plt.ylabel('Irradiance $W/m^2$');

.. ipython::

    In [1]: cs = tus.get_clearsky(times, model='simplified_solis', aod700=0.2, precipitable_water=3)

    In [1]: cs.plot();

    In [1]: plt.title('Simplfied Solis, aod700=0.2, precipitable_water=3');

    @savefig location-solis.png width=6in
    In [1]: plt.ylabel('Irradiance $W/m^2$');


See the sections below for more detail on the clear sky models.


.. _ineichen:

Ineichen and Perez
------------------

The Ineichen and Perez clear sky model parameterizes irradiance in terms
of the Linke turbidity [Ine02]_. pvlib-python implements this model in
the :py:func:`pvlib.clearsky.ineichen` function.

Turbidity data
^^^^^^^^^^^^^^

pvlib includes a file with monthly climatological turbidity values for
the globe. The code below creates turbidity maps for a few months of
the year. You could run it in a loop to create plots for all months.

.. ipython::

    In [1]: import calendar

    In [1]: import os

    In [1]: import scipy.io

    In [1]: pvlib_path = os.path.dirname(os.path.abspath(pvlib.clearsky.__file__))

    In [1]: filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.mat')

    In [1]: mat = scipy.io.loadmat(filepath)

    In [1]: mat['LinkeTurbidity']

    In [1]: mat['LinkeTurbidity'] / 20.

.. code-block:: python

    # data is in units of 20 x turbidity
    In [1]: linke_turbidity_table = mat['LinkeTurbidity'] / 20.

    In [1]: month = 1

    In [1]: linke_turbidity_table

    In [1]: plt.figure();

    In [1]: plt.imshow(linke_turbidity_table[:, :, month-1], vmin=1, vmax=5);

    In [1]: plt.title('Linke turbidity, ' + calendar.month_name[month]);

    In [1]: plt.colorbar(shrink=0.5);

    In [1]: plt.tight_layout();

    @savefig turbidity-1.png width=10in
    In [1]: plt.show();

.. code-block:: python

    In [1]: month = 7

    In [1]: plt.figure();

    In [1]: plt.imshow(linke_turbidity_table[:, :, month-1], vmin=1, vmax=5);

    In [1]: plt.title('Linke turbidity, ' + calendar.month_name[month]);

    In [1]: plt.colorbar(shrink=0.5);

    In [1]: plt.tight_layout();

    @savefig turbidity-7.png width=10in
    In [1]: plt.show();

The :py:func:`~pvlib.clearsky.lookup_linke_turbidity` function takes a
time, latitude, and longitude and gets the corresponding climatological
turbidity value for that time at those coordinates. By default, the
:py:func:`~pvlib.clearsky.lookup_linke_turbidity` function will linearly
interpolate turbidity from month to month. This removes discontinuities
in multi-month PV models. Here's a plot of a few locations in the
Southwest U.S. with and without interpolation. We chose points that are
relatively close so that you can get a better sense of the spatial
variability of the data set.

.. ipython::

    In [1]: times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1D')

    In [1]: sites = [(32, -111, 'Tucson1'), (32.2, -110.9, 'Tucson2'),
       ...:          (33.5, -112.1, 'Phoenix'), (35.1, -106.6, 'Albuquerque')]

    In [1]: plt.figure();

    In [1]: for lat, lon, name in sites:
       ...:     turbidity = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon, interp_turbidity=False)
       ...:     turbidity.plot(label=name)

    In [1]: plt.legend();

    @savefig turbidity-no-interp.png width=6in
    In [1]: plt.ylabel('Linke Turbidity');

.. ipython::

    In [1]: plt.figure();

    In [1]: for lat, lon, name in sites:
       ...:     turbidity = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon)
       ...:     turbidity.plot(label=name)

    In [1]: plt.legend();

    @savefig turbidity-yes-interp.png width=6in
    In [1]: plt.ylabel('Linke Turbidity');

Examples
^^^^^^^^

A clear sky time series using basic pvlib functions.

.. ipython::

    In [1]: latitude, longitude, tz, altitude, name = 32.2, -111, 'US/Arizona', 700, 'Tucson'

    In [1]: times = pd.date_range(start='2014-01-01', end='2014-01-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_zenith = solpos['apparent_zenith']

    In [1]: airmass = pvlib.atmosphere.relativeairmass(apparent_zenith)

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: airmass = pvlib.atmosphere.absoluteairmass(airmass, pressure)

    In [1]: linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    In [1]: dni_extra = pvlib.irradiance.extraradiation(apparent_zenith.index.dayofyear)

    # an input is a pandas Series, so solis is a DataFrame
    In [1]: ineichen = clearsky.ineichen(apparent_zenith, airmass, linke_turbidity, altitude, dni_extra)

    In [1]: plt.figure();

    In [1]: ax = ineichen.plot()

    In [1]: ax.set_ylabel('Irradiance $W/m^2$');

    In [1]: ax.legend(loc=2);

    @savefig ineichen-vs-time-climo.png width=6in
    In [1]: plt.show();

The input data types determine the returned output type. Array input
results in an OrderedDict of array output, and Series input results in a
DataFrame output. The keys are 'ghi', 'dni', and 'dhi'.

Grid with a clear sky irradiance for a few turbidity values.

.. ipython::

    In [1]: times = pd.date_range(start='2014-09-01', end='2014-09-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_zenith = solpos['apparent_zenith']

    In [1]: airmass = pvlib.atmosphere.relativeairmass(apparent_zenith)

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: airmass = pvlib.atmosphere.absoluteairmass(airmass, pressure)

    In [1]: linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    In [1]: print('climatological linke_turbidity = {}'.format(linke_turbidity.mean()))

    In [1]: dni_extra = pvlib.irradiance.extraradiation(apparent_zenith.index.dayofyear)

    In [1]: linke_turbidities = [linke_turbidity.mean(), 2, 4]

    In [1]: fig, axes = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, squeeze=True, figsize=(12, 4))

    In [1]: axes = axes.flatten()

    In [1]: for linke_turbidity, ax in zip(linke_turbidities, axes):
       ...:     ineichen = clearsky.ineichen(apparent_zenith, airmass, linke_turbidity, altitude, dni_extra)
       ...:     ineichen.plot(ax=ax, title='Linke turbidity = {:0.1f}'.format(linke_turbidity));

    In [1]: ax.legend(loc=1);

    @savefig ineichen-grid.png width=10in
    In [1]: plt.show();



Validation
^^^^^^^^^^

See [Ine02]_, [Ren12]_.

Will Holmgren compared pvlib's Ineichen model and climatological
turbidity to `SoDa's McClear service
<http://www.soda-pro.com/web-services/radiation/cams-mcclear>`_ in
Arizona. Here are links to an
`ipynb notebook
<https://forecasting.energy.arizona.edu/media/ineichen_vs_mcclear.ipynb>`_
and its `html rendering
<https://forecasting.energy.arizona.edu/media/ineichen_vs_mcclear.html>`_.


.. _simplified_solis:

Simplified Solis
----------------

The Simplified Solis model parameterizes irradiance in terms of
precipitable water and aerosol optical depth [Ine08ss]_. pvlib-python
implements this model in the :py:func:`pvlib.clearsky.simplified_solis`
function.

Aerosol and precipitable water data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a number of sources for aerosol and precipitable water data
of varying accuracy, global coverage, and temporal resolution.
Ground based aerosol data can be obtained from
`Aeronet <http://aeronet.gsfc.nasa.gov>`_. Precipitable water can be obtained
from `radiosondes <http://weather.uwyo.edu/upperair/sounding.html>`_,
`ESRL GPS-MET <http://gpsmet.noaa.gov/cgi-bin/gnuplots/rti.cgi>`_, or
derived from surface relative humidity using functions such as
:py:func:`pvlib.atmosphere.gueymard94_pw`.
Numerous gridded products from satellites, weather models, and climate models
contain one or both of aerosols and precipitable water. Consider data
from the `ECMWF <https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets>`_
and `SoDa <http://www.soda-pro.com/web-services/radiation/cams-mcclear>`_.

Aerosol optical depth is a function of wavelength, and the Simplified
Solis model requires AOD at 700 nm. Models exist to convert AOD between
different wavelengths, as well as convert Linke turbidity to AOD and PW
[Ine08con]_, [Ine16]_.


Examples
^^^^^^^^

A clear sky time series using basic pvlib functions.

.. ipython::

    In [1]: latitude, longitude, tz, altitude, name = 32.2, -111, 'US/Arizona', 700, 'Tucson'

    In [1]: times = pd.date_range(start='2014-01-01', end='2014-01-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_elevation = solpos['apparent_elevation']

    In [1]: aod700 = 0.1

    In [1]: precipitable_water = 1

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: dni_extra = pvlib.irradiance.extraradiation(apparent_elevation.index.dayofyear)

    # an input is a Series, so solis is a DataFrame
    In [1]: solis = clearsky.simplified_solis(apparent_elevation, aod700, precipitable_water,
       ...:                                   pressure, dni_extra)

    In [1]: ax = solis.plot();

    In [1]: ax.set_ylabel('Irradiance $W/m^2$');

    In [1]: ax.legend(loc=2);

    @savefig solis-vs-time-0.1-1.png width=6in
    In [1]: plt.show();

The input data types determine the returned output type. Array input
results in an OrderedDict of array output, and Series input results in a
DataFrame output. The keys are 'ghi', 'dni', and 'dhi'.

Irradiance as a function of solar elevation.

.. ipython::

    In [1]: apparent_elevation = pd.Series(np.linspace(-10, 90, 101))

    In [1]: aod700 = 0.1

    In [1]: precipitable_water = 1

    In [1]: pressure = 101325

    In [1]: dni_extra = 1364

    In [1]: solis = clearsky.simplified_solis(apparent_elevation, aod700,
       ...:                                   precipitable_water, pressure, dni_extra)

    In [1]: ax = solis.plot();

    In [1]: ax.set_xlabel('Apparent elevation (deg)');

    In [1]: ax.set_ylabel('Irradiance $W/m^2$');

    In [1]: ax.set_title('Irradiance vs Solar Elevation')

    @savefig solis-vs-elevation.png width=6in
    In [1]: ax.legend(loc=2);


Grid with a clear sky irradiance for a few PW and AOD values.

.. ipython::

    In [1]: times = pd.date_range(start='2014-09-01', end='2014-09-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_elevation = solpos['apparent_elevation']

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: dni_extra = pvlib.irradiance.extraradiation(apparent_elevation.index.dayofyear)

    In [1]: aod700 = [0.01, 0.1]

    In [1]: precipitable_water = [0.5, 5]

    In [1]: fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, squeeze=True)

    In [1]: axes = axes.flatten()

    In [1]: [clearsky.simplified_solis(apparent_elevation, aod, pw, pressure, dni_extra).plot(ax=ax, title='aod700={}, pw={}'.format(aod, pw)) for (aod, pw), ax in zip(itertools.chain(itertools.product(aod700, precipitable_water)), axes)];

    @savefig solis-grid.png width=10in
    In [1]: plt.show();

Contour plots of irradiance as a function of both PW and AOD.

.. ipython::

    In [1]: aod700 = np.linspace(0, 0.5, 101)

    In [1]: precipitable_water = np.linspace(0, 10, 101)

    In [1]: apparent_elevation = 70

    In [1]: pressure = 101325

    In [1]: dni_extra = 1364

    In [1]: aod700, precipitable_water = np.meshgrid(aod700, precipitable_water)

    # inputs are arrays, so solis is an OrderedDict
    In [1]: solis = clearsky.simplified_solis(apparent_elevation, aod700,
       ...:                                   precipitable_water, pressure,
       ...:                                   dni_extra)

    In [1]: cmap = plt.get_cmap('viridis')

    In [1]: n = 15

    In [1]: vmin = None

    In [1]: vmax = None

    In [1]: def plot_solis(key):
       ...:     irrad = solis[key]
       ...:     fig, ax = plt.subplots()
       ...:     im = ax.contour(aod700, precipitable_water, irrad[:, :], n, cmap=cmap, vmin=vmin, vmax=vmax)
       ...:     imf = ax.contourf(aod700, precipitable_water, irrad[:, :], n, cmap=cmap, vmin=vmin, vmax=vmax)
       ...:     ax.set_xlabel('AOD')
       ...:     ax.set_ylabel('Precipitable water (cm)')
       ...:     ax.clabel(im, colors='k', fmt='%.0f')
       ...:     fig.colorbar(imf, label='{} (W/m**2)'.format(key))
       ...:     ax.set_title('{}, elevation={}'.format(key, apparent_elevation))

.. ipython::

    In [1]: plot_solis('ghi')

    @savefig solis-ghi.png width=10in
    In [1]: plt.show()

    In [1]: plot_solis('dni')

    @savefig solis-dni.png width=10in
    In [1]: plt.show()

    In [1]: plot_solis('dhi')

    @savefig solis-dhi.png width=10in
    In [1]: plt.show()


Validation
^^^^^^^^^^

See [Ine16]_.

We encourage users to compare the pvlib implementation to Ineichen's
`Excel tool <http://www.unige.ch/energie/fr/equipe/ineichen/solis-tool/>`_.


References
----------

.. [Ine02] P. Ineichen and R. Perez, "A New airmass independent formulation for
   the Linke turbidity coefficient", Solar Energy, 73, pp. 151-157,
   2002.

.. [Ine08ss] P. Ineichen, "A broadband simplified version of the
   Solis clear sky model," Solar Energy, 82, 758-762 (2008).

.. [Ine16] P. Ineichen, "Validation of models that estimate the clear
   sky global and beam solar irradiance," Solar Energy, 132,
   332-344 (2016).

.. [Ine08con] P. Ineichen, "Conversion function between the Linke turbidity
   and the atmospheric water vapor and aerosol content", Solar Energy,
   82, 1095 (2008).

.. [Ren12] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
   Sky Models: Implementation and Analysis", Sandia National
   Laboratories, SAND2012-2389, 2012.
