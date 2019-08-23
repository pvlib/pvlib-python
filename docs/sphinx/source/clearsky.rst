.. _clearsky:

Clear sky
=========

This section reviews the clear sky modeling capabilities of
pvlib-python.

pvlib-python supports two ways to generate clear sky irradiance:

1. A :py:class:`~pvlib.location.Location` object's
   :py:meth:`~pvlib.location.Location.get_clearsky` method.
2. The functions contained in the :py:mod:`~pvlib.clearsky` module,
   including :py:func:`~pvlib.clearsky.ineichen` and
   :py:func:`~pvlib.clearsky.simplified_solis`.

Users that work with simple time series data may prefer to use
:py:meth:`~pvlib.location.Location.get_clearsky`, while users
that want finer control, more explicit code, or work with
multidimensional data may prefer to use the basic functions in the
:py:mod:`~pvlib.clearsky` module.

The :ref:`location` subsection demonstrates the easiest way to obtain a
time series of clear sky data for a location. The :ref:`ineichen` and
:ref:`simplified_solis` subsections detail the clear sky algorithms and
input data. The :ref:`detect_clearsky` subsection demonstrates the use
of the clear sky detection algorithm.

We'll need these imports for the examples below.

.. ipython::

    In [1]: import os

    In [1]: import itertools

    In [1]: import matplotlib.pyplot as plt

    In [1]: import pandas as pd

    In [1]: import pvlib

    In [1]: from pvlib import clearsky, atmosphere, solarposition

    In [1]: from pvlib.location import Location

    In [1]: from pvlib.iotools import read_tmy3


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
The :py:meth:`~pvlib.location.Location.get_clearsky` method always
returns a :py:class:`pandas.DataFrame`.

.. ipython::

    In [1]: tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')

    In [1]: times = pd.date_range(start='2016-07-01', end='2016-07-04', freq='1min', tz=tus.tz)

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

    In [1]: import tables

    In [1]: pvlib_path = os.path.dirname(os.path.abspath(pvlib.clearsky.__file__))

    In [1]: filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.h5')

    In [1]: def plot_turbidity_map(month, vmin=1, vmax=100):
       ...:     plt.figure();
       ...:     with tables.open_file(filepath) as lt_h5_file:
       ...:         ltdata = lt_h5_file.root.LinkeTurbidity[:, :, month-1]
       ...:     plt.imshow(ltdata, vmin=vmin, vmax=vmax);
       ...:     # data is in units of 20 x turbidity
       ...:     plt.title('Linke turbidity x 20, ' + calendar.month_name[month]);
       ...:     plt.colorbar(shrink=0.5);
       ...:     plt.tight_layout();

    @savefig turbidity-1.png width=10in
    In [1]: plot_turbidity_map(1)

    @savefig turbidity-7.png width=10in
    In [1]: plot_turbidity_map(7)

The :py:func:`~pvlib.clearsky.lookup_linke_turbidity` function takes a
time, latitude, and longitude and gets the corresponding climatological
turbidity value for that time at those coordinates. By default, the
:py:func:`~pvlib.clearsky.lookup_linke_turbidity` function will linearly
interpolate turbidity from month to month, assuming that the raw data is
valid on 15th of each month. This interpolation removes discontinuities
in multi-month PV models. Here's a plot of a few locations in the
Southwest U.S. with and without interpolation. We chose points that are
relatively close so that you can get a better sense of the spatial noise
and variability of the data set. Note that the altitude of these sites
varies from 300 m to 1500 m.

.. ipython::

    In [1]: times = pd.date_range(start='2015-01-01', end='2016-01-01', freq='1D')

    In [1]: sites = [(32, -111, 'Tucson1'), (32.2, -110.9, 'Tucson2'),
       ...:          (33.5, -112.1, 'Phoenix'), (35.1, -106.6, 'Albuquerque')]

    In [1]: plt.figure();

    In [1]: for lat, lon, name in sites:
       ...:     turbidity = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon, interp_turbidity=False)
       ...:     turbidity.plot(label=name)

    In [1]: plt.legend();

    In [1]: plt.title('Raw data (no interpolation)');

    @savefig turbidity-no-interp.png width=6in
    In [1]: plt.ylabel('Linke Turbidity');

    In [1]: plt.figure();

    In [1]: for lat, lon, name in sites:
       ...:     turbidity = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon)
       ...:     turbidity.plot(label=name)

    In [1]: plt.legend();

    In [1]: plt.title('Interpolated to the day');

    @savefig turbidity-yes-interp.png width=6in
    In [1]: plt.ylabel('Linke Turbidity');

The :py:func:`~pvlib.atmosphere.kasten96_lt` function can be used to calculate
Linke turbidity [Kas96]_ as input to the clear sky Ineichen and Perez function.
The Kasten formulation requires precipitable water and broadband aerosol
optical depth (AOD). According to Molineaux, broadband AOD can be approximated
by a single measurement at 700-nm [Mol98]_. An alternate broadband AOD
approximation from Bird and Hulstrom combines AOD measured at two
wavelengths [Bir80]_, and is implemented in
:py:func:`~pvlib.atmosphere.bird_hulstrom80_aod_bb`.

.. ipython::

    In [1]: pvlib_data = os.path.join(os.path.dirname(pvlib.__file__), 'data')

    In [1]: mbars = 100  # conversion factor from mbars to Pa

    In [1]: tmy_file = os.path.join(pvlib_data, '703165TY.csv')  # TMY file

    In [1]: tmy_data, tmy_header = read_tmy3(tmy_file, coerce_year=1999)  # read TMY data

    In [1]: tl_historic = clearsky.lookup_linke_turbidity(time=tmy_data.index,
       ...:     latitude=tmy_header['latitude'], longitude=tmy_header['longitude'])

    In [1]: solpos = solarposition.get_solarposition(time=tmy_data.index,
       ...:     latitude=tmy_header['latitude'], longitude=tmy_header['longitude'],
       ...:     altitude=tmy_header['altitude'], pressure=tmy_data['Pressure']*mbars,
       ...:     temperature=tmy_data['DryBulb'])

    In [1]: am_rel = atmosphere.get_relative_airmass(solpos.apparent_zenith)

    In [1]: am_abs = atmosphere.get_absolute_airmass(am_rel, tmy_data['Pressure']*mbars)

    In [1]: airmass = pd.concat([am_rel, am_abs], axis=1).rename(
       ...:     columns={0: 'airmass_relative', 1: 'airmass_absolute'})

    In [1]: tl_calculated = atmosphere.kasten96_lt(
       ...:     airmass.airmass_absolute, tmy_data['Pwat'], tmy_data['AOD'])

    In [1]: tl = pd.concat([tl_historic, tl_calculated], axis=1).rename(
       ...:     columns={0:'Historic', 1:'Calculated'})

    In [1]: tl.index = tmy_data.index.tz_convert(None)  # remove timezone

    In [1]: tl.resample('W').mean().plot();

    In [1]: plt.grid()

    In [1]: plt.title('Comparison of Historic Linke Turbidity Factors vs. \n'
       ...:     'Kasten Pyrheliometric Formula at {name:s}, {state:s} ({usaf:d}TY)'.format(
       ...:     name=tmy_header['Name'], state=tmy_header['State'], usaf=tmy_header['USAF']));

    In [1]: plt.ylabel('Linke Turbidity Factor, TL');

    @savefig kasten-tl.png width=10in
    In [1]: plt.tight_layout()


Examples
^^^^^^^^

A clear sky time series using only basic pvlib functions.

.. ipython::

    In [1]: latitude, longitude, tz, altitude, name = 32.2, -111, 'US/Arizona', 700, 'Tucson'

    In [1]: times = pd.date_range(start='2014-01-01', end='2014-01-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_zenith = solpos['apparent_zenith']

    In [1]: airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

    In [1]: linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    In [1]: dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # an input is a pandas Series, so solis is a DataFrame
    In [1]: ineichen = clearsky.ineichen(apparent_zenith, airmass, linke_turbidity, altitude, dni_extra)

    In [1]: plt.figure();

    In [1]: ax = ineichen.plot()

    In [1]: ax.set_ylabel('Irradiance $W/m^2$');

    In [1]: ax.set_title('Ineichen Clear Sky Model');

    @savefig ineichen-vs-time-climo.png width=6in
    In [1]: ax.legend(loc=2);


The input data types determine the returned output type. Array input
results in an OrderedDict of array output, and Series input results in a
DataFrame output. The keys are 'ghi', 'dni', and 'dhi'.

Grid with a clear sky irradiance for a few turbidity values.

.. ipython::

    In [1]: times = pd.date_range(start='2014-09-01', end='2014-09-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_zenith = solpos['apparent_zenith']

    In [1]: airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

    In [1]: linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    In [1]: print('climatological linke_turbidity = {}'.format(linke_turbidity.mean()))

    In [1]: dni_extra = pvlib.irradiance.get_extra_radiation(times)

    In [1]: linke_turbidities = [linke_turbidity.mean(), 2, 4]

    In [1]: fig, axes = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, squeeze=True, figsize=(12, 4))

    In [1]: axes = axes.flatten()

    In [1]: for linke_turbidity, ax in zip(linke_turbidities, axes):
       ...:     ineichen = clearsky.ineichen(apparent_zenith, airmass, linke_turbidity, altitude, dni_extra)
       ...:     ineichen.plot(ax=ax, title='Linke turbidity = {:0.1f}'.format(linke_turbidity));

    @savefig ineichen-grid.png width=10in
    In [1]: ax.legend(loc=1);

    @suppress
    In [1]: plt.close();


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

Aerosol optical depth (AOD) is a function of wavelength, and the Simplified
Solis model requires AOD at 700 nm.
:py:func:`~pvlib.atmosphere.angstrom_aod_at_lambda` is useful for converting
AOD between different wavelengths using the Angstrom turbidity model. The
Angstrom exponent, :math:`\alpha`, can be calculated from AOD at two
wavelengths with :py:func:`~pvlib.atmosphere.angstrom_alpha`.
[Ine08con]_, [Ine16]_, [Ang61]_.

.. ipython::

    In [1]: aod1240nm = 1.2  # fictitious AOD measured at 1240-nm

    In [1]: aod550nm = 3.1  # fictitious AOD measured at 550-nm

    In [1]: alpha_exponent = atmosphere.angstrom_alpha(aod1240nm, 1240, aod550nm, 550)

    In [1]: aod700nm = atmosphere.angstrom_aod_at_lambda(aod1240nm, 1240, alpha_exponent, 700)

    In [1]: aod380nm = atmosphere.angstrom_aod_at_lambda(aod550nm, 550, alpha_exponent, 380)

    In [1]: aod500nm = atmosphere.angstrom_aod_at_lambda(aod550nm, 550, alpha_exponent, 500)

    In [1]: aod_bb = atmosphere.bird_hulstrom80_aod_bb(aod380nm, aod500nm)

    In [1]: print('compare AOD at 700-nm = {:g}, to estimated broadband AOD = {:g}, '
       ...:     'with alpha = {:g}'.format(aod700nm, aod_bb, alpha_exponent))

Examples
^^^^^^^^

A clear sky time series using only basic pvlib functions.

.. ipython::

    In [1]: latitude, longitude, tz, altitude, name = 32.2, -111, 'US/Arizona', 700, 'Tucson'

    In [1]: times = pd.date_range(start='2014-01-01', end='2014-01-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_elevation = solpos['apparent_elevation']

    In [1]: aod700 = 0.1

    In [1]: precipitable_water = 1

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # an input is a Series, so solis is a DataFrame
    In [1]: solis = clearsky.simplified_solis(apparent_elevation, aod700, precipitable_water,
       ...:                                   pressure, dni_extra)

    In [1]: ax = solis.plot();

    In [1]: ax.set_ylabel('Irradiance $W/m^2$');

    In [1]: ax.set_title('Simplified Solis Clear Sky Model');

    @savefig solis-vs-time-0.1-1.png width=6in
    In [1]: ax.legend(loc=2);

    @suppress
    In [1]: plt.close();

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

    @suppress
    In [1]: plt.close();


Grid with clear sky irradiance for a few PW and AOD values.

.. ipython::

    In [1]: times = pd.date_range(start='2014-09-01', end='2014-09-02', freq='1Min', tz=tz)

    In [1]: solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    In [1]: apparent_elevation = solpos['apparent_elevation']

    In [1]: pressure = pvlib.atmosphere.alt2pres(altitude)

    In [1]: dni_extra = pvlib.irradiance.get_extra_radiation(times)

    In [1]: aod700 = [0.01, 0.1]

    In [1]: precipitable_water = [0.5, 5]

    In [1]: fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, squeeze=True)

    In [1]: axes = axes.flatten()

    @savefig solis-grid.png width=10in
    In [1]: for (aod, pw), ax in zip(itertools.chain(itertools.product(aod700, precipitable_water)), axes):
       ...:     cs = clearsky.simplified_solis(apparent_elevation, aod, pw, pressure, dni_extra)
       ...:     cs.plot(ax=ax, title='aod700={}, pw={}'.format(aod, pw))

    @suppress
    In [1]: plt.close();

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

    In [1]: n = 15

    In [1]: vmin = None

    In [1]: vmax = None

    In [1]: def plot_solis(key):
       ...:     irrad = solis[key]
       ...:     fig, ax = plt.subplots()
       ...:     im = ax.contour(aod700, precipitable_water, irrad[:, :], n, vmin=vmin, vmax=vmax)
       ...:     imf = ax.contourf(aod700, precipitable_water, irrad[:, :], n, vmin=vmin, vmax=vmax)
       ...:     ax.set_xlabel('AOD')
       ...:     ax.set_ylabel('Precipitable water (cm)')
       ...:     ax.clabel(im, colors='k', fmt='%.0f')
       ...:     fig.colorbar(imf, label='{} (W/m**2)'.format(key))
       ...:     ax.set_title('{}, elevation={}'.format(key, apparent_elevation))

.. ipython::

    @savefig solis-ghi.png width=10in
    In [1]: plot_solis('ghi')

    @suppress
    In [1]: plt.close();

    @savefig solis-dni.png width=10in
    In [1]: plot_solis('dni')

    @suppress
    In [1]: plt.close();

    @savefig solis-dhi.png width=10in
    In [1]: plot_solis('dhi')

    @suppress
    In [1]: plt.close();


Validation
^^^^^^^^^^

See [Ine16]_.

We encourage users to compare the pvlib implementation to Ineichen's
`Excel tool <http://www.unige.ch/energie/fr/equipe/ineichen/solis-tool/>`_.

.. _detect_clearsky:

Detect Clearsky
---------------

The :py:func:`~pvlib.clearsky.detect_clearsky` function implements the
[Ren16]_ algorithm to detect the clear and cloudy points of a time
series. The algorithm was designed and validated for analyzing GHI time
series only. Users may attempt to apply it to other types of time series
data using different filter settings, but should be skeptical of the
results.

The algorithm detects clear sky times by comparing statistics for a
measured time series and an expected clearsky time series. Statistics
are calculated using a sliding time window (e.g., 10 minutes). An
iterative algorithm identifies clear periods, uses the identified
periods to estimate bias in the clearsky data, scales the clearsky data
and repeats.

Clear times are identified by meeting 5 criteria. Default values for
these thresholds are appropriate for 10 minute windows of 1 minute GHI
data.

Next, we show a simple example of applying the algorithm to synthetic
GHI data. We first generate and plot the clear sky and measured data.

.. ipython:: python

    abq = Location(35.04, -106.62, altitude=1619)

    times = pd.date_range(start='2012-04-01 10:30:00', tz='Etc/GMT+7', periods=30, freq='1min')

    cs = abq.get_clearsky(times)

    # scale clear sky data to account for possibility of different turbidity
    ghi = cs['ghi']*.953

    # add a cloud event
    ghi['2012-04-01 10:42:00':'2012-04-01 10:44:00'] = [500, 300, 400]

    # add an overirradiance event
    ghi['2012-04-01 10:56:00'] = 950

    fig, ax = plt.subplots()

    ghi.plot(label='input');

    cs['ghi'].plot(label='ineichen clear');

    ax.set_ylabel('Irradiance $W/m^2$');

    @savefig detect-clear-ghi.png width=10in
    plt.legend(loc=4);

    @suppress
    plt.close();

Now we run the synthetic data and clear sky estimate through the
:py:func:`~pvlib.clearsky.detect_clearsky` function.

.. ipython:: python

    clear_samples = clearsky.detect_clearsky(ghi, cs['ghi'], cs.index, 10)

    fig, ax = plt.subplots()

    clear_samples.astype(int).plot();

    @savefig detect-clear-detected.png width=10in
    ax.set_ylabel('Clear (1) or Cloudy (0)');

    @suppress
    plt.close();

The algorithm detected the cloud event and the overirradiance event.


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

.. [Ren16] Reno, M.J. and C.W. Hansen, "Identification of periods of clear
   sky irradiance in time series of GHI measurements" Renewable Energy,
   v90, p. 520-531, 2016.

.. [Mol98] B. Molineaux, P. Ineichen, and N. O’Neill, “Equivalence of
   pyrheliometric and monochromatic aerosol optical depths at a single key
   wavelength.,” Appl. Opt., vol. 37, no. 30, pp. 7008–18, Oct. 1998.

.. [Kas96] F. Kasten, “The linke turbidity factor based on improved values
   of the integral Rayleigh optical thickness,” Sol. Energy, vol. 56, no. 3,
   pp. 239–244, Mar. 1996.

.. [Bir80] R. E. Bird and R. L. Hulstrom, “Direct Insolation Models,”
   1980.

.. [Ang61] A. ÅNGSTRÖM, “Techniques of Determinig the Turbidity of the
   Atmosphere,” Tellus A, vol. 13, no. 2, pp. 214–223, 1961.
