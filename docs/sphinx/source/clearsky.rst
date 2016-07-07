.. _clearsky:

Clear sky
=========

Clear sky irradiance data is essential to many PV modeling tasks. Here, we
review the clear sky modeling capabilities of pvlib-python. The
:ref:`location` section demonstrates the easiest way to obtain a time
series of clear sky data for a location. The :ref:`ineichen` and
:ref:`simplified_solis` sections detail the clear sky algorithms and
input data.

We'll need these imports for the examples below.

.. ipython:: python

    import itertools
    import pandas as pd
    import matplotlib.pyplot as plt

    # seaborn makes the plots look nicer
    import seaborn as sns
    sns.set_color_codes()

    import pvlib
    from pvlib import clearsky
    from pvlib import atmosphere
    from pvlib.location import Location


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
attenuation.

.. ipython:: python

    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='2016-07-01', end='2016-07-04',
                             freq='1min', tz=tus.tz)
    cs = tus.get_clearsky(times)  # ineichen with climatology table by default
    cs.plot();
    plt.ylabel('Irradiance $W/m^2$');
    @savefig location-basic.png width=6in
    plt.title('Ineichen, climatological turbidity');

The :py:meth:`~pvlib.location.Location.get_clearsky` method accepts a
model keyword argument and propagates additional arguments to the
functions that do the computation.

.. ipython:: python

    cs = tus.get_clearsky(times, model='ineichen', linke_turbidity=3)
    cs.plot();
    plt.title('Ineichen, linke_turbidity=3');
    @savefig location-ineichen.png width=6in
    plt.ylabel('Irradiance $W/m^2$');

.. ipython:: python

    cs = tus.get_clearsky(times, model='simplified_solis',
                          aod700=0.2, precipitable_water=3)
    cs.plot();
    plt.title('Simplfied Solis, aod700=0.2, precipitable_water=3');
    @savefig location-solis.png width=6in
    plt.ylabel('Irradiance $W/m^2$');


See the sections below for more detail on the clear sky models.


.. _ineichen:

Ineichen
--------

The Ineichen and Perez clear sky model parameterizes irradiance
in terms of the Linke turbidity [Ine02]_.

Turbidity data
^^^^^^^^^^^^^^

pvlib includes a file with monthly climatological turbidity values for
the globe. The code below creates turbidity maps for a few months of
the year. You could run it in a loop to create plots for all months.

.. ipython:: python

    import calendar
    import os
    import scipy.io

    pvlib_path = os.path.dirname(os.path.abspath(pvlib.clearsky.__file__))
    filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.mat')

    mat = scipy.io.loadmat(filepath)
    linke_turbidity_table = mat['LinkeTurbidity'] / 20.

    month = 1
    plt.imshow(linke_turbidity_table[:, :, month-1], vmin=1, vmax=5);
    plt.title(calendar.month_name[1+month]);
    plt.colorbar(shrink=0.5);
    plt.tight_layout();
    @savefig turbidity-1.png width=10in
    plt.show();

.. ipython:: python

    month = 7
    plt.imshow(linke_turbidity_table[:, :, month-1], vmin=1, vmax=5);
    plt.title(calendar.month_name[month]);
    plt.colorbar(shrink=0.5);
    plt.tight_layout();
    @savefig turbidity-7.png width=10in
    plt.show();

The :py:func:`~pvlib.clearsky.lookup_linke_turbidity` function takes a
time, latitude, and longitude and gets the corresponding climatological
turbidity value for that time at those coordinates. By default, the
:py:func:`~pvlib.clearsky.lookup_linke_turbidity` function will linearly
interpolate turbidity from month to month. This removes discontinuities
in multi-month PV models. Here's a plot of a few locations in the
Southwest U.S. with and without interpolation. We have intentionally
shown points that are relatively close so that you can get a sense of
the variability of the data set.

.. ipython:: python

    times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1D')
    plt.figure();
    pvlib.clearsky.lookup_linke_turbidity(times, 32, -111, interp_turbidity=False).plot(label='Tucson1');
    pvlib.clearsky.lookup_linke_turbidity(times, 32.2, -110.9, interp_turbidity=False).plot(label='Tucson2');
    pvlib.clearsky.lookup_linke_turbidity(times, 33.5, -112.1, interp_turbidity=False).plot(label='Phoenix');
    pvlib.clearsky.lookup_linke_turbidity(times, 35.1, -106.6, interp_turbidity=False).plot(label='Albuquerque');
    plt.legend();
    @savefig turbidity-no-interp.png width=6in
    plt.ylabel('Linke Turbidity');

.. ipython:: python

    times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1D')
    pvlib.clearsky.lookup_linke_turbidity(times, 32, -111).plot(label='Tucson1');
    pvlib.clearsky.lookup_linke_turbidity(times, 32.2, -110.9).plot(label='Tucson2');
    pvlib.clearsky.lookup_linke_turbidity(times, 33.5, -112.1).plot(label='Phoenix');
    pvlib.clearsky.lookup_linke_turbidity(times, 35.1, -106.6).plot(label='Albuquerque');
    plt.legend();
    @savefig turbidity-yes-interp.png width=6in
    plt.ylabel('Linke Turbidity');

Examples
^^^^^^^^

.. ipython:: python



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
precipitable water and aerosol optical depth [Ine08ss]_.

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
contain one or both of aerosols and precipitable water.

Aerosol optical depth is a function of wavelength, and the Simplified
Solis model requires AOD at 700 nm. Models exist to convert AOD between
different wavelengths, as well as convert Linke turbidity to AOD and PW
[Ine08con]_, [Ine16]_.


Examples
^^^^^^^^

A clear sky time series using basic pvlib functions.

.. ipython:: python

    latitude, longitude, tz, altitude, name = 32.2, -111, 'US/Arizona', 700, 'Tucson'
    times = pd.date_range(start='2014-01-01', end='2014-01-02', freq='1Min', tz=tz)
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    apparent_elevation = solpos['apparent_elevation']
    aod700 = 0.1
    precipitable_water = 1
    pressure = pvlib.atmosphere.alt2pres(altitude)
    dni_extra = pvlib.irradiance.extraradiation(apparent_elevation.index.dayofyear)

    solis = clearsky.simplified_solis(apparent_elevation, aod700, precipitable_water,
                                      pressure, dni_extra)
    ax = solis.plot();
    ax.set_ylabel('Irradiance $W/m^2$');
    ax.legend(loc=2);
    @savefig solis-vs-time-0.1-1.png width=6in
    plt.show();


Irradiance as a function of solar elevation.

.. ipython:: python

    apparent_elevation = pd.Series(np.linspace(-10, 90, 101))
    aod700 = 0.1
    precipitable_water = 1
    pressure = 101325
    dni_extra = 1364

    solis = clearsky.simplified_solis(apparent_elevation, aod700,
                                      precipitable_water, pressure, dni_extra)
    ax = solis.plot()
    ax.set_xlabel('Apparent elevation (deg)');
    ax.set_ylabel('Irradiance $W/m^2$');
    ax.set_title('Irradiance vs Solar Elevation')
    @savefig solis-vs-elevation.png width=6in
    ax.legend(loc=2);


Grid with a clear sky irradiance for a few PW and AOD values.

.. ipython:: python

    times = pd.date_range(start='2014-09-01', end='2014-09-02', freq='1Min', tz=tz)
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    apparent_elevation = solpos['apparent_elevation']
    pressure = pvlib.atmosphere.alt2pres(altitude)
    dni_extra = pvlib.irradiance.extraradiation(apparent_elevation.index.dayofyear)
    aod700 = [0.01, 0.1]
    precipitable_water = [0.5, 5]

    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, squeeze=True)
    axes = axes.flatten()

    for (aod, pw), ax in zip(itertools.chain(itertools.product(aod700, precipitable_water)), axes):
        solis = clearsky.simplified_solis(apparent_elevation, aod, pw,
                                          pressure, dni_extra)
        solis.plot(ax=ax, title='aod700={}, pw={}'.format(aod, pw))

    @savefig solis-grid.png width=10in
    plt.show();

Contour plots of irradiance as a function of both PW and AOD.

.. ipython:: python

    aod700 = np.linspace(0, 0.5, 101)
    precipitable_water = np.linspace(0, 10, 101)
    apparent_elevation = 70
    pressure = 101325
    dni_extra = 1364

    aod700, precipitable_water = np.meshgrid(aod700, precipitable_water)

    solis = clearsky.simplified_solis(apparent_elevation, aod700,
                                      precipitable_water, pressure,
                                      dni_extra)
    cmap = plt.get_cmap('viridis')
    n = 15
    vmin = None
    vmax = None

    def plot_solis(key):
        irrad = solis[key]
        fig, ax = plt.subplots()
        im = ax.contour(aod700, precipitable_water, irrad[:, :], n, cmap=cmap, vmin=vmin, vmax=vmax)
        imf = ax.contourf(aod700, precipitable_water, irrad[:, :], n, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('AOD')
        ax.set_ylabel('Precipitable water (cm)')
        ax.clabel(im, colors='k', fmt='%.0f')
        fig.colorbar(imf, label='{} (W/m**2)'.format(key))
        ax.set_title('{}, elevation={}'.format(key, apparent_elevation))

.. ipython:: python

    plot_solis('ghi')
    @savefig solis-ghi.png width=10in
    plt.show()

    plot_solis('dni')
    @savefig solis-dni.png width=10in
    plt.show()

    plot_solis('dhi')
    @savefig solis-dhi.png width=10in
    plt.show()


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
