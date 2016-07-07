.. _clearsky:

Clear sky
=========

Clear sky expectations are essential to many PV modeling tasks.
Here, we briefly review the clear sky modeling capabilities of pvlib-python.

.. ipython:: python

    import datetime
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


Location
--------

The easiest way to get clear sky data is to use a
:py:class:`~pvlib.location.Location` object's
:py:meth:`~pvlib.location.Location.get_clearsky` method. The
:py:meth:`~pvlib.location.Location.get_clearsky` method does the dirty
work of calculating solar position, extraterrestrial irradiance,
airmass, and atmospheric pressure, as appropriate, leaving the user to
only specify the most important parameters. We encourage users to
examine the source code.

.. ipython:: python

    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='2016-07-01', end='2016-07-04', freq='1min', tz=tus.tz)
    cs = tus.get_clearsky(times)  # ineichen with lookup table by default
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
    plt.ylabel('Irradiance $W/m^2$')

.. ipython:: python

    cs = tus.get_clearsky(times, model='simplified_solis',
                          aod700=0.2, precipitable_water=3)
    cs.plot();
    plt.title('Simplfied Solis, aod700=0.2, precipitable_water=3');
    @savefig location-solis.png width=6in
    plt.ylabel('Irradiance $W/m^2$');


Ineichen
--------

The Ineichen and Perez (2002) model has proven to be a popular model.
The model parameterizes irradiance in terms of the Linke turbidity.

Turbidity
^^^^^^^^^

The code below creates turbidity plots for a few months of the year.
You could run it in a loop to create plots for all months.

.. ipython:: python

    import calendar
    import os
    import scipy.io

    pvlib_path = os.path.dirname(os.path.abspath(pvlib.clearsky.__file__))
    filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.mat')

    mat = scipy.io.loadmat(filepath)
    linke_turbidity_table = mat['LinkeTurbidity'] / 20

    month = 1
    plt.figure(figsize=(20,10));
    plt.imshow(linke_turbidity_table[:, :, month-1], vmin=1, vmax=5);
    plt.title(calendar.month_name[1+month]);
    @savefig turbidity-1.png width=10in
    plt.colorbar();

.. ipython:: python

    month = 7
    plt.figure(figsize=(20,10));
    plt.imshow(linke_turbidity_table[:, :, month-1], vmin=1, vmax=5);
    plt.title(calendar.month_name[month]);
    @savefig turbidity-7.png width=10in
    plt.colorbar();


Here's a plot of selected areas in the Southwest U.S. We have
intentionally shown points that are relatively close so that you can get
a sense of the variability of the data set.

.. ipython:: python

    times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1D')
    plt.figure()
    pvlib.clearsky.lookup_linke_turbidity(times, 32, -111, interp_turbidity=False).plot(label='Tucson1');
    pvlib.clearsky.lookup_linke_turbidity(times, 32.2, -110.9, interp_turbidity=False).plot(label='Tucson2');
    pvlib.clearsky.lookup_linke_turbidity(times, 33.5, -112.1, interp_turbidity=False).plot(label='Phoenix');
    pvlib.clearsky.lookup_linke_turbidity(times, 35.1, -106.6, interp_turbidity=False).plot(label='Albuquerque');
    plt.legend();
    @savefig turbidity-no-interp.png width=6in
    plt.ylabel('Linke Turbidity');

By default, the :py:func:`~pvlib.clearsky.lookup_linke_turbidity`
function will linearly interpolate turbidity from month to month. This
removes discontinuities in multi-month PV models. See the source code
for details.

.. ipython:: python

    times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1D')
    pvlib.clearsky.lookup_linke_turbidity(times, 32, -111).plot(label='Tucson1');
    pvlib.clearsky.lookup_linke_turbidity(times, 32.2, -110.9).plot(label='Tucson2');
    pvlib.clearsky.lookup_linke_turbidity(times, 33.5, -112.1).plot(label='Phoenix');
    pvlib.clearsky.lookup_linke_turbidity(times, 35.1, -106.6).plot(label='Albuquerque');
    plt.legend();
    @savefig turbidity-yes-interp.png width=6in
    plt.ylabel('Linke Turbidity');

Will Holmgren did some rough analysis comparing pvlib's ineichen model
to `SoDa's McClear service
<http://www.soda-pro.com/web-services/radiation/cams-mcclear>`_ in
Arizona. Here are links to an
`ipynb notebook
<https://forecasting.energy.arizona.edu/media/ineichen_vs_mcclear.ipynb>`_
and its `html rendering
<https://forecasting.uaren.org/media/ineichen_vs_mcclear.html>`_.


Simplified Solis
----------------

The Simplified Solis model parameterizes irradiance in terms of
precipitable water and aerosol optical depth.

Ground based aerosol data can be obtained from
`Aeronet <http://aeronet.gsfc.nasa.gov>`_. Precipitable water can be obtained
from `ESRL <http://gpsmet.noaa.gov/cgi-bin/gnuplots/rti.cgi>`_.

.. ipython:: python

    aod700 = 0.1
    precipitable_water = 1
    apparent_elevation = pd.Series(np.linspace(-10, 90, 101))
    pressure = 101325
    dni_extra = 1364

    solis = clearsky.simplified_solis(apparent_elevation, aod700,
                                      precipitable_water, pressure, dni_extra)
    ax = solis.plot()
    ax.set_xlabel('apparent elevation (deg)');
    ax.set_ylabel('irradiance (W/m**2)');
    @savefig solis-vs-elevation.png width=6in
    ax.legend(loc=2);


.. ipython:: python

    from pvlib.location import Location

    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.date_range(start=datetime.datetime(2014,1,1), end=datetime.datetime(2014,1,2), freq='1Min').tz_localize(tus.tz)
    solpos = pvlib.solarposition.get_solarposition(times, tus.latitude, tus.longitude)
    ephem_data = solpos

    aod700 = 0.1
    precipitable_water = 1
    apparent_elevation = solpos['apparent_elevation']
    pressure = pvlib.atmosphere.alt2pres(tus.altitude)
    dni_extra = pvlib.irradiance.extraradiation(apparent_elevation.index.dayofyear)

    solis = clearsky.simplified_solis(apparent_elevation, aod700, precipitable_water, pressure, dni_extra)
    @savefig solis-vs-time.png width=6in
    solis.plot();



.. ipython:: python

    times = pd.date_range(start=datetime.datetime(2014,9,1), end=datetime.datetime(2014,9,2), freq='1Min').tz_localize(tus.tz)
    solpos = pvlib.solarposition.get_solarposition(times, tus.latitude, tus.longitude)
    ephem_data = solpos

    apparent_elevation = solpos['apparent_elevation']
    pressure = pvlib.atmosphere.alt2pres(tus.altitude)
    dni_extra = pvlib.irradiance.extraradiation(apparent_elevation.index.dayofyear)

    aod700 = [0.01, 0.1]
    precipitable_water = [0.5, 5]

    for aod, pw in itertools.product(aod700, precipitable_water):
        solis = clearsky.simplified_solis(apparent_elevation, aod, pw,
                                          pressure, dni_extra)
        fig, ax = plt.subplots()
        solis.plot(ax=ax, title='aod700={}, pw={}'.format(aod, pw))
        ax.set_ylim(0, 1100)
        file = 'aod{}_pw{}.png'.format(aod, pw)
        @savefig file width=6in
        plt.show()


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
        fig, ax = plt.subplots(figsize=(12,9))
        im = ax.contour(aod700, precipitable_water, irrad[:, :], n, cmap=cmap, vmin=vmin, vmax=vmax)
        imf = ax.contourf(aod700, precipitable_water, irrad[:, :], n, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('AOD')
        ax.set_ylabel('Precipitable water (cm)')
        ax.clabel(im, colors='k', fmt='%.0f')
        fig.colorbar(imf, label='{} (W/m**2)'.format(key))
        ax.set_title('{}, elevation={}'.format(key, apparent_elevation))

.. ipython:: python

    plot_solis('ghi')
    @savefig solis-ghi.png width=6in
    plt.show()

    plot_solis('dni')
    @savefig solis-dni.png width=6in
    plt.show()

    plot_solis('dhi')
    @savefig solis-dhi.png width=6in
    plt.show()

We encourage users to compare the pvlib implementation to Ineichen's
`Excel tool <http://www.unige.ch/energie/fr/equipe/ineichen/solis-tool/>`_.
