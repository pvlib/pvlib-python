.. _timetimezones:

Time and time zones
===================

Dealing with time and time zones can be a frustrating experience in any
programming language and for any application. pvlib-python relies on
:py:mod:`pandas` and `pytz <http://pythonhosted.org/pytz/>`_ to handle
time and time zones. Therefore, the vast majority of the information in
this document applies to any time series analysis using pandas and is
not specific to pvlib-python.

General functionality
---------------------

pvlib makes extensive use of pandas due to its excellent time series
functionality. Take the time to become familiar with pandas' `Time
Series / Date functionality page
<http://pandas.pydata.org/pandas-docs/version/0.18.0/timeseries.html>`_.
It is also worthwhile to become familiar with pure Python's
:py:mod:`python:datetime` module, although we usually recommend
using the corresponding pandas functionality where possible.

First, we'll import the libraries that we'll use to explore the basic
time and time zone functionality in python and pvlib.

.. ipython:: python

    import datetime
    import pandas as pd
    import pytz


Finding a time zone
*******************

pytz is based on the Olson time zone database. You can obtain a list of
all valid time zone strings with ``pytz.all_timezones``. It's a long
list, so we only print every 20th time zone.

.. ipython:: python

    len(pytz.all_timezones)
    pytz.all_timezones[::20]

Wikipedia's `List of tz database time zones
<https://en.wikipedia.org/wiki/List_of_tz_database_time_zones>`_ is also
good reference.

The ``pytz.country_timezones`` function is useful, too.

.. ipython:: python

    pytz.country_timezones('US')

And don't forget about Python's :py:func:`python:filter` function.

.. ipython:: python

    list(filter(lambda x: 'GMT' in x, pytz.all_timezones))

Note that while pytz has ``'EST'`` and ``'MST'``, it does not have
``'PST'``. Use ``'Etc/GMT+8'`` instead, or see :ref:`fixedoffsets`.

Timestamps
**********

:py:class:`pandas.Timestamp` and :py:class:`pandas.DatetimeIndex`
can be created in many ways. Here we focus on the time zone issues
surrounding them; see the pandas documentation for more information.

First, create a time zone naive pandas.Timestamp.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00')

You can specify the time zone using the ``tz`` keyword argument or the
``tz_localize`` method of Timestamp and DatetimeIndex objects.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00', tz='America/Denver')
    pd.Timestamp('2015-1-1 00:00').tz_localize('America/Denver')

Localized Timestamps can be converted from one time zone to another.

.. ipython:: python

    midnight_mst = pd.Timestamp('2015-1-1 00:00', tz='America/Denver')
    corresponding_utc = midnight_mst.tz_convert('UTC')  # returns a new Timestamp
    corresponding_utc

It does not make sense to convert a time stamp that has not been
localized, and pandas will raise an exception if you try to do so.

.. ipython:: python
   :okexcept:

    midnight = pd.Timestamp('2015-1-1 00:00')
    midnight.tz_convert('UTC')

The difference between ``tz_localize`` and ``tz_convert`` is a common
source of confusion for new users. Just remember: localize first,
convert later.

Daylight savings time
*********************

Some time zones are aware of daylight savings time and some are not. For
example the winter time results are the same for US/Mountain and MST,
but the summer time results are not.

Note the UTC offset in winter...

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00').tz_localize('US/Mountain')
    pd.Timestamp('2015-1-1 00:00').tz_localize('Etc/GMT+7')

vs. the UTC offset in summer...

.. ipython:: python

    pd.Timestamp('2015-6-1 00:00').tz_localize('US/Mountain')
    pd.Timestamp('2015-6-1 00:00').tz_localize('Etc/GMT+7')

pandas and pytz make this time zone handling possible because pandas
stores all times as integer nanoseconds since January 1, 1970.
Here is the pandas time representation of the integers 1 and 1e9.

.. ipython:: python

    pd.Timestamp(1)
    pd.Timestamp(1e9)

So if we specify times consistent with the specified time zone, pandas
will use the same integer to represent them.

.. ipython:: python

    # US/Mountain
    pd.Timestamp('2015-6-1 01:00', tz='US/Mountain').value

    # MST
    pd.Timestamp('2015-6-1 00:00', tz='Etc/GMT+7').value

    # Europe/Berlin
    pd.Timestamp('2015-6-1 09:00', tz='Europe/Berlin').value

    # UTC
    pd.Timestamp('2015-6-1 07:00', tz='UTC').value

    # UTC
    pd.Timestamp('2015-6-1 07:00').value

It's ultimately these integers that are used when calculating quantities
in pvlib such as solar position.

As stated above, pandas will assume UTC if you do not specify a time
zone. This is dangerous, and we recommend using localized timeseries,
even if it is UTC.


.. _fixedoffsets:

Fixed offsets
*************

The ``'Etc/GMT*'`` time zones mentioned above provide fixed offset
specifications, but watch out for the counter-intuitive sign convention.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00', tz='Etc/GMT-2')

Fixed offset time zones can also be specified as offset minutes
from UTC using ``pytz.FixedOffset``.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00', tz=pytz.FixedOffset(120))

You can also specify the fixed offset directly in the ``tz_localize``
method, however, be aware that this is not documented and that the
offset must be in seconds, not minutes.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00', tz=7200)

Yet another way to specify a time zone with a fixed offset is by using
the string formulation.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00+0200')


Native Python objects
*********************

Sometimes it's convenient to use native Python
:py:class:`python:datetime.date` and
:py:class:`python:datetime.datetime` objects, so we demonstrate their
use next. pandas Timestamp objects can also be created from time zone
aware or naive
:py:class:`python:datetime.datetime` objects. The behavior is as
expected.

.. ipython:: python

    # tz naive python datetime.datetime object
    naive_python_dt = datetime.datetime(2015, 6, 1, 0)

    # tz naive pandas Timestamp object
    pd.Timestamp(naive_python_dt)

    # tz aware python datetime.datetime object
    aware_python_dt = pytz.timezone('US/Mountain').localize(naive_python_dt)

    # tz aware pandas Timestamp object
    pd.Timestamp(aware_python_dt)

One thing to watch out for is that python
:py:class:`python:datetime.date` objects gain time information when
passed to ``Timestamp``.

.. ipython:: python

    # tz naive python datetime.date object (no time info)
    naive_python_date = datetime.date(2015, 6, 1)

    # tz naive pandas Timestamp object (time=midnight)
    pd.Timestamp(naive_python_date)

You cannot localize a native Python date object.

.. ipython:: python
   :okexcept:

    # fail
    pytz.timezone('US/Mountain').localize(naive_python_date)


pvlib-specific functionality
----------------------------

.. note::

    This section applies to pvlib >= 0.3. Version 0.2 of pvlib used a
    ``Location`` object's ``tz`` attribute to auto-magically correct for
    some time zone issues. This behavior was counter-intuitive to many
    users and was removed in version 0.3.

How does this general functionality interact with pvlib? Perhaps the two
most common places to get tripped up with time and time zone issues in
solar power analysis occur during data import and solar position
calculations.

Data import
***********

Let's first examine how pvlib handles time when it imports a TMY3 file.

.. ipython:: python

    import os
    import inspect
    import pvlib

    # some gymnastics to find the example file
    pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))
    file_abspath = os.path.join(pvlib_abspath, 'data', '703165TY.csv')
    tmy3_data, tmy3_metadata = pvlib.iotools.read_tmy3(file_abspath)

    tmy3_metadata

The metadata has a ``'TZ'`` key with a value of ``-9.0``. This is the
UTC offset in hours in which the data has been recorded. The
:py:func:`~pvlib.tmy.readtmy3` function read the data in the file,
created a :py:class:`~pandas.DataFrame` with that data, and then
localized the DataFrame's index to have this fixed offset. Here, we
print just a few of the rows and columns of the large dataframe.

.. ipython:: python

    tmy3_data.index.tz

    tmy3_data.loc[tmy3_data.index[0:3], ['GHI', 'DNI', 'AOD']]

The :py:func:`~pvlib.tmy.readtmy2` function also returns a DataFrame
with a localized DatetimeIndex.

Solar position
**************

The correct solar position can be immediately calculated from the
DataFrame's index since the index has been localized.

.. ipython:: python

    solar_position = pvlib.solarposition.get_solarposition(tmy3_data.index,
                                                           tmy3_metadata['latitude'],
                                                           tmy3_metadata['longitude'])

    ax = solar_position.loc[solar_position.index[0:24], ['apparent_zenith', 'apparent_elevation', 'azimuth']].plot()

    ax.legend(loc=1);
    ax.axhline(0, color='darkgray');  # add 0 deg line for sunrise/sunset
    ax.axhline(180, color='darkgray');  # add 180 deg line for azimuth at solar noon
    ax.set_ylim(-60, 200);  # zoom in, but cuts off full azimuth range
    ax.set_xlabel('Local time ({})'.format(solar_position.index.tz));
    @savefig solar-position.png width=6in
    ax.set_ylabel('(degrees)');

`According to the US Navy
<http://aa.usno.navy.mil/rstt/onedaytable?ID=AA&year=1997&month=1&day=1&state=AK&place=sand+point>`_,
on January 1, 1997 at Sand Point, Alaska, sunrise was at 10:09 am, solar
noon was at 1:46 pm, and sunset was at 5:23 pm. This is consistent with
the data plotted above (and depressing).

Solar position (assumed UTC)
****************************

What if we had a DatetimeIndex that was not localized, such as the one
below? The solar position calculator will assume UTC time.

.. ipython:: python

    index = pd.date_range(start='1997-01-01 01:00', freq='1h', periods=24)
    index

    solar_position_notz = pvlib.solarposition.get_solarposition(index,
                                                                tmy3_metadata['latitude'],
                                                                tmy3_metadata['longitude'])

    ax = solar_position_notz.loc[solar_position_notz.index[0:24], ['apparent_zenith', 'apparent_elevation', 'azimuth']].plot()

    ax.legend(loc=1);
    ax.axhline(0, color='darkgray');  # add 0 deg line for sunrise/sunset
    ax.axhline(180, color='darkgray');  # add 180 deg line for azimuth at solar noon
    ax.set_ylim(-60, 200);  # zoom in, but cuts off full azimuth range
    ax.set_xlabel('Time (UTC)');
    @savefig solar-position-nolocal.png width=6in
    ax.set_ylabel('(degrees)');

This looks like the plot above, but shifted by 9 hours.

Solar position (calculate and convert)
**************************************

In principle, one could localize the tz-naive solar position data to
UTC, and then convert it to the desired time zone.

.. ipython:: python

    fixed_tz = pytz.FixedOffset(tmy3_metadata['TZ'] * 60)
    solar_position_hack = solar_position_notz.tz_localize('UTC').tz_convert(fixed_tz)

    solar_position_hack.index

    ax = solar_position_hack.loc[solar_position_hack.index[0:24], ['apparent_zenith', 'apparent_elevation', 'azimuth']].plot()

    ax.legend(loc=1);
    ax.axhline(0, color='darkgray');  # add 0 deg line for sunrise/sunset
    ax.axhline(180, color='darkgray');  # add 180 deg line for azimuth at solar noon
    ax.set_ylim(-60, 200);  # zoom in, but cuts off full azimuth range
    ax.set_xlabel('Local time ({})'.format(solar_position_hack.index.tz));
    @savefig solar-position-hack.png width=6in
    ax.set_ylabel('(degrees)');

Note that the time has been correctly localized and converted, however,
the calculation bounds still correspond to the original assumed-UTC range.

For this and other reasons, we recommend that users supply time zone
information at the beginning of a calculation rather than localizing and
converting the results at the end of a calculation.
