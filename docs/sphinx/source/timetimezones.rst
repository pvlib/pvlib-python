.. _timetimezones:

Time and time zones
===================

Dealing with time and time zones can be a frustrating experience in any
programming language and for any application. pvlib relies on :py:mod:`pandas`
and
pytz to handle time and time zones. Therefore, the vast majority of the
information in this document applies to any time series analysis using
pandas and is not specific to pvlib-python.

pvlib makes extensive use of pandas due to its excellent time series
functionality. Take the time to become familiar with pandas' `Time
Series / Date functionality page
<http://pandas.pydata.org/pandas-docs/version/0.18.0/timeseries.html>`_.
It is also worthwhile to become familiar with pure Python's
:py:mod:`python:datetime` module, although we typically recommend
using the corresponding pandas functionality where it exists.

.. ipython:: python

    import datetime
    import numpy as np
    import pandas as pd
    import pytz

You can obtain a list of all valid time zone strings with
pytz.all_timezones. Here, we print only every 20th time zone.

.. ipython:: python

    len(pytz.all_timezones)
    pytz.all_timezones[::20]

:py:class:`pandas.Timestamp`'s and :py:class:`pandas.DatetimeIndex`'s
can be created in many ways. Here
we focus on the time zone issues surrounding them; see the pandas
documentation for more information.

First, create a time zone naive pandas.Timestamp.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00')

You can specify the time zone using the tz keyword argument or
the tz_localize method of Timestamp
and DatetimeIndex objects.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00', tz='America/Denver')
    pd.Timestamp('2015-1-1 00:00').tz_localize('America/Denver')

Some time zones are aware of daylight savings time and some are not. For
example the winter time results are the same for US/Mountain and MST,
but the summer time results are not.

Note the UTC offset in winter...

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00').tz_localize('US/Mountain')
    pd.Timestamp('2015-1-1 00:00').tz_localize('MST')

vs. the UTC offset in summer...

.. ipython:: python

    pd.Timestamp('2015-6-1 00:00').tz_localize('US/Mountain')
    pd.Timestamp('2015-6-1 00:00').tz_localize('MST')

pandas and pytz make this time zone handling possible because pandas
stores all times as integer nanoseconds since January 1, 1970.
Here is the pandas time representation of the integer 1.

.. ipython:: python

    pd.Timestamp(1)

So if we specify times consistent with the specified time zone, pandas
will use the same integer to represent them.


.. ipython:: python

    # US/Mountain
    pd.Timestamp('2015-6-1 01:00').tz_localize('US/Mountain').value

    # MST
    pd.Timestamp('2015-6-1 00:00').tz_localize('MST').value

    # UTC
    pd.Timestamp('2015-6-1 07:00').tz_localize('UTC').value

    # UTC
    pd.Timestamp('2015-6-1 07:00').value

As stated above, pandas will assume UTC if you do not specify a time
zone. This is dangerous, and we always recommend using using localized
timeseries, even if it is UTC.

Timezones can also be specified with a fixed offset in minutes from UTC.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00').tz_localize(pytz.FixedOffset(120))

You can also specify the fixed offset directly in the tz_localize
method, however, be aware that this is not documented and that the
offset must be in seconds, not minutes.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00').tz_localize(7200)

Yet another way to specify a time zone with a fixed offset is by using
the string formulation.

.. ipython:: python

    pd.Timestamp('2015-1-1 00:00+0200')

pandas time objects can also be created from time zone aware or naive
datetime.date or datetime.datetime objects. The behavior is generally as
expected.

.. ipython:: python

    # tz naive
    pd.Timestamp(datetime.datetime(2015,6,1,0))

    # start is tz aware python datetime object
    start = pytz.timezone('US/Mountain').localize(datetime.datetime(2015, 6, 1, 0))
    pd.Timestamp(start)

