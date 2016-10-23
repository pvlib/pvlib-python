"""
Collection of functions used in pvlib_python
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytz


def cosd(angle):
    """
    Cosine with angle input in degrees

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Cosine of the angle
    """

    res = np.cos(np.radians(angle))
    return res


def sind(angle):
    """
    Sine with angle input in degrees

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Sin of the angle
    """

    res = np.sin(np.radians(angle))
    return res


def tand(angle):
    """
    Tan with angle input in degrees

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Tan of the angle
    """

    res = np.tan(np.radians(angle))
    return res


def asind(number):
    """
    Inverse Sine returning an angle in degrees

    Parameters
    ----------
    number : float
        Input number

    Returns
    -------
    result : float
        arcsin result
    """

    res = np.degrees(np.arcsin(number))
    return res


def localize_to_utc(time, location):
    """
    Converts or localizes a time series to UTC.

    Parameters
    ----------
    time : datetime.datetime, pandas.DatetimeIndex,
           or pandas.Series/DataFrame with a DatetimeIndex.
    location : pvlib.Location object

    Returns
    -------
    pandas object localized to UTC.
    """
    import datetime as dt
    import pytz

    if isinstance(time, dt.datetime):
        if time.tzinfo is None:
            time = pytz.timezone(location.tz).localize(time)
        time_utc = time.astimezone(pytz.utc)
    else:
        try:
            time_utc = time.tz_convert('UTC')
        except TypeError:
            time_utc = time.tz_localize(location.tz).tz_convert('UTC')

    return time_utc


def datetime_to_djd(time):
    """
    Converts a datetime to the Dublin Julian Day

    Parameters
    ----------
    time : datetime.datetime
        time to convert

    Returns
    -------
    float
        fractional days since 12/31/1899+0000
    """

    if time.tzinfo is None:
        time_utc = pytz.utc.localize(time)
    else:
        time_utc = time.astimezone(pytz.utc)

    djd_start = pytz.utc.localize(dt.datetime(1899, 12, 31, 12))
    djd = (time_utc - djd_start).total_seconds() * 1.0/(60 * 60 * 24)

    return djd


def djd_to_datetime(djd, tz='UTC'):
    """
    Converts a Dublin Julian Day float to a datetime.datetime object

    Parameters
    ----------
    djd : float
        fractional days since 12/31/1899+0000
    tz : str
        timezone to localize the result to

    Returns
    -------
    datetime.datetime
       The resultant datetime localized to tz
    """

    djd_start = pytz.utc.localize(dt.datetime(1899, 12, 31, 12))

    utc_time = djd_start + dt.timedelta(days=djd)
    return utc_time.astimezone(pytz.timezone(tz))


def _pandas_to_doy(pd_object):
    """
    Finds the day of year for a pandas datetime-like object.

    Useful for delayed evaluation of the dayofyear attribute.

    Parameters
    ----------
    pd_object : DatetimeIndex or Timestamp

    Returns
    -------
    dayofyear
    """
    return pd_object.dayofyear


def _doy_to_datetimeindex(doy, epoch_year=2014):
    """
    Convert a day of year scalar or array to a pd.DatetimeIndex.

    Parameters
    ----------
    doy : numeric
        Contains days of the year

    Returns
    -------
    pd.DatetimeIndex
    """
    doy = np.atleast_1d(doy).astype('float')
    epoch = pd.Timestamp('{}-12-31'.format(epoch_year - 1))
    timestamps = [epoch + dt.timedelta(days=adoy) for adoy in doy]
    return pd.DatetimeIndex(timestamps)


def _datetimelike_scalar_to_doy(time):
    return pd.DatetimeIndex([pd.Timestamp(time)]).dayofyear


def _datetimelike_scalar_to_datetimeindex(time):
    return pd.DatetimeIndex([pd.Timestamp(time)])


def _scalar_out(input):
    if np.isscalar(input):
        output = input
    else:  #
        # works if it's a 1 length array and
        # will throw a ValueError otherwise
        output = np.asscalar(input)

    return output


def _array_out(input):
    if isinstance(input, pd.Series):
        output = input.values
    else:
        output = input

    return output


def _build_kwargs(keys, input_dict):
    """
    Parameters
    ----------
    keys : iterable
        Typically a list of strings.
    adict : dict-like
        A dictionary from which to attempt to pull each key.

    Returns
    -------
    kwargs : dict
        A dictionary with only the keys that were in input_dict
    """

    kwargs = {}
    for key in keys:
        try:
            kwargs[key] = input_dict[key]
        except KeyError:
            pass

    return kwargs
