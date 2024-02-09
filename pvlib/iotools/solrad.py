"""Functions to read data from the NOAA SOLRAD network.
"""

import numpy as np
import pandas as pd
import urllib
import warnings

# pvlib conventions
BASE_HEADERS = (
    'year', 'julian_day', 'month', 'day', 'hour', 'minute', 'decimal_time',
    'solar_zenith', 'ghi', 'ghi_flag', 'dni', 'dni_flag', 'dhi', 'dhi_flag',
    'uvb', 'uvb_flag', 'uvb_temp', 'uvb_temp_flag'
)

# following README_SOLRAD.txt variable names for remaining
STD_HEADERS = ('std_dw_psp', 'std_direct', 'std_diffuse', 'std_uvb')

HEADERS = BASE_HEADERS + STD_HEADERS

DPIR_HEADERS = ('dpir', 'dpir_flag', 'dpirc', 'dpirc_flag', 'dpird',
                'dpird_flag')

MADISON_HEADERS = BASE_HEADERS + DPIR_HEADERS + STD_HEADERS + (
    'std_dpir', 'std_dpirc', 'std_dpird')


# as specified in README_SOLRAD.txt file. excludes 1 space between columns
WIDTHS = [4, 3] + 4*[2] + [6, 6] + 5*[7, 1] + 4*[9]
MADISON_WIDTHS = [4, 3] + 4*[2] + [6, 6] + 8*[7, 1] + 7*[9]
# add 1 to make fields contiguous (required by pandas.read_fwf)
WIDTHS = [w + 1 for w in WIDTHS]
MADISON_WIDTHS = [w + 1 for w in MADISON_WIDTHS]
# no space after last column
WIDTHS[-1] -= 1
MADISON_WIDTHS[-1] -= 1

DTYPES = [
    'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64',
    'float64', 'float64', 'int64', 'float64', 'int64', 'float64', 'int64',
    'float64', 'int64', 'float64', 'int64', 'float64', 'float64',
    'float64', 'float64']

MADISON_DTYPES = [
    'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64',
    'float64', 'int64', 'float64', 'int64', 'float64', 'int64', 'float64',
    'int64', 'float64', 'int64', 'float64', 'int64', 'float64', 'int64',
    'float64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64',
    'float64', 'float64']


def read_solrad(filename):
    """
    Read NOAA SOLRAD fixed-width file into pandas dataframe.  The SOLRAD
    network is described in [1]_ and [2]_.

    Parameters
    ----------
    filename: str
        filepath or url to read for the fixed-width file.

    Returns
    -------
    data: Dataframe
        A dataframe with DatetimeIndex and all of the variables in the
        file.

    See Also
    --------
    get_solrad

    Notes
    -----
    SOLRAD data resolution is described by the README_SOLRAD.txt:
    "Before 1-jan. 2015 the data were reported as 3-min averages;
    on and after 1-Jan. 2015, SOLRAD data are reported as 1-min.
    averages of 1-sec. samples."
    Here, missing data is flagged as NaN, rather than -9999.9.

    References
    ----------
    .. [1] NOAA SOLRAD Network
       `https://www.esrl.noaa.gov/gmd/grad/solrad/index.html
       <https://www.esrl.noaa.gov/gmd/grad/solrad/index.html>`_

    .. [2] B. B. Hicks et. al., (1996), The NOAA Integrated Surface
       Irradiance Study (ISIS). A New Surface Radiation Monitoring
       Program. Bull. Amer. Meteor. Soc., 77, 2857-2864.
       :doi:`10.1175/1520-0477(1996)077<2857:TNISIS>2.0.CO;2`
    """
    if 'msn' in str(filename):
        names = MADISON_HEADERS
        widths = MADISON_WIDTHS
        dtypes = MADISON_DTYPES
    else:
        names = HEADERS
        widths = WIDTHS
        dtypes = DTYPES

    # read in data
    data = pd.read_fwf(filename, header=None, skiprows=2, names=names,
                       widths=widths, na_values=-9999.9)

    # loop here because dtype kwarg not supported in read_fwf until 0.20
    for (col, _dtype) in zip(data.columns, dtypes):
        ser = data[col].astype(_dtype)
        if _dtype == 'float64':
            # older verions of pandas/numpy read '-9999.9' as
            # -9999.8999999999996 and fail to set nan in read_fwf,
            # so manually set nan
            ser = ser.where(ser > -9999, other=np.nan)
        data[col] = ser

    # set index
    # columns do not have leading 0s, so must zfill(2) to comply
    # with %m%d%H%M format
    dts = data[['month', 'day', 'hour', 'minute']].astype(str).apply(
        lambda x: x.str.zfill(2))
    dtindex = pd.to_datetime(
        data['year'].astype(str) + dts['month'] + dts['day'] + dts['hour'] +
        dts['minute'], format='%Y%m%d%H%M', utc=True)
    data = data.set_index(dtindex)
    try:
        # to_datetime(utc=True) does not work in older versions of pandas
        data = data.tz_localize('UTC')
    except TypeError:
        pass

    return data


def get_solrad(station, start, end,
               url="https://gml.noaa.gov/aftp/data/radiation/solrad/"):
    """Request data from NOAA SOLRAD and read it into a Dataframe.

    A list of stations and their descriptions can be found in [1]_,
    The data files are described in [2]_.

    Data is returned for complete days, including ``start`` and ``end``.

    Parameters
    ----------
    station : str
        Three letter station abbreviation.
    start : datetime-like
        First day of the requested period
    end : datetime-like
        Last day of the requested period
    url : str, default: 'https://gml.noaa.gov/aftp/data/radiation/solrad/'
        API endpoint URL

    Returns
    -------
    data : pd.DataFrame
        Dataframe with data from SOLRAD.
    meta : dict
        Metadata.

    See Also
    --------
    read_solrad

    Notes
    -----
    Recent SOLRAD data is 1-minute averages.  Prior to 2015-01-01, it was
    3-minute averages.

    References
    ----------
    .. [1] https://gml.noaa.gov/grad/solrad/index.html
    .. [2] https://gml.noaa.gov/aftp/data/radiation/solrad/README_SOLRAD.txt

    Examples
    --------
    >>> # Retrieve one month of irradiance data from the ABQ SOLRAD station
    >>> data, metadata = pvlib.iotools.get_solrad(
    >>>     station='abq', start="2020-01-01", end="2020-01-31")
    """
    # Use pd.to_datetime so that strings (e.g. '2021-01-01') are accepted
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Generate list of filenames
    dates = pd.date_range(start.floor('d'), end, freq='d')
    station = station.lower()
    filenames = [
        f"{station}/{d.year}/{station}{d.strftime('%y')}{d.dayofyear:03}.dat"
        for d in dates
    ]

    dfs = []  # Initialize list of monthly dataframes
    for f in filenames:
        try:
            dfi = read_solrad(url + f)
            dfs.append(dfi)
        except urllib.error.HTTPError:
            warnings.warn(f"The following file was not found: {f}")

    data = pd.concat(dfs, axis='rows')

    meta = {'station': station,
            'filenames': filenames}

    return data, meta
