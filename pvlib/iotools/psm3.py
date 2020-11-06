
"""
Get PSM3 TMY
see https://developer.nrel.gov/docs/solar/nsrdb/psm3_data_download/
"""

import io
import requests
import pandas as pd
from json import JSONDecodeError

NSRDB_API_BASE = "https://developer.nrel.gov"
PSM_URL = NSRDB_API_BASE + "/api/nsrdb/v2/solar/psm3-download.csv"
TMY_URL = NSRDB_API_BASE + "/api/nsrdb/v2/solar/psm3-tmy-download.csv"
PSM5MIN_URL = NSRDB_API_BASE + "/api/nsrdb/v2/solar/psm3-5min-download.csv"

# 'relative_humidity', 'total_precipitable_water' are not available
ATTRIBUTES = (
    'air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo',
    'surface_pressure', 'wind_direction', 'wind_speed')
PVLIB_PYTHON = 'pvlib python'


def get_psm3(latitude, longitude, api_key, email, names='tmy', interval=60,
             attributes=ATTRIBUTES, leap_day=False, full_name=PVLIB_PYTHON,
             affiliation=PVLIB_PYTHON, timeout=30):
    """
    Retrieve NSRDB PSM3 timeseries weather data from the PSM3 API.  The NSRDB
    is described in [1]_ and the PSM3 API is described in [2]_, [3]_, and [4]_.

    Parameters
    ----------
    latitude : float or int
        in decimal degrees, between -90 and 90, north is positive
    longitude : float or int
        in decimal degrees, between -180 and 180, east is positive
    api_key : str
        NREL Developer Network API key
    email : str
        NREL API uses this to automatically communicate messages back
        to the user only if necessary
    names : str, default 'tmy'
        PSM3 API parameter specifing year or TMY variant to download, see notes
        below for options
    interval : int, {60, 5, 15, 30}
        interval size in minutes, must be 5, 15, 30 or 60.  Only used for
        single-year requests (i.e., it is ignored for tmy/tgy/tdy requests).
    attributes : list of str, optional
        meteorological fields to fetch. If not specified, defaults to
        ``pvlib.iotools.psm3.ATTRIBUTES``. See references [2]_, [3]_, and [4]_
        for lists of available fields.
    leap_day : boolean, default False
        include leap day in the results.  Only used for single-year requests
        (i.e., it is ignored for tmy/tgy/tdy requests).
    full_name : str, default 'pvlib python'
        optional
    affiliation : str, default 'pvlib python'
        optional
    timeout : int, default 30
        time in seconds to wait for server response before timeout

    Returns
    -------
    headers : dict
        metadata from NREL PSM3 about the record, see
        :func:`pvlib.iotools.parse_psm3` for fields
    data : pandas.DataFrame
        timeseries data from NREL PSM3

    Raises
    ------
    requests.HTTPError
        if the request response status is not ok, then the ``'errors'`` field
        from the JSON response or any error message in the content will be
        raised as an exception, for example if the `api_key` was rejected or if
        the coordinates were not found in the NSRDB

    Notes
    -----
    The required NREL developer key, `api_key`, is available for free by
    registering at the `NREL Developer Network <https://developer.nrel.gov/>`_.

    .. warning:: The "DEMO_KEY" `api_key` is severely rate limited and may
        result in rejected requests.

    The PSM3 API `names` parameter must be a single value from one of these
    lists:

    +-----------+-------------------------------------------------------------+
    | Category  | Allowed values                                              |
    +===========+=============================================================+
    | Year      | 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, |
    |           | 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, |
    |           | 2018, 2019                                                  |
    +-----------+-------------------------------------------------------------+
    | TMY       | tmy, tmy-2016, tmy-2017, tdy-2017, tgy-2017,                |
    |           | tmy-2018, tdy-2018, tgy-2018, tmy-2019, tdy-2019, tgy-2019  |
    +-----------+-------------------------------------------------------------+

    .. warning:: PSM3 is limited to data found in the NSRDB, please consult the
        references below for locations with available data.  Additionally,
        querying data with < 30-minute resolution uses a different API endpoint
        with fewer available fields (see [4]_).

    See Also
    --------
    pvlib.iotools.read_psm3, pvlib.iotools.parse_psm3

    References
    ----------

    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `Physical Solar Model (PSM) v3
       <https://developer.nrel.gov/docs/solar/nsrdb/psm3_data_download/>`_
    .. [3] `Physical Solar Model (PSM) v3 TMY
       <https://developer.nrel.gov/docs/solar/nsrdb/psm3_tmy_data_download/>`_
    .. [4] `Physical Solar Model (PSM) v3 - Five Minute Temporal Resolution
       <https://developer.nrel.gov/docs/solar/nsrdb/psm3-5min-download/>`_
    """
    # The well know text (WKT) representation of geometry notation is strict.
    # A POINT object is a string with longitude first, then the latitude, with
    # four decimals each, and exactly one space between them.
    longitude = ('%9.4f' % longitude).strip()
    latitude = ('%8.4f' % latitude).strip()
    # TODO: make format_WKT(object_type, *args) in tools.py

    # convert to string to accomodate integer years being passed in
    names = str(names)

    # required query-string parameters for request to PSM3 API
    params = {
        'api_key': api_key,
        'full_name': full_name,
        'email': email,
        'affiliation': affiliation,
        'reason': PVLIB_PYTHON,
        'mailing_list': 'false',
        'wkt': 'POINT(%s %s)' % (longitude, latitude),
        'names': names,
        'attributes':  ','.join(attributes),
        'leap_day': str(leap_day).lower(),
        'utc': 'false',
        'interval': interval
    }
    # request CSV download from NREL PSM3
    if any(prefix in names for prefix in ('tmy', 'tgy', 'tdy')):
        URL = TMY_URL
    elif interval in (5, 15):
        URL = PSM5MIN_URL
    else:
        URL = PSM_URL
    response = requests.get(URL, params=params, timeout=timeout)
    if not response.ok:
        # if the API key is rejected, then the response status will be 403
        # Forbidden, and then the error is in the content and there is no JSON
        try:
            errors = response.json()['errors']
        except JSONDecodeError:
            errors = response.content.decode('utf-8')
        raise requests.HTTPError(errors, response=response)
    # the CSV is in the response content as a UTF-8 bytestring
    # to use pandas we need to create a file buffer from the response
    fbuf = io.StringIO(response.content.decode('utf-8'))
    return parse_psm3(fbuf)


def parse_psm3(fbuf):
    """
    Parse an NSRDB PSM3 weather file (formatted as SAM CSV).  The NSRDB
    is described in [1]_ and the SAM CSV format is described in [2]_.

    Parameters
    ----------
    fbuf: file-like object
        File-like object containing data to read.

    Returns
    -------
    headers : dict
        metadata from NREL PSM3 about the record, see notes for fields
    data : pandas.DataFrame
        timeseries data from NREL PSM3

    Notes
    -----
    The return is a tuple with two items. The first item is a header with
    metadata from NREL PSM3 about the record containing the following fields:

    * Source
    * Location ID
    * City
    * State
    * Country
    * Latitude
    * Longitude
    * Time Zone
    * Elevation
    * Local Time Zone
    * Clearsky DHI Units
    * Clearsky DNI Units
    * Clearsky GHI Units
    * Dew Point Units
    * DHI Units
    * DNI Units
    * GHI Units
    * Solar Zenith Angle Units
    * Temperature Units
    * Pressure Units
    * Relative Humidity Units
    * Precipitable Water Units
    * Wind Direction Units
    * Wind Speed
    * Cloud Type -15
    * Cloud Type 0
    * Cloud Type 1
    * Cloud Type 2
    * Cloud Type 3
    * Cloud Type 4
    * Cloud Type 5
    * Cloud Type 6
    * Cloud Type 7
    * Cloud Type 8
    * Cloud Type 9
    * Cloud Type 10
    * Cloud Type 11
    * Cloud Type 12
    * Fill Flag 0
    * Fill Flag 1
    * Fill Flag 2
    * Fill Flag 3
    * Fill Flag 4
    * Fill Flag 5
    * Surface Albedo Units
    * Version

    The second item is a dataframe with the PSM3 timeseries data.

    Examples
    --------
    >>> # Read a local PSM3 file:
    >>> with open(filename, 'r') as f:  # doctest: +SKIP
    ...     metadata, df = iotools.parse_psm3(f)  # doctest: +SKIP

    See Also
    --------
    pvlib.iotools.read_psm3, pvlib.iotools.get_psm3

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `Standard Time Series Data File Format
       <https://rredc.nrel.gov/solar/old_data/nsrdb/2005-2012/wfcsv.pdf>`_
    """
    # The first 2 lines of the response are headers with metadata
    header_fields = fbuf.readline().split(',')
    header_fields[-1] = header_fields[-1].strip()  # strip trailing newline
    header_values = fbuf.readline().split(',')
    header_values[-1] = header_values[-1].strip()  # strip trailing newline
    header = dict(zip(header_fields, header_values))
    # the response is all strings, so set some header types to numbers
    header['Local Time Zone'] = int(header['Local Time Zone'])
    header['Time Zone'] = int(header['Time Zone'])
    header['Latitude'] = float(header['Latitude'])
    header['Longitude'] = float(header['Longitude'])
    header['Elevation'] = int(header['Elevation'])
    # get the column names so we can set the dtypes
    columns = fbuf.readline().split(',')
    columns[-1] = columns[-1].strip()  # strip trailing newline
    # Since the header has so many columns, excel saves blank cols in the
    # data below the header lines.
    columns = [col for col in columns if col != '']
    dtypes = dict.fromkeys(columns, float)  # all floats except datevec
    dtypes.update(Year=int, Month=int, Day=int, Hour=int, Minute=int)
    dtypes['Cloud Type'] = int
    dtypes['Fill Flag'] = int
    data = pd.read_csv(
        fbuf, header=None, names=columns, usecols=columns, dtype=dtypes,
        delimiter=',', lineterminator='\n')  # skip carriage returns \r
    # the response 1st 5 columns are a date vector, convert to datetime
    dtidx = pd.to_datetime(
        data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    # in USA all timezones are integers
    tz = 'Etc/GMT%+d' % -header['Time Zone']
    data.index = pd.DatetimeIndex(dtidx).tz_localize(tz)

    return header, data


def read_psm3(filename):
    """
    Read an NSRDB PSM3 weather file (formatted as SAM CSV).  The NSRDB
    is described in [1]_ and the SAM CSV format is described in [2]_.

    Parameters
    ----------
    filename: str
        Filename of a file containing data to read.

    Returns
    -------
    headers : dict
        metadata from NREL PSM3 about the record, see
        :func:`pvlib.iotools.parse_psm3` for fields
    data : pandas.DataFrame
        timeseries data from NREL PSM3

    See Also
    --------
    pvlib.iotools.parse_psm3, pvlib.iotools.get_psm3

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `Standard Time Series Data File Format
       <https://rredc.nrel.gov/solar/old_data/nsrdb/2005-2012/wfcsv.pdf>`_
    """
    with open(str(filename), 'r') as fbuf:
        content = parse_psm3(fbuf)
    return content
