"""
Functions for reading and retrieving data from NSRDB PSM4. See:
https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-aggregated-v4-0-0-download/
https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-tmy-v4-0-0-download/
https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-conus-v4-0-0-download/
https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-full-disc-v4-0-0-download/
"""

import io
from urllib.parse import urljoin
import requests
import pandas as pd
from json import JSONDecodeError
from pvlib import tools

NSRDB_API_BASE = "https://developer.nrel.gov/api/nsrdb/v2/solar/"
PSM4_AGG_ENDPOINT = "nsrdb-GOES-aggregated-v4-0-0-download.csv"
PSM4_TMY_ENDPOINT = "nsrdb-GOES-tmy-v4-0-0-download.csv"
PSM4_CON_ENDPOINT = "nsrdb-GOES-conus-v4-0-0-download.csv"
PSM4_FUL_ENDPOINT = "nsrdb-GOES-full-disc-v4-0-0-download.csv"
PSM4_AGG_URL = urljoin(NSRDB_API_BASE, PSM4_AGG_ENDPOINT)
PSM4_TMY_URL = urljoin(NSRDB_API_BASE, PSM4_TMY_ENDPOINT)
PSM4_CON_URL = urljoin(NSRDB_API_BASE, PSM4_CON_ENDPOINT)
PSM4_FUL_URL = urljoin(NSRDB_API_BASE, PSM4_FUL_ENDPOINT)

PARAMETERS = (
    'air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo',
    'surface_pressure', 'wind_direction', 'wind_speed')
PVLIB_PYTHON = 'pvlib python'

# Dictionary mapping PSM4 response names to pvlib names
VARIABLE_MAP = {
    'GHI': 'ghi',
    'DHI': 'dhi',
    'DNI': 'dni',
    'Clearsky GHI': 'ghi_clear',
    'Clearsky DHI': 'dhi_clear',
    'Clearsky DNI': 'dni_clear',
    'Solar Zenith Angle': 'solar_zenith',
    'Temperature': 'temp_air',
    'Dew Point': 'temp_dew',
    'Relative Humidity': 'relative_humidity',
    'Pressure': 'pressure',
    'Wind Speed': 'wind_speed',
    'Wind Direction': 'wind_direction',
    'Surface Albedo': 'albedo',
    'Precipitable Water': 'precipitable_water',
    'AOD': 'aod',
}

# Dictionary mapping pvlib names to PSM4 request names
# Note, PSM4 uses different names for the same variables in the
# response and the request
REQUEST_VARIABLE_MAP = {
    'ghi': 'ghi',
    'dhi': 'dhi',
    'dni': 'dni',
    'ghi_clear': 'clearsky_ghi',
    'dhi_clear': 'clearsky_dhi',
    'dni_clear': 'clearsky_dni',
    'solar_zenith': 'solar_zenith_angle',
    'temp_air': 'air_temperature',
    'temp_dew': 'dew_point',
    'relative_humidity': 'relative_humidity',
    'pressure': 'surface_pressure',
    'wind_speed': 'wind_speed',
    'wind_direction': 'wind_direction',
    'albedo': 'surface_albedo',
    'precipitable_water': 'total_precipitable_water',
    'aod': 'aod',
}


def get_nsrdb_psm4_aggregated(latitude, longitude, api_key, email,
                              year, time_step=60,
                              parameters=PARAMETERS, leap_day=True,
                              full_name=PVLIB_PYTHON,
                              affiliation=PVLIB_PYTHON,
                              utc=False, map_variables=True, url=None,
                              timeout=30):
    """
    Retrieve NSRDB PSM4 timeseries weather data from the PSM4 NSRDB GOES
    Aggregated v4 API.

    The NSRDB is described in [1]_ and the PSM4 NSRDB GOES Aggregated v4 API is
    described in [2]_.

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
    year : int or str
        PSM4 API parameter specifing year (e.g. ``2023``) to download. The
        allowed values update periodically, so consult the NSRDB reference
        below for the current set of options. Called ``names`` in NSRDB API.
    time_step : int, {60, 30}
        time step in minutes, must be 60 or 30 for PSM4 Aggregated. Called
        ``interval`` in NSRDB API.
    parameters : list of str, optional
        meteorological fields to fetch. If not specified, defaults to
        ``pvlib.iotools.psm4.PARAMETERS``. See reference [2]_ for a list of
        available fields. Alternatively, pvlib names may also be used (e.g.
        'ghi' rather than 'GHI'); see :const:`REQUEST_VARIABLE_MAP`. To
        retrieve all available fields, set ``parameters=[]``.
    leap_day : bool, default : True
        include leap day in the results
    full_name : str, default 'pvlib python'
        optional
    affiliation : str, default 'pvlib python'
        optional
    utc: bool, default : False
        retrieve data with timestamps converted to UTC. False returns
        timestamps in local standard time of the selected location
    map_variables : bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional
        Full API endpoint URL. If not specified, the PSM4 GOES Aggregated v4
        URL is used.
    timeout : int, default 30
        time in seconds to wait for server response before timeout

    Returns
    -------
    data : pandas.DataFrame
        timeseries data from NREL PSM4
    metadata : dict
        metadata from NREL PSM4 about the record, see
        :func:`pvlib.iotools.read_nsrdb_psm4` for fields

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

    .. warning:: PSM4 is limited to data found in the NSRDB, please consult
        the references below for locations with available data.

    See Also
    --------
    pvlib.iotools.get_nsrdb_psm4_tmy, pvlib.iotools.get_nsrdb_psm4_conus,
    pvlib.iotools.get_nsrdb_psm4_full_disc, pvlib.iotools.read_nsrdb_psm4

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `NSRDB GOES Aggregated V4.0.0
       <https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-aggregated-v4-0-0-download/>`_
    """
    # The well know text (WKT) representation of geometry notation is strict.
    # A POINT object is a string with longitude first, then the latitude, with
    # four decimals each, and exactly one space between them.
    longitude = ('%9.4f' % longitude).strip()
    latitude = ('%8.4f' % latitude).strip()
    # TODO: make format_WKT(object_type, *args) in tools.py

    # convert pvlib names in parameters to PSM4 convention
    parameters = [REQUEST_VARIABLE_MAP.get(a, a) for a in parameters]

    # required query-string parameters for request to PSM4 API
    params = {
        'api_key': api_key,
        'full_name': full_name,
        'email': email,
        'affiliation': affiliation,
        'reason': PVLIB_PYTHON,
        'mailing_list': 'false',
        'wkt': 'POINT(%s %s)' % (longitude, latitude),
        'names': year,
        'attributes':  ','.join(parameters),
        'leap_day': str(leap_day).lower(),
        'utc': str(utc).lower(),
        'interval': time_step
    }
    # request CSV download from NREL PSM4
    if url is None:
        url = PSM4_AGG_URL

    response = requests.get(url, params=params, timeout=timeout)
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
    return read_nsrdb_psm4(fbuf, map_variables)


def get_nsrdb_psm4_tmy(latitude, longitude, api_key, email, year='tmy',
                       time_step=60, parameters=PARAMETERS, leap_day=False,
                       full_name=PVLIB_PYTHON, affiliation=PVLIB_PYTHON,
                       utc=False, map_variables=True, url=None, timeout=30):
    """
    Retrieve NSRDB PSM4 timeseries weather data from the PSM4 NSRDB GOES
    TMY v4 API.

    The NSRDB is described in [1]_ and the PSM4 NSRDB GOES TMY v4 API is
    described in [2]_.

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
    year : str, default 'tmy'
        PSM4 API parameter specifing TMY variant to download (e.g. ``'tmy'``
        or ``'tgy-2022'``).  The allowed values update periodically, so
        consult the NSRDB references below for the current set of options.
        Called ``names`` in NSRDB API.
    time_step : int, {60}
        time step in minutes. Must be 60 for typical year requests. Called
        ``interval`` in NSRDB API.
    parameters : list of str, optional
        meteorological fields to fetch. If not specified, defaults to
        ``pvlib.iotools.psm4.PARAMETERS``. See reference [2]_ for a list of
        available fields. Alternatively, pvlib names may also be used (e.g.
        'ghi' rather than 'GHI'); see :const:`REQUEST_VARIABLE_MAP`. To
        retrieve all available fields, set ``parameters=[]``.
    leap_day : bool, default : False
        Include leap day in the results. Ignored for tmy/tgy/tdy requests.
    full_name : str, default 'pvlib python'
        optional
    affiliation : str, default 'pvlib python'
        optional
    utc: bool, default : False
        retrieve data with timestamps converted to UTC. False returns
        timestamps in local standard time of the selected location
    map_variables : bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional
        Full API endpoint URL. If not specified, the PSM4 GOES TMY v4 URL is
        used.
    timeout : int, default 30
        time in seconds to wait for server response before timeout

    Returns
    -------
    data : pandas.DataFrame
        timeseries data from NREL PSM4
    metadata : dict
        metadata from NREL PSM4 about the record, see
        :func:`pvlib.iotools.read_nsrdb_psm4` for fields

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

    .. warning:: PSM4 is limited to data found in the NSRDB, please consult
        the references below for locations with available data.

    See Also
    --------
    pvlib.iotools.get_nsrdb_psm4_aggregated,
    pvlib.iotools.get_nsrdb_psm4_conus, pvlib.iotools.get_nsrdb_psm4_full_disc,
    pvlib.iotools.read_nsrdb_psm4

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `NSRDB GOES Tmy V4.0.0
       <https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-tmy-v4-0-0-download/>`_
    """
    # The well know text (WKT) representation of geometry notation is strict.
    # A POINT object is a string with longitude first, then the latitude, with
    # four decimals each, and exactly one space between them.
    longitude = ('%9.4f' % longitude).strip()
    latitude = ('%8.4f' % latitude).strip()
    # TODO: make format_WKT(object_type, *args) in tools.py

    # convert pvlib names in parameters to PSM4 convention
    parameters = [REQUEST_VARIABLE_MAP.get(a, a) for a in parameters]

    # required query-string parameters for request to PSM4 API
    params = {
        'api_key': api_key,
        'full_name': full_name,
        'email': email,
        'affiliation': affiliation,
        'reason': PVLIB_PYTHON,
        'mailing_list': 'false',
        'wkt': 'POINT(%s %s)' % (longitude, latitude),
        'names': year,
        'attributes':  ','.join(parameters),
        'leap_day': str(leap_day).lower(),
        'utc': str(utc).lower(),
        'interval': time_step
    }
    # request CSV download from NREL PSM4
    if url is None:
        url = PSM4_TMY_URL

    response = requests.get(url, params=params, timeout=timeout)
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
    return read_nsrdb_psm4(fbuf, map_variables)


def get_nsrdb_psm4_conus(latitude, longitude, api_key, email, year,
                         time_step=60, parameters=PARAMETERS, leap_day=True,
                         full_name=PVLIB_PYTHON, affiliation=PVLIB_PYTHON,
                         utc=False, map_variables=True, url=None, timeout=30):
    """
    Retrieve NSRDB PSM4 timeseries weather data from the PSM4 NSRDB GOES CONUS
    v4 API.

    The NSRDB is described in [1]_ and the PSM4 NSRDB GOES CONUS v4 API is
    described in [2]_.

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
    year : int or str
        PSM4 API parameter specifing year (e.g. ``2023``) to download. The
        allowed values update periodically, so consult the NSRDB reference
        below for the current set of options. Called ``names`` in NSRDB API.
    time_step : int, {60, 5, 15, 30}
        time step in minutes. Called ``interval`` in NSRDB API.
    parameters : list of str, optional
        meteorological fields to fetch. If not specified, defaults to
        ``pvlib.iotools.psm4.PARAMETERS``. See reference [2]_ for a list of
        available fields. Alternatively, pvlib names may also be used (e.g.
        'ghi' rather than 'GHI'); see :const:`REQUEST_VARIABLE_MAP`. To
        retrieve all available fields, set ``parameters=[]``.
    leap_day : bool, default : True
        include leap day in the results
    full_name : str, default 'pvlib python'
        optional
    affiliation : str, default 'pvlib python'
        optional
    utc: bool, default : False
        retrieve data with timestamps converted to UTC. False returns
        timestamps in local standard time of the selected location
    map_variables : bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional
        Full API endpoint URL. If not specified, the PSM4 GOES CONUS v4 URL is
        used.
    timeout : int, default 30
        time in seconds to wait for server response before timeout

    Returns
    -------
    data : pandas.DataFrame
        timeseries data from NREL PSM4
    metadata : dict
        metadata from NREL PSM4 about the record, see
        :func:`pvlib.iotools.read_nsrdb_psm4` for fields

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

    .. warning:: PSM4 is limited to data found in the NSRDB, please consult
        the references below for locations with available data.

    See Also
    --------
    pvlib.iotools.get_nsrdb_psm4_aggregated,
    pvlib.iotools.get_nsrdb_psm4_tmy, pvlib.iotools.get_nsrdb_psm4_full_disc,
    pvlib.iotools.read_nsrdb_psm4

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `NSRDB GOES Conus V4.0.0
       <https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-conus-v4-0-0-download/>`_
    """
    # The well know text (WKT) representation of geometry notation is strict.
    # A POINT object is a string with longitude first, then the latitude, with
    # four decimals each, and exactly one space between them.
    longitude = ('%9.4f' % longitude).strip()
    latitude = ('%8.4f' % latitude).strip()
    # TODO: make format_WKT(object_type, *args) in tools.py

    # convert pvlib names in parameters to PSM4 convention
    parameters = [REQUEST_VARIABLE_MAP.get(a, a) for a in parameters]

    # required query-string parameters for request to PSM4 API
    params = {
        'api_key': api_key,
        'full_name': full_name,
        'email': email,
        'affiliation': affiliation,
        'reason': PVLIB_PYTHON,
        'mailing_list': 'false',
        'wkt': 'POINT(%s %s)' % (longitude, latitude),
        'names': year,
        'attributes':  ','.join(parameters),
        'leap_day': str(leap_day).lower(),
        'utc': str(utc).lower(),
        'interval': time_step
    }
    # request CSV download from NREL PSM4
    if url is None:
        url = PSM4_CON_URL

    response = requests.get(url, params=params, timeout=timeout)
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
    return read_nsrdb_psm4(fbuf, map_variables)


def get_nsrdb_psm4_full_disc(latitude, longitude, api_key, email,
                             year, time_step=60,
                             parameters=PARAMETERS, leap_day=True,
                             full_name=PVLIB_PYTHON,
                             affiliation=PVLIB_PYTHON, utc=False,
                             map_variables=True, url=None, timeout=30):
    """
    Retrieve NSRDB PSM4 timeseries weather data from the PSM4 NSRDB GOES Full
    Disc v4 API.

    The NSRDB is described in [1]_ and the PSM4 NSRDB GOES Full Disc v4 API is
    described in [2]_.

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
    year : int or str
        PSM4 API parameter specifing year (e.g. ``2023``) to download. The
        allowed values update periodically, so consult the NSRDB reference
        below for the current set of options. Called ``names`` in NSRDB API.
    time_step : int, {60, 10, 30}
        time step in minutes, must be 10, 30 or 60. Called ``interval`` in
        NSRDB API.
    parameters : list of str, optional
        meteorological fields to fetch. If not specified, defaults to
        ``pvlib.iotools.psm4.PARAMETERS``. See reference [2]_ for a list of
        available fields. Alternatively, pvlib names may also be used (e.g.
        'ghi' rather than 'GHI'); see :const:`REQUEST_VARIABLE_MAP`. To
        retrieve all available fields, set ``parameters=[]``.
    leap_day : bool, default : True
        include leap day in the results
    full_name : str, default 'pvlib python'
        optional
    affiliation : str, default 'pvlib python'
        optional
    utc: bool, default : False
        retrieve data with timestamps converted to UTC. False returns
        timestamps in local standard time of the selected location
    map_variables : bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional
        Full API endpoint URL. If not specified, the PSM4 GOES Full Disc v4
        URL is used.
    timeout : int, default 30
        time in seconds to wait for server response before timeout

    Returns
    -------
    data : pandas.DataFrame
        timeseries data from NREL PSM4
    metadata : dict
        metadata from NREL PSM4 about the record, see
        :func:`pvlib.iotools.read_nsrdb_psm4` for fields

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

    .. warning:: PSM4 is limited to data found in the NSRDB, please consult
        the references below for locations with available data.

    See Also
    --------
    pvlib.iotools.get_nsrdb_psm4_aggregated,
    pvlib.iotools.get_nsrdb_psm4_tmy, pvlib.iotools.get_nsrdb_psm4_conus,
    pvlib.iotools.read_nsrdb_psm4

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `NSRDB GOES Full Disc V4.0.0
       <https://developer.nrel.gov/docs/solar/nsrdb/nsrdb-GOES-full-disc-v4-0-0-download/>`_
    """
    # The well know text (WKT) representation of geometry notation is strict.
    # A POINT object is a string with longitude first, then the latitude, with
    # four decimals each, and exactly one space between them.
    longitude = ('%9.4f' % longitude).strip()
    latitude = ('%8.4f' % latitude).strip()
    # TODO: make format_WKT(object_type, *args) in tools.py

    # convert pvlib names in parameters to PSM4 convention
    parameters = [REQUEST_VARIABLE_MAP.get(a, a) for a in parameters]

    # required query-string parameters for request to PSM4 API
    params = {
        'api_key': api_key,
        'full_name': full_name,
        'email': email,
        'affiliation': affiliation,
        'reason': PVLIB_PYTHON,
        'mailing_list': 'false',
        'wkt': 'POINT(%s %s)' % (longitude, latitude),
        'names': year,
        'attributes':  ','.join(parameters),
        'leap_day': str(leap_day).lower(),
        'utc': str(utc).lower(),
        'interval': time_step
    }
    # request CSV download from NREL PSM4
    if url is None:
        url = PSM4_FUL_URL

    response = requests.get(url, params=params, timeout=timeout)
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
    return read_nsrdb_psm4(fbuf, map_variables)


def read_nsrdb_psm4(filename, map_variables=True):
    """
    Read an NSRDB PSM4 weather file (formatted as SAM CSV).

    The NSRDB is described in [1]_ and the SAM CSV format is described in [2]_.

    Parameters
    ----------
    filename: str, path-like, or buffer
        Filename or in-memory buffer of a file containing data to read.
    map_variables: bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.

    Returns
    -------
    data : pandas.DataFrame
        timeseries data from NREL PSM4
    metadata : dict
        metadata from NREL PSM4 about the record, see notes for fields

    Notes
    -----
    The return is a tuple with two items. The first item is a dataframe with
    the PSM4 timeseries data.

    The second item is a dictionary with metadata from NREL PSM4 about the
    record containing the following fields:

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
    * Wind Speed Units
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

    Examples
    --------
    >>> # Read a local PSM4 file:
    >>> df, metadata = iotools.read_nsrdb_psm4("data.csv")  # doctest: +SKIP

    >>> # Read a file object or an in-memory buffer:
    >>> with open(filename, 'r') as f:  # doctest: +SKIP
    ...     df, metadata = iotools.read_nsrdb_psm4(f)  # doctest: +SKIP

    See Also
    --------
    pvlib.iotools.get_nsrdb_psm4_aggregated
    pvlib.iotools.get_nsrdb_psm4_tmy
    pvlib.iotools.get_nsrdb_psm4_conus
    pvlib.iotools.get_nsrdb_psm4_full_disc
    pvlib.iotools.read_psm3

    References
    ----------
    .. [1] `NREL National Solar Radiation Database (NSRDB)
       <https://nsrdb.nrel.gov/>`_
    .. [2] `Standard Time Series Data File Format
       <https://web.archive.org/web/20170207203107/https://sam.nrel.gov/sites/default/files/content/documents/pdf/wfcsv.pdf>`_
    """
    with tools._file_context_manager(filename) as fbuf:
        # The first 2 lines of the response are headers with metadata
        metadata_fields = fbuf.readline().split(',')
        metadata_values = fbuf.readline().split(',')
        # get the column names so we can set the dtypes
        columns = fbuf.readline().split(',')
        columns[-1] = columns[-1].strip()  # strip trailing newline
        # Since the header has so many columns, excel saves blank cols in the
        # data below the header lines.
        columns = [col for col in columns if col != '']
        dtypes = dict.fromkeys(columns, float)
        dtypes.update({'Year': int, 'Month': int, 'Day': int, 'Hour': int,
                       'Minute': int, 'Cloud Type': int, 'Fill Flag': int})

        data = pd.read_csv(
            fbuf, header=None, names=columns, usecols=columns, dtype=dtypes,
            delimiter=',', lineterminator='\n')  # skip carriage returns \r

    metadata_fields[-1] = metadata_fields[-1].strip()  # trailing newline
    metadata_values[-1] = metadata_values[-1].strip()  # trailing newline
    metadata = dict(zip(metadata_fields, metadata_values))
    # the response is all strings, so set some metadata types to numbers
    metadata['Local Time Zone'] = int(metadata['Local Time Zone'])
    metadata['Time Zone'] = int(metadata['Time Zone'])
    metadata['Latitude'] = float(metadata['Latitude'])
    metadata['Longitude'] = float(metadata['Longitude'])
    metadata['Elevation'] = int(metadata['Elevation'])

    # the response 1st 5 columns are a date vector, convert to datetime
    dtidx = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    # in USA all timezones are integers
    tz = 'Etc/GMT%+d' % -metadata['Time Zone']
    data.index = pd.DatetimeIndex(dtidx).tz_localize(tz)

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
        metadata['latitude'] = metadata.pop('Latitude')
        metadata['longitude'] = metadata.pop('Longitude')
        metadata['altitude'] = metadata.pop('Elevation')

    return data, metadata
