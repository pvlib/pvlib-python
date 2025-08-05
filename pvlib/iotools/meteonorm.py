"""Functions for retrieving data from Meteonorm."""

import pandas as pd
import requests
from urllib.parse import urljoin

URL = 'https://api.meteonorm.com/v1/'

VARIABLE_MAP = {
    'global_horizontal_irradiance': 'ghi',
    'diffuse_horizontal_irradiance': 'dhi',
    'direct_normal_irradiance': 'dni',
    'direct_horizontal_irradiance': 'bhi',
    'global_clear_sky_irradiance': 'ghi_clear',
    'diffuse_clear_sky_irradiance': 'dhi_clear',
    'direct_normal_clear_sky_irradiance': 'dni_clear',
    'direct_horizontal_clear_sky_irradiance': 'bhi_clear',
    'diffuse_tilted_irradiance': 'poa_diffuse',
    'direct_tilted_irradiance': 'poa_direct',
    'global_tilted_irradiance': 'poa',
    'temperature': 'temp_air',
    'dew_point_temperature': 'temp_dew',
}

TIME_STEP_MAP = {
    '1h': '1_hour',
    'h': '1_hour',
    '15min': '15_minutes',
    '1min': '1_minute',
    'min': '1_minute',
}


def get_meteonorm_observation(
        latitude, longitude, start, end, api_key, endpoint='training',
        parameters='all', *, surface_tilt=0, surface_azimuth=180,
        time_step='15min', horizon='auto', interval_index=False,
        map_variables=True, url=URL):
    """
    Retrieve historical and near real-time observational data from Meteonorm.

    The Meteonorm data options are described in [1]_ and the API is described
    in [2]_. A detailed list of API options can be found in [3]_.

    This function supports retrieval of observation data, either the
    'training' or the 'realtime' endpoints.

    Parameters
    ----------
    latitude : float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    start : datetime like
        First timestamp of the requested period. If a timezone is not
        specified, UTC is assumed.
    end : datetime like
        Last timestamp of the requested period. If a timezone is not
        specified, UTC is assumed.
    api_key : str
        Meteonorm API key.
    endpoint : str, default : training
        API endpoint, see [3]_. Must be one of:

        * ``'training'`` - historical data with a 7-day delay
        * ``'realtime'`` - near-real time (past 7-days)

    parameters : list or 'all', default : 'all'
        List of parameters to request or `'all'` to get all parameters.
    surface_tilt : float, default : 0
        Tilt angle from horizontal plane.
    surface_azimuth : float, default : 180
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270).
    time_step : {'1min', '15min', '1h'}, default : '15min'
        Frequency of the time series.
    horizon : str or list, default : 'auto'
        Specification of the horizon line. Can be either 'flat', 'auto', or
        a list of 360 integer horizon elevation angles.
    interval_index : bool, default : False
        Index is pd.DatetimeIndex when False, and pd.IntervalIndex when True.
        This is an experimental feature which may be removed without warning.
    map_variables : bool, default : True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional
        Base URL of the Meteonorm API. The default is
        :const:`pvlib.iotools.meteonorm.URL`.

    Raises
    ------
    requests.HTTPError
        Raises an error when an incorrect request is made.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the start (left) of the
        interval unless ``interval_index`` is set to True.
    meta : dict
        Metadata.

    Examples
    --------
    >>> # Retrieve historical time series data
    >>> df, meta = pvlib.iotools.get_meteonorm_observatrion(  # doctest: +SKIP
    ...     latitude=50, longitude=10,  # doctest: +SKIP
    ...     start='2023-01-01', end='2025-01-01',  # doctest: +SKIP
    ...     api_key='redacted',  # doctest: +SKIP
    ...     endpoint='training')  # doctest: +SKIP

    See Also
    --------
    pvlib.iotools.get_meteonorm_tmy

    References
    ----------
    .. [1] `Meteonorm
       <https://meteonorm.com/>`_
    .. [2] `Meteonorm API
       <https://docs.meteonorm.com/docs/getting-started>`_
    .. [3] `Meteonorm API reference
       <https://docs.meteonorm.com/api>`_
    """
    endpoint_base = 'observation/'

    data, meta = _get_meteonorm(
        latitude, longitude, start, end, api_key,
        endpoint_base, endpoint,
        parameters, surface_tilt, surface_azimuth,
        time_step, horizon, interval_index,
        map_variables, url)
    return data, meta


def get_meteonorm_forecast(
        latitude, longitude, start, end, api_key, endpoint='precision',
        parameters='all', *, surface_tilt=0, surface_azimuth=180,
        time_step='15min', horizon='auto', interval_index=False,
        map_variables=True, url=URL):
    """
    Retrieve historical and near real-time observational data from Meteonorm.

    The Meteonorm data options are described in [1]_ and the API is described
    in [2]_. A detailed list of API options can be found in [3]_.

    This function supports retrieval of forecasting data, either the
    'training' or the 'basic' endpoints.

    Parameters
    ----------
    latitude : float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    start : datetime like
        First timestamp of the requested period. If a timezone is not
        specified, UTC is assumed.
    end : datetime like
        Last timestamp of the requested period. If a timezone is not
        specified, UTC is assumed.
    api_key : str
        Meteonorm API key.
    endpoint : str, default : precision
        API endpoint, see [3]_. Must be one of:

        * ``'precision'`` - forecast with 1-min, 15-min, or hourly
          resolution
        * ``'basic'`` - forecast with hourly resolution

    parameters : list or 'all', default : 'all'
        List of parameters to request or `'all'` to get all parameters.
    surface_tilt : float, default : 0
        Tilt angle from horizontal plane.
    surface_azimuth : float, default : 180
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270).
    time_step : {'1min', '15min', '1h'}, default : '15min'
        Frequency of the time series. The endpoint ``'basic'`` only
        supports ``time_step='1h'``.
    horizon : str or list, default : 'auto'
        Specification of the horizon line. Can be either 'flat', 'auto', or
        a list of 360 integer horizon elevation angles.
    interval_index : bool, default : False
        Index is pd.DatetimeIndex when False, and pd.IntervalIndex when True.
        This is an experimental feature which may be removed without warning.
    map_variables : bool, default : True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional
        Base URL of the Meteonorm API. The default is
        :const:`pvlib.iotools.meteonorm.URL`.

    Raises
    ------
    requests.HTTPError
        Raises an error when an incorrect request is made.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the start (left) of the
        interval unless ``interval_index`` is set to True.
    meta : dict
        Metadata.

    See Also
    --------
    pvlib.iotools.get_meteonorm_observation,
    pvlib.iotools.get_meteonorm_tmy

    References
    ----------
    .. [1] `Meteonorm
       <https://meteonorm.com/>`_
    .. [2] `Meteonorm API
       <https://docs.meteonorm.com/docs/getting-started>`_
    .. [3] `Meteonorm API reference
       <https://docs.meteonorm.com/api>`_
    """
    endpoint_base = 'forecast/'

    data, meta = _get_meteonorm(
        latitude, longitude, start, end, api_key,
        endpoint_base, endpoint,
        parameters, surface_tilt, surface_azimuth,
        time_step, horizon, interval_index,
        map_variables, url)
    return data, meta


def _get_meteonorm(
        latitude, longitude, start, end, api_key,
        endpoint_base, endpoint,
        parameters, surface_tilt, surface_azimuth,
        time_step, horizon, interval_index,
        map_variables, url):

    # Relative date strings are not yet supported
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize('UTC') if start.tzinfo is None else start
    end = end.tz_localize('UTC') if end.tzinfo is None else end

    params = {
        'lat': latitude,
        'lon': longitude,
        'start': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'parameters': parameters,
        'surface_tilt': surface_tilt,
        'surface_azimuth': surface_azimuth,
        'horizon': horizon,
        'response_format': 'json',
    }

    # Allow specifying single parameters as string
    if isinstance(parameters, str):
        parameters = [parameters]

    # allow the use of pvlib parameter names
    parameter_dict = {v: k for k, v in VARIABLE_MAP.items()}
    parameters = [parameter_dict.get(p, p) for p in parameters]
    # convert list to string with values separated by commas
    params['parameters'] = ','.join(parameters)

    if not isinstance(horizon, str):
        params['horizon'] = ','.join(map(str, horizon))

    if 'basic' not in endpoint:
        params['frequency'] = TIME_STEP_MAP.get(time_step, time_step)
    else:
        if time_step not in ['1h', '1_hour']:
            raise ValueError("The 'forecast/basic' api endpoint only "
                             "supports ``time_step='1h'``.")

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(
        urljoin(url, endpoint_base + endpoint.lstrip('/')),
        headers=headers, params=params)

    if not response.ok:
        # response.raise_for_status() does not give a useful error message
        raise requests.HTTPError("Meteonorm API returned an error: "
                                 + response.json()['error']['message'])

    data, meta = _parse_meteonorm(response, interval_index, map_variables)

    return data, meta


TMY_ENDPOINT = 'climate/tmy'


def get_meteonorm_tmy(latitude, longitude, api_key,
                      parameters='all', *, surface_tilt=0,
                      surface_azimuth=180, time_step='1h', horizon='auto',
                      terrain_situation='open', albedo=None, turbidity='auto',
                      random_seed=None, clear_sky_radiation_model='esra',
                      data_version='latest', future_scenario=None,
                      future_year=None, interval_index=False,
                      map_variables=True, url=URL):
    """
    Retrieve TMY irradiance and weather data from Meteonorm.

    The Meteonorm data options are described in [1]_ and the API is described
    in [2]_. A detailed list of API options can be found in [3]_.

    Parameters
    ----------
    latitude : float
        In decimal degrees, north is positive (ISO 19115).
    longitude : float
        In decimal degrees, east is positive (ISO 19115).
    api_key : str
        Meteonorm API key.
    parameters : list or 'all', default : 'all'
        List of parameters to request or `'all'` to get all parameters.
    surface_tilt : float, default : 0
        Tilt angle from horizontal plane.
    surface_azimuth : float, default : 180
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270).
    time_step : {'1min', '1h'}, default : '1h'
        Frequency of the time series.
    horizon : str, optional
        Specification of the horizon line. Can be either 'flat' or 'auto', or
        specified as a list of 360 integer horizon elevation angles.
        'auto'.
    terrain_situation : str, default : 'open'
        Local terrain situation. Must be one of: ['open', 'depression',
        'cold_air_lake', 'sea_lake', 'city', 'slope_south',
        'slope_west_east'].
    albedo : float, optional
        Constant ground albedo. If no value is specified a baseline albedo of
        0.2 is used and albedo changes due to snow fall are modeled. If a value
        is specified, then snow fall is not modeled.
    turbidity : list or 'auto', optional
        List of 12 monthly mean atmospheric Linke turbidity values. The default
        is 'auto'.
    random_seed : int, optional
        Random seed to be used for stochastic processes. Two identical requests
        with the same random seed will yield identical results.
    clear_sky_radiation_model : str, default : 'esra'
        Which clearsky model to use. Must be either `'esra'` or `'solis'`.
    data_version : str, default : 'latest'
        Version of Meteonorm climatological data to be used.
    future_scenario : str, optional
        Future climate scenario.
    future_year : int, optional
        Central year for a 20-year reference period in the future.
    interval_index : bool, default : False
        Index is pd.DatetimeIndex when False, and pd.IntervalIndex when True.
        This is an experimental feature which may be removed without warning.
    map_variables : bool, default : True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    url : str, optional.
        Base URL of the Meteonorm API. `'climate/tmy'` is
        appended to the URL. The default is:
        :const:`pvlib.iotools.meteonorm.URL`.

    Raises
    ------
    requests.HTTPError
        Raises an error when an incorrect request is made.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the start (left) of the
        interval unless ``interval_index`` is set to True.
    meta : dict
        Metadata.

    See Also
    --------
    pvlib.iotools.get_meteonorm_observation,
    pvlib.iotools.get_meteonorm_forecast

    References
    ----------
    .. [1] `Meteonorm
       <https://meteonorm.com/>`_
    .. [2] `Meteonorm API
       <https://docs.meteonorm.com/docs/getting-started>`_
    .. [3] `Meteonorm API reference
       <https://docs.meteonorm.com/api>`_
    """
    params = {
        'lat': latitude,
        'lon': longitude,
        'surface_tilt': surface_tilt,
        'surface_azimuth': surface_azimuth,
        'frequency': TIME_STEP_MAP.get(time_step, time_step),
        'parameters': parameters,
        'horizon': horizon,
        'situation': terrain_situation,
        'turbidity': turbidity,
        'clear_sky_radiation_model': clear_sky_radiation_model,
        'data_version': data_version,
        'random_seed': random_seed,
        'future_scenario': future_scenario,
        'future_year': future_year,
        'response_format': 'json',
    }

    # Allow specifying single parameters as string
    if isinstance(parameters, str):
        parameters = [parameters]

    # allow the use of pvlib parameter names
    parameter_dict = {v: k for k, v in VARIABLE_MAP.items()}
    parameters = [parameter_dict.get(p, p) for p in parameters]
    # convert list to string with values separated by commas
    params['parameters'] = ','.join(parameters)

    if not isinstance(horizon, str):
        params['horizon'] = ','.join(map(str, horizon))

    if not isinstance(turbidity, str):
        params['turbidity'] = ','.join(map(str, turbidity))

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(
        urljoin(url, TMY_ENDPOINT.lstrip('/')), headers=headers, params=params)

    if not response.ok:
        # response.raise_for_status() does not give a useful error message
        raise requests.HTTPError("Meteonorm API returned an error: "
                                 + response.json()['error']['message'])

    data, meta = _parse_meteonorm(response, interval_index, map_variables)

    return data, meta


def _parse_meteonorm(response, interval_index, map_variables):
    data_json = response.json()['values']
    # identify empty columns
    empty_columns = [k for k, v in data_json.items() if v is None]
    # remove empty columns
    _ = [data_json.pop(k) for k in empty_columns]

    data = pd.DataFrame(data_json)

    # xxx: experimental feature - see parameter description
    if interval_index:
        data.index = pd.IntervalIndex.from_arrays(
            left=pd.to_datetime(response.json()['start_times']),
            right=pd.to_datetime(response.json()['end_times']),
            closed='left',
        )
    else:
        data.index = pd.to_datetime(response.json()['start_times'])

    meta = response.json()['meta']

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
        meta['latitude'] = meta.pop('lat')
        meta['longitude'] = meta.pop('lon')

    return data, meta
