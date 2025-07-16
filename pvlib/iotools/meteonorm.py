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
    'diffuse_tilted_irradiance': 'poa_diffuse',
    'direct_tilted_irradiance': 'poa_direct',
    'global_tilted_irradiance': 'poa',
    'temperature': 'temp_air',
    'dew_point_temperature': 'temp_dew',
}

time_step_map = {
    '1h': '1_hour',
    'h': '1_hour',
    '15min': '15_minutes',
    '1min': '1_minute',
    'min': '1_minute',
}


def get_meteonorm(latitude, longitude, start, end, api_key, endpoint,
                  parameters='all', *, surface_tilt=0, surface_azimuth=180,
                  time_step='15min', horizon='auto', interval_index=False,
                  map_variables=True, url=URL):
    """
    Retrieve irradiance and weather data from Meteonorm.

    The Meteonorm data options are described in [1]_ and the API is described
    in [2]_. A detailed list of API options can be found in [3]_.

    This function supports historical and forecast data, but not TMY.

    Parameters
    ----------
    latitude: float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    start: datetime like, optional
        First timestamp of the requested period. If a timezone is not
        specified, UTC is assumed. A relative datetime string is also allowed.
    end: datetime like, optional
        Last timestamp of the requested period. If a timezone is not
        specified, UTC is assumed. A relative datetime string is also allowed.
    api_key: str
        Meteonorm API key.
    endpoint : str
        API endpoint, see [3]_. Must be one of:

        * ``'/observation/training'`` - historical data with a 7-day delay
        * ``'/observation/realtime'`` - near-real time (past 7-days)
        * ``'/forecast/basic'`` - forcast with hourly resolution
        * ``'/forecast/precision'`` - forecast with 15-min resolution

    parameters : list, optional
        List of parameters to request or 'all' to get all parameters. The
        default is 'all'.
    surface_tilt: float, optional
        Tilt angle from horizontal plane. The default is 0.
    surface_azimuth: float, optional
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270). The default is 180.
    time_step : {'1min', '15min', '1h'}, optional
        Frequency of the time series. The parameter is ignored when requesting
        forcasting data. The default is '15min'.
    horizon : optional
        Specification of the horizon line. Can be either a 'flat', 'auto', or
        a list of 360 horizon elevation angles. The default is 'auto'.
    interval_index: bool, optional
        Whether the index of the returned data object is of the type
        pd.DatetimeIndex or pd.IntervalIndex. This is an experimental feature
        which may be removed without warning. The default is False.
    map_variables: bool, optional
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. The default is True. See variable
        :const:`VARIABLE_MAP`.
    url: str, optional
        Base URL of the Meteonorm API. The ``endpoint`` parameter is
        appended to the url. The default is
        :const:`pvlib.iotools.meteonorm.URL`.

    Raises
    ------
    requests.HTTPError
        Raises an error when an incorrect request is made.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the start (left) of the
        interval unless ``interval_index`` is set to False.
    meta : dict
        Metadata.

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
    }

    # convert list to string with values separated by commas
    if not isinstance(params['parameters'], (str, type(None))):
        # allow the use of pvlib parameter names
        parameter_dict = {v: k for k, v in VARIABLE_MAP.items()}
        parameters = [parameter_dict.get(p, p) for p in parameters]
        params['parameters'] = ','.join(parameters)

    if horizon not in ['auto', 'flat']:
        params['horizon'] = ','.join(horizon)

    if 'forecast' not in endpoint.lower():
        params['frequency'] = time_step_map.get(time_step, time_step)

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(
        urljoin(url, endpoint), headers=headers, params=params)
    print(response)
    if not response.ok:
        # response.raise_for_status() does not give a useful error message
        raise requests.HTTPError(response.json())

    data, meta = _parse_meteonorm(response, interval_index, map_variables)

    return data, meta


TMY_ENDPOINT = 'climate/tmy'


def get_meteonorm_tmy(latitude, longitude, api_key,
                      parameters='all', *, surface_tilt=0,
                      surface_azimuth=180, time_step='15min', horizon='auto',
                      terrain='open', albedo=0.2, turbidity='auto',
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
    latitude: float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    api_key: str
        Meteonorm API key.
    parameters: list, optional
        List of parameters to request or 'all' to get all parameters. The
        default is 'all'.
    surface_tilt: float, optional
        Tilt angle from horizontal plane. The default is 0.
    surface_azimuth : float, optional
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270). The default is 180.
    time_step: {'1min', '1h'}, optional
        Frequency of the time series. The default is '1h'.
    horizon: optional
        Specification of the hoirzon line. Can be either 'flat' or 'auto', or
        specified as a list of 360 horizon elevation angles. The default is
        'auto'.
    terrain: string, optional
        Local terrain situation. Must be one of: ['open', 'depression',
        'cold_air_lake', 'sea_lake', 'city', 'slope_south',
        'slope_west_east']. The default is 'open'.
    albedo: float, optional
        Ground albedo. Albedo changes due to snow fall are modelled. The
        default is 0.2.
    turbidity: list or 'auto', optional
        List of 12 monthly mean atmospheric Linke turbidity values. The default
        is 'auto'.
    random_seed: int, optional
        Random seed to be used for stochastic processes. Two identical requests
        with the same random seed will yield identical results.
    clear_sky_radiation_model : {'esra', 'solis'}
        Which clearsky model to use. The default is 'esra'.
    data_version : string, optional
        Version of Meteonorm climatological data to be used. The default is
        'latest'.
    future_scenario: string, optional
        Future climate scenario.
    future_year : integer, optional
        Central year for a 20-year reference period in the future.
    interval_index: bool, optional
        Whether the index of the returned data object is of the type
        pd.DatetimeIndex or pd.IntervalIndex. This is an experimental feature
        which may be removed without warning. The default is False.
    map_variables: bool, optional
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`. The default is
        True.
    url: str, optional.
        Base URL of the Meteonorm API. 'climate/tmy'` is
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
        interval unless ``interval_index`` is set to False.
    meta : dict
        Metadata.

    See Also
    --------
    pvlib.iotools.get_meteonorm

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
        'frequency': time_step,
        'parameters': parameters,
        'horizon': horizon,
        'terrain': terrain,
        'turbidity': turbidity,
        'clear_sky_radiation_model': clear_sky_radiation_model,
        'data_version': data_version,
    }

    # convert list to string with values separated by commas
    if not isinstance(params['parameters'], (str, type(None))):
        # allow the use of pvlib parameter names
        parameter_dict = {v: k for k, v in VARIABLE_MAP.items()}
        parameters = [parameter_dict.get(p, p) for p in parameters]
        params['parameters'] = ','.join(parameters)

    if horizon not in ['auto', 'flat']:
        params['horizon'] = ','.join(horizon)

    if turbidity != 'auto':
        params['turbidity'] = ','.join(turbidity)

    if random_seed is not None:
        params['random_seed'] = random_seed

    if future_scenario is not None:
        params['future_scenario'] = future_scenario

    if future_year is not None:
        params['future_year'] = future_year

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(
        urljoin(url, TMY_ENDPOINT), headers=headers, params=params)

    if not response.ok:
        # response.raise_for_status() does not give a useful error message
        raise requests.HTTPError(response.json())

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
            closed='both',
        )
    else:
        data.index = pd.to_datetime(response.json()['start_times'])

    meta = response.json()['meta']

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
        meta['latitude'] = meta.pop('lat')
        meta['longitude'] = meta.pop('lon')

    return data, meta
