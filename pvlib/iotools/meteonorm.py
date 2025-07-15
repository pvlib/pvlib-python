"""Functions for reading and retrieving data from Meteonorm."""

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
                  parameters="all", *, surface_tilt=0, surface_azimuth=180,
                  time_step='15min', horizon='auto', interval_index=False,
                  map_variables=True, url=URL):
    """
    Retrieve irradiance and weather data from Meteonorm.

    The Meteonorm data options are described in [1]_ and the API is described
    in [2]_. A detailed list of API options can be found in [3]_.

    This function supports the end points 'realtime' for data for the past 7
    days, 'training' for historical data with a delay of 7 days. The function
    does not support TMY climate data.

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
        API end point, see [3]_. Must be one of:

            * '/observation/training'
            * '/observation/realtime'
            * '/forecast/basic'
            * '/forecast/precision'

    parameters : list, optional
        List of parameters to request or "all" to get all parameters. The
        default is "all".
    surface_tilt: float, default: 0
        Tilt angle from horizontal plane.
    surface_azimuth: float, default: 180
        Orientation (azimuth angle) of the (fixed) plane. Clockwise from north
        (north=0, east=90, south=180, west=270).
    time_step : {'1min', '15min', '1h'}, optional
        ime step of the time series. The default is '15min'. Ignored if
        requesting forecast data.
    horizon : optional
        Specification of the hoirzon line. Can be either 'flat' or 'auto', or
        specified as a list of 360 horizon elevation angles. The default is
        'auto'.
    interval_index: bool, optional
        Whether the index of the returned data object is of the type
        pd.DatetimeIndex or pd.IntervalIndex. This is an experimental feature
        which may be removed without warning. The default is False.
    map_variables: bool, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. The default is True. See variable
        :const:`VARIABLE_MAP`.
    url: str, default: :const:`pvlib.iotools.meteonorm.URL`
        Base url of the Meteonorm API. The ``endpoint`` parameter is
        appended to the url.

    Raises
    ------
    requests.HTTPError
        Raises an error when an incorrect request is made.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the start (left) of the
        interval.
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
        'surface_tilt': surface_tilt,
        'surface_azimuth': surface_azimuth,
        'horizon': horizon,
        'parameters': parameters,
    }

    if 'forecast' not in endpoint.lower():
        params['frequency'] = time_step_map.get(time_step, time_step)

    # convert list to string with values separated by commas
    if not isinstance(params['parameters'], (str, type(None))):
        # allow the use of pvlib parameter names
        parameter_dict = {v: k for k, v in VARIABLE_MAP.items()}
        parameters = [parameter_dict.get(p, p) for p in parameters]
        params['parameters'] = ','.join(parameters)

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(urljoin(url, endpoint), headers=headers, params=params)

    if not response.ok:
        # response.raise_for_status() does not give a useful error message
        raise requests.HTTPError(response.json())

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
