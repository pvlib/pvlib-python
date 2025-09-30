"""Functions for reading and retrieving data from NASA POWER."""

import pandas as pd
import requests
import numpy as np

URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'

DEFAULT_PARAMETERS = [
    'dni', 'dhi', 'ghi', 'temp_air', 'wind_speed'
]

VARIABLE_MAP = {
    'ALLSKY_SFC_SW_DWN': 'ghi',
    'ALLSKY_SFC_SW_DIFF': 'dhi',
    'ALLSKY_SFC_SW_DNI': 'dni',
    'CLRSKY_SFC_SW_DWN': 'ghi_clear',
    'T2M': 'temp_air',
    'WS2M': 'wind_speed_2m',
    'WS10M': 'wind_speed',
}


def get_nasa_power(latitude, longitude, start, end,
                   parameters=DEFAULT_PARAMETERS, *, community='re',
                   elevation=None, wind_height=None, wind_surface=None,
                   map_variables=True, url=URL):
    """
    Retrieve irradiance and weather data from NASA POWER.

    A general description of NASA POWER is given in [1]_ and the API is
    described in [2]_. A detailed list of the available parameters can be
    found in [3]_.

    Parameters
    ----------
    latitude: float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    start: datetime like
        First timestamp of the requested period.
    end: datetime like
        Last timestamp of the requested period.
    parameters: str, list
        List of parameters. The default parameters are mentioned below; for the
        full list see [3]_. Note that the pvlib naming conventions can also be
        used.

        * Global Horizontal Irradiance (GHI) [Wm⁻²]
        * Diffuse Horizontal Irradiance (DHI) [Wm⁻²]
        * Direct Normal Irradiance (DNI) [Wm⁻²]
        * Air temperature at 2 m [C]
        * Wind speed at 10 m [m/s]

    community: str, default 're'
        Can be one of the following depending on which parameters are of
        interest. Note that in many cases this choice
        might affect the units of the parameter.

        * ``'re'``: renewable energy
        * ``'sb'``: sustainable buildings
        * ``'ag'``: agroclimatology

    elevation: float, optional
        The custom site elevation in meters to produce the corrected
        atmospheric pressure adjusted for elevation.
    wind_height: float, optional
        The custom wind height in meters to produce the wind speed adjusted
        for height. Has to be between 10 and 300 m; see [4]_.
    wind_surface: str, optional
        The definable surface type to adjust the wind speed. For a list of the
        surface types see [4]_. If you provide a wind surface alias please
        include a site elevation with the request.
    map_variables: bool, default True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.

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

    References
    ----------
    .. [1] `NASA Prediction Of Worldwide Energy Resources (POWER)
       <https://power.larc.nasa.gov/>`_
    .. [2] `NASA POWER API
       <https://power.larc.nasa.gov/api/pages/>`_
    .. [3] `NASA POWER API parameters
       <https://power.larc.nasa.gov/parameters/>`_
    .. [4] `NASA POWER corrected wind speed parameters
       <https://power.larc.nasa.gov/docs/methodology/meteorology/wind/>`_
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # allow the use of pvlib parameter names
    parameter_dict = {v: k for k, v in VARIABLE_MAP.items()}
    parameters = [parameter_dict.get(p, p) for p in parameters]

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start': start.strftime('%Y%m%d'),
        'end': end.strftime('%Y%m%d'),
        'community': community,
        'parameters': ','.join(parameters),  # make parameters in a string
        'format': 'json',
        'user': None,
        'header': True,
        'time-standard': 'utc',
        'site-elevation': elevation,
        'wind-elevation': wind_height,
        'wind-surface': wind_surface,
    }

    response = requests.get(url, params=params)
    if not response.ok:
        # response.raise_for_status() does not give a useful error message
        raise requests.HTTPError(response.json())

    # Parse the data to dataframe
    data = response.json()
    hourly_data = data['properties']['parameter']
    df = pd.DataFrame(hourly_data)
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H').tz_localize('UTC')

    # Create metadata dictionary
    meta = data['header']
    meta['times'] = data['times']
    meta['parameters'] = data['parameters']

    meta['longitude'] = data['geometry']['coordinates'][0]
    meta['latitude'] = data['geometry']['coordinates'][1]
    meta['altitude'] = data['geometry']['coordinates'][2]

    # Replace NaN values
    df = df.replace(meta['fill_value'], np.nan)

    # Rename according to pvlib convention
    if map_variables:
        df = df.rename(columns=VARIABLE_MAP)

    return df, meta
