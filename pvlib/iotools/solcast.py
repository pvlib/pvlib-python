""" Functions to access data from the Solcast API.
"""

import requests
import pandas as pd
from dataclasses import dataclass


BASE_URL = "https://api.solcast.com.au/data"

@dataclass
class ParameterMap:
    solcast_name: str
    pvlib_name: str
    conversion: callable=lambda x: x

# define the conventions between Solcast and PVLib nomenclature and units
VARIABLE_MAP = [
    ParameterMap("air_temp", "temp_air"),  # air_temp -> temp_air (deg C)
    ParameterMap("surface_pressure", "pressure", lambda x: x*100),  # surface_pressure (hPa) -> pressure (Pa)
    ParameterMap("dewpoint_temp", "temp_dew"),  # dewpoint_temp -> temp_dew (deg C)
    ParameterMap("gti", "poa_global"),  # gti (W/m^2) -> poa_global (W/m^2)
    ParameterMap("wind_speed_10m", "wind_speed"),  # wind_speed_10m (m/s) -> wind_speed (m/s)
    ParameterMap("wind_direction_10m", "wind_direction"),  # wind_direction_10m (deg) -> wind_direction  (deg) (Convention?)
    ParameterMap(
        "azimuth", "solar_azimuth", lambda x: abs(x) if x <= 0 else 360 - x
                 ),  # azimuth -> solar_azimuth (degrees) (different convention)
    ParameterMap("precipitable_water", "precipitable_water", lambda x: x*10),  # precipitable_water (kg/m2) -> precipitable_water (cm)
    ParameterMap("zenith", "solar_zenith")  # zenith -> solar_zenith
]


def get_solcast_tmy(
    latitude, longitude, api_key, map_variables=True, **kwargs
):
    """Get the irradiance and weather for a Typical Meteorological Year (TMY) at a requested location.

    Derived from satellite (clouds and irradiance over non-polar continental areas) and
    numerical weather models (other data). The TMY is calculated with data from 2007 to 2023.

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive
    longitude : float
        in decimal degrees, between -180 and 180, east is positive
    api_key : str
        To access Solcast data you will need an API key: https://toolkit.solcast.com.au/register.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    kwargs:
        Optional parameters passed to the API. See https://docs.solcast.com.au/ for full list of parameters.

    Returns
    -------
    df : pandas.DataFrame
        containing the values for the parameters requested

    Examples
    --------
    get_solcast_tmy(
        latitude=-33.856784,
        longitude=151.215297,
        api_key="your-key"
    )

    you can pass any of the parameters listed in the API docs, like time_zone:

    get_solcast_tmy(
        latitude=-33.856784,
        longitude=151.215297,
        time_zone=10,
        api_key="your-key"
    )

    """

    params = dict(
        latitude=latitude,
        longitude=longitude,
        format="json",
        **kwargs
    )

    return _get_solcast(
        endpoint="tmy/radiation_and_weather",
        params=params,
        api_key=api_key,
        map_variables=map_variables
    )


def get_solcast_historic(
    latitude,
    longitude,
    start,
    api_key,
    end=None,
    duration=None,
    map_variables=True,
    **kwargs
):
    """Get historical irradiance and weather estimated actuals

    for up to 31 days of data at a time for a requested location,
    derived from satellite (clouds and irradiance
    over non-polar continental areas) and numerical weather models (other data).
    Data is available from 2007-01-01T00:00Z up to real time estimated actuals.

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive
    longitude : float
        in decimal degrees, between -180 and 180, east is positive
    start : datetime-like
        First day of the requested period
    end : optional, datetime-like
        Last day of the requested period
    duration : optional, default is None
        Must include one of end_date and duration. ISO_8601 compliant duration for the historic data.
        Must be within 31 days of the start_date.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    api_key : str
        To access Solcast data you will need an API key: https://toolkit.solcast.com.au/register.
    kwargs:
        Optional parameters passed to the GET request

    See https://docs.solcast.com.au/ for full list of parameters.

    Returns
    -------
    df : pandas.DataFrame
        containing the values for the parameters requested

    Examples
    --------
    get_solcast_historic(
        latitude=-33.856784,
        longitude=151.215297,
        start='2007-01-01T00:00Z',
        duration='P1D',
        api_key="your-key"
    )

    you can pass any of the parameters listed in the API docs, for example using the end parameter instead

    get_solcast_historic(
        latitude=-33.856784,
        longitude=151.215297,
        start='2007-01-01T00:00Z',
        end='2007-01-02T00:00Z',
        api_key="your-key"
    )
    """

    params = dict(
        latitude=latitude,
        longitude=longitude,
        start=start,
        end=end,
        duration=duration,
        api_key=api_key,
        format="json",
        **kwargs
    )

    return _get_solcast(
        endpoint="historic/radiation_and_weather",
        params=params,
        api_key=api_key,
        map_variables=map_variables
    )

def get_solcast_forecast(
    latitude, longitude, api_key, map_variables=True, **kwargs
):
    """Get irradiance and weather forecasts from the present time up to 14 days ahead

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive
    longitude : float
        in decimal degrees, between -180 and 180, east is positive
    api_key : str
        To access Solcast data you will need an API key: https://toolkit.solcast.com.au/register.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    kwargs:
        Optional parameters passed to the GET request

    See https://docs.solcast.com.au/ for full list of parameters.

    Returns
    -------
    df : pandas.DataFrame
        containing the values for the parameters requested

    Examples
    --------
    get_solcast_forecast(
        latitude=-33.856784,
        longitude=151.215297,
        api_key="your-key"
    )

    you can pass any of the parameters listed in the API docs, like asking for specific variables
    get_solcast_forecast(
        latitude=-33.856784,
        longitude=151.215297,
        output_parameters='dni,clearsky_dni',
        api_key="your-key"
    )
    """

    params = dict(
        latitude=latitude,
        longitude=longitude,
        format="json",
        **kwargs
    )

    return _get_solcast(
        endpoint="forecast/radiation_and_weather",
        params=params,
        api_key=api_key,
        map_variables=map_variables
    )

def get_solcast_live(
    latitude, longitude, api_key, map_variables=True, **kwargs
):
    """Get irradiance and weather estimated actuals for near real-time and past 7 days

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive
    longitude : float
        in decimal degrees, between -180 and 180, east is positive
    api_key : str
        To access Solcast data you will need an API key: https://toolkit.solcast.com.au/register.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    kwargs:
        Optional parameters passed to the GET request

    See https://docs.solcast.com.au/ for full list of parameters.

    Returns
    -------
    df : pandas.DataFrame
        containing the values for the parameters requested

    Examples
    --------
    get_solcast_live(
        latitude=-33.856784,
        longitude=151.215297,
        api_key="your-key"
    )

    you can pass any of the parameters listed in the API docs, like

    get_solcast_live(
        latitude=-33.856784,
        longitude=151.215297,
        terrain_shading=True,
        api_key="your-key"
    )

    use map_variables=False to avoid converting the data to PVLib's conventions

    get_solcast_live(
        latitude=-33.856784,
        longitude=151.215297,
        map_variables=False,
        api_key="your-key"
    )
    """

    params = dict(
        latitude=latitude,
        longitude=longitude,
        format="json",
        **kwargs
    )

    return _get_solcast(
        endpoint="live/radiation_and_weather",
        params=params,
        api_key=api_key,
        map_variables=map_variables
    )

def solcast2pvlib(df):
    """Formats the data from Solcast to PVLib's conventions.
    """
    # move from period_end to period_middle as per pvlib convention
    df["period_mid"] = pd.to_datetime(df.period_end) - pd.Timedelta(df.period.values[0]) / 2
    df = df.set_index("period_mid").drop(columns=["period_end", "period"])

    # rename and convert variables
    for variable in VARIABLE_MAP:
        if variable.solcast_name in df.columns:
            df.rename(columns={variable.solcast_name: variable.pvlib_name}, inplace=True)
            df[variable.pvlib_name] = df[variable.pvlib_name].apply(variable.conversion)
    return df

def _get_solcast(
        endpoint,
        params,
        api_key,
        map_variables
):
    """retrieves weather, irradiance and power data from the Solcast API

    Parameters
    ----------
    endpoint : str
        one of Solcast API endpoint:
            - live/radiation_and_weather
            - forecast/radiation_and_weather
            - historic/radiation_and_weather
            - tmy/radiation_and_weather
    params : dict
        parameters to be passed to the API
    api_key : str
        To access Solcast data you will need an API key: https://toolkit.solcast.com.au/register.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.

    Returns
    -------
    A pandas.DataFrame with the data if the request is successful, an error message otherwise
    """

    response = requests.get(
        url= '/'.join([BASE_URL, endpoint]),
        params=params,
        headers={"Authorization": f"Bearer {api_key}"}
    )

    if response.status_code == 200:
        j = response.json()
        df = pd.DataFrame.from_dict(j[list(j.keys())[0]])
        if map_variables:
            return solcast2pvlib(df)
        else:
            return df
    else:
        raise Exception(response.json())
