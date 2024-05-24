"""Functions to read and retrieve Meteonorm data."""

import requests
import pandas as pd

URL = 'https://mdx.meteotest.ch/api_v1'


@dataclass
class ParameterMap:
    meteonorm_name: str
    pvlib_name: str
    conversion: callable = lambda x: x


# define the conventions between Meteonorm and pvlib nomenclature and units
VARIABLE_MAP = [
    # 'Gh' is GHI without horizon effects
    ParameterMap('Gh hor', 'ghi'),
    ParameterMap('Gh max', 'ghi_clear'),
    ParameterMap('Bn', 'dni'),
    ParameterMap('Bh', 'bhi'),
    # 'Dh' is 'DHI' without horizon effects
    ParameterMap('Dh hor', 'dhi'),
    ParameterMap('Dh min	', 'dhi_clear'),
    # Check units of wind stuff XXX
    ParameterMap('DD', 'wind_direction'),
    ParameterMap('FF', 'wind_speed'),
    ParameterMap('rh', 'relative_humidity'),
    # surface_pressure (hPa) -> pressure (Pa)
    ParameterMap('p', 'pressure', lambda x: x*100),
    ParameterMap('w', 'precipitable_water'),  # cm
    ParameterMap('hs', 'solar_elevation'),
    ParameterMap('az', 'solar_azimuth'),
    ParameterMap('rho', 'albedo'),
    ParameterMap('ta', 'temp_air'),
    ParameterMap('td', 'temp_dew'),
]

# inclination, azimuth, parameters, randomseed (default is 1), local situation, horname (optional, =auto -> topographic horizon)


def get_meteonorm_tmy(latitude, longitude, *, api_key, altitude=None,
                      parameters=None, map_variables=True, URL=URL):
    """Get irradiance and weather for a Typical Meteorological Year (TMY) at a
    requested location.

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive
    longitude : float
        in decimal degrees, between -180 and 180, east is positive
    api_key : str
        To access Meteonorm data you will need an API key [2]_.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    altitude : float, optional
        DESCRIPTION.
    parameters : list, optional
        DESCRIPTION.

    Raises
    ------
    requests
        DESCRIPTION.

    Returns
    -------
    data : pandas.DataFrame
        DESCRIPTION.
    meta : dict
        DESCRIPTION.

    """
    params = {
        'key': api_key,
        'service': 'meteonorm',
        'action': 'calculatestandard',
        'lat': latitude,
        'lon': longitude,
        'format': 'json',
    }

    # Optional variable, defaults to XXX
    if altitude is not None:
        params['altitude'] = altitude

    response = requests.get(URL, params=params)

    if not response.ok:
        raise requests.HTTPError(response.json())

    data = pd.DataFrame(response.json()['payload']['meteonorm']['target']).T
    meta = response.json()['payload']['_metadata']

    # rename and convert variables
    for variable in VARIABLE_MAP:
        if variable.meteonorm_name in data.columns:
            data.rename(
                columns={variable.meteonorm_name: variable.pvlib_name},
                inplace=True
            )
            data[variable.pvlib_name] = data[
                variable.pvlib_name].apply(variable.conversion)

    return data, meta
