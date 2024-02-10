"""Functions to retrieve and parse irradiance data from Solargis."""

import pandas as pd
import requests
import io

URL = 'https://solargis.info/ws/rest/datadelivery/request?'

VARIABLE_MAP = {
    'GHI': 'ghi',
    'GHI_C': 'ghi_clear',  # this is stated in documentation
    'GHIc': 'ghi_clear',  # this is used in practice
    'DNI': 'dni',
    'DNI_C': 'dni_clear',
    'DNIc': 'dni_clear',
    'DIF': 'dhi',
    'GTI': 'poa_global',
    'GTI_C': 'poa_global_clear',
    'GTIc': 'poa_global_clear',
    'SE': 'solar_elevation',
    'SA': 'solar_azimuth',
    'TEMP': 'temp_air',
    'TD': 'temp_dew',
    'AP': 'pressure',
    'RH': 'relative_humidity',
    'WS': 'wind_speed',
    'WD': 'wind_direction',
    'INC': 'aoi',  # angle of incidence of direct irradiance
    'PWAT': 'precipitable_water',  # [kg/m2]
    }

METADATA_FIELDS = [
    'issued', 'site name', 'latitude', 'longitude', 'elevation',
    'summarization type', 'summarization period'
]


def get_solargis(latitude, longitude, start, end, variables, summarization,
                 api_key, timestamp_type='center', tz='GMT+00',
                 terrain_shading=True, url=URL, map_variables=True,
                 timeout=30):
    """
    Retrieve irradiance time series data from Solargis.

    The Solargis [1]_ API is described in [2]_.

    Parameters
    ----------
    latitude: float
        In decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude: float
        In decimal degrees, between -180 and 180, east is positive (ISO 19115)
    start : datetime-like
        Start date of time series. Assumes UTC.
    end : datetime-like
        End date of time series. Assumes UTC.
    variables : list
        List of variables to request, see [2]_ for options.
    summarization : str, {'MIN_5', 'MIN_10', 'MIN_15', 'MIN_30', 'HOURLY', 'DAILY', 'MONTHLY', 'YEARLY'}
        DESCRIPTION.
    api_key : str
        API key.
    timestamp_type : {'start', 'center', 'end'}, default: 'center'
        How to label time intervals in the return data.
    tz : str, default : 'GMT+00'
        Timezone of `start` and `end` in the format "GMT+hh" or "GMT-hh".
    terrain_shading : boolean, default: True
        Whether to account for horizon shading.
    url : str, default : :const:`pvlib.iotools.solargis.URL`
        Base url of Solargis API.
    map_variables : boolean, default: True
        When true, renames columns of the Dataframe to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    timeout : int or float, default: 30
        Time in seconds to wait for server response before timeout

    Returns
    -------
    data : DataFrame
        DataFrame containing time series data.
    meta : dict
        Dictionary containing metadata.

    Notes
    -----
    Each XMl request is limited to retrieving 31 days of data.

    The variable units depends on the time frequency, e.g., the unit for
    sub-hourly irradiance data is W/m^2, for hourly data it is Wh/m^2, and for
    daily data it is kWh/m^2.

    Raises
    ------
    HTTPError
        If an incorrect request is made.

    References
    ----------
    .. [1] `Solargis <https://solargis.com>`_
    .. [2] `Solargis API User Guide
       <https://solargis.atlassian.net/wiki/spaces/public/pages/7602367/Solargis+API+User+Guide>`_
    """  # noqa: E501
    # Use pd.to_datetime so that strings (e.g. '2021-01-01') are accepted
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    headers = {'Content-Type': 'application/xml'}

    # Solargis recommends creating a unique site_id for each location request.
    # The site_id does not impact the data retrieval and is used for debugging.
    site_id = f"latitude_{latitude}_longitude_{longitude}"

    request_xml = f'''<ws:dataDeliveryRequest
    dateFrom="{start.strftime('%Y-%m-%d')}"
    dateTo="{end.strftime('%Y-%m-%d')}"
    xmlns="http://geomodel.eu/schema/data/request"
    xmlns:ws="http://geomodel.eu/schema/ws/data"
    xmlns:geo="http://geomodel.eu/schema/common/geo"
    xmlns:pv="http://geomodel.eu/schema/common/pv"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <site id="{site_id}" name="" lat="{latitude}" lng="{longitude}">
    </site>
    <processing key="{' '.join(variables)}" summarization="{summarization}"
    terrainShading="{str(terrain_shading).lower()}">
    <timestampType>{timestamp_type.upper()}</timestampType>
    <timeZone>{tz}</timeZone>
    </processing>
    </ws:dataDeliveryRequest>'''
    response = requests.post(url + "key=" + api_key, headers=headers,
                             data=request_xml.encode('utf8'), timeout=timeout)

    if response.ok is False:
        raise requests.HTTPError(response.json())

    # Parse metadata
    header = pd.read_xml(io.StringIO(response.text))
    meta_lines = header['metadata'].iloc[0].split('#')
    meta_lines = [line.strip() for line in meta_lines]
    meta = {}
    for line in meta_lines:
        if ':' in line:
            key = line.split(':')[0].lower()
            if key in METADATA_FIELDS:
                meta[key] = ':'.join(line.split(':')[1:])
    meta['latitude'] = float(meta['latitude'])
    meta['longitude'] = float(meta['longitude'])
    meta['altitude'] = float(meta.pop('elevation').replace('m a.s.l.', ''))

    # Parse data
    data = pd.read_xml(io.StringIO(response.text), xpath='.//doc:row',
                       namespaces={'doc': 'http://geomodel.eu/schema/ws/data'})
    data.index = pd.to_datetime(data['dateTime'])
    data = data['values'].str.split(' ', expand=True)
    data = data.astype(float)
    data.columns = header['columns'].iloc[0].split()

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)

    data = data.replace(-9, pd.NA)

    return data, meta
