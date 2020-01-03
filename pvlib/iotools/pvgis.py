"""
Get, read, and parse data from `PVGIS <https://ec.europa.eu/jrc/en/pvgis>`_.

For more information, see the following links:
* `Interactive Tools <https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html>`_
* `Data downloads <https://ec.europa.eu/jrc/en/PVGIS/downloads/data>`_
* `User manual docs <https://ec.europa.eu/jrc/en/PVGIS/docs/usermanual>`_

More detailed information about the API for TMY and hourly radiation are here:
* `TMY <https://ec.europa.eu/jrc/en/PVGIS/tools/tmy>`_
* `hourly radiation <https://ec.europa.eu/jrc/en/PVGIS/tools/hourly-radiation>`_
"""
import io
import os
import tempfile
import requests
import pandas as pd
from pvlib.iotools import read_epw

URL = 'https://re.jrc.ec.europa.eu/api/'


def get_pvgis_tmy(lat, lon, outputformat='json', usehorizon=True,
                  userhorizon=None, startyear=None, endyear=None):
    """
    Get TMY data from PVGIS. For more information see documentation for PVGIS
    `TMY tools <https://ec.europa.eu/jrc/en/PVGIS/tools/tmy>`_

    Parmaeters
    ----------
    lat : float
        Latitude in degrees north
    lon : float
        Longitude in dgrees east
    outputformat : string [default 'json']
        Must be in ``['csv', 'basic', 'epw', 'json']``. See link for more info.
    usehorizon : bool [default True]
        include effects of horizon
    userhorizon : list of float [default None]
        elevation of horizon in degrees at eight cardinal directions clockwise
        fron north, _EG_: north, north-east, east, south-east, etc.
    startyear : integer [default None]
        first year to calculate TMY
    endyear : integer [default None]
        last year to calculate TMY, must be at least 10 years from first year
    
    Returns
    -------
    data : pandas.DataFrame
        the weather data
    months_selected : list
        TMY year for each month, ``None`` for basic and EPW
    inputs : dict
        the inputs, ``None`` for basic
    meta : list or dict
        meta data, ``None`` for basic, for EPW contains the temporary filename
    """
    # use requests to format the query string by passing params dictionary
    params = {'lat':lat, 'lon': lon, 'outputformat': outputformat}
    # pvgis only likes 0 for False, and 1 for True, not strings, also the
    # default for usehorizon is already 1 (ie: True), so only set if False
    if not usehorizon:
        params['usehorizon'] = 0
    res = requests.get(URL + 'tmy', params=params)
    # PVGIS returns really well formatted error messages in JSON for HTTP/1.1
    # 400 BAD REQUEST so try to return that if possible, otherwise raise the
    # HTTP/1.1 error caught by requests
    if not res.ok:
        try:
            err_msg = res.json()
        except:
            res.raise_for_status()
        else:
            raise requests.HTTPError(err_msg)
    if outputformat == 'json':
        src = res.json()
        return _parse_pvgis_tmy_json(src)
    elif outputformat == 'csv':
        src = io.BytesIO(res.content)
        try:
            data = _parse_pvgis_tmy_csv(src)
        finally:
            src.close()
    elif outputformat == 'basic':
        src = io.BytesIO(res.content)
        try:
            data = _parse_pvgis_tmy_basic(src)
        finally:
            src.close()
    elif outputformat == 'epw':
        with tempfile.TemporaryFile(delete=False) as f:
            f.write(res.content)
        data, inputs = read_epw(f.name)
        data = (data, None, inputs, f.name)
    else:
        raise ValueError('unknown output format %s' % outputformat)
    return data

def _parse_pvgis_tmy_json(src):
    inputs = src['inputs']
    meta = src['meta']
    months_selected = src['outputs']['months_selected']
    data = pd.DataFrame(src['outputs']['tmy_hourly'])
    data.index = pd.to_datetime(
        data['time(UTC)'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time(UTC)', axis=1)
    return data, months_selected, inputs, meta


def _parse_pvgis_tmy_csv(src):
    # the first 3 rows are latitude, longitude, elevation
    inputs = {}
    # 'Latitude (decimal degrees): 45.000\r\n'
    inputs['latitude'] = float(src.readline().split(b':')[1])
    # 'Longitude (decimal degrees): 8.000\r\n'
    inputs['longitude'] = float(src.readline().split(b':')[1])
    # Elevation (m): 1389.0\r\n
    inputs['elevation'] = float(src.readline().split(b':')[1])
    # then there's a 13 row comma separated table with two columns: month, year
    # which contains the year used for that month in the
    src.readline()  # get "month,year\r\n"
    months_selected = []
    for month in range(12):
        months_selected.append((month+1, int(src.readline().split(b',')[1])))
    # then there's the TMY (typical meteorological year) data
    # first there's a header row:
    #    time(UTC),T2m,RH,G(h),Gb(n),Gd(h),IR(h),WS10m,WD10m,SP
    headers = [h.decode('utf-8').strip() for h in src.readline().split(b',')]
    data = pd.DataFrame(
        [src.readline().split(b',') for _ in range(8760)], columns=headers)
    dtidx = data['time(UTC)'].apply(lambda dt: dt.decode('utf-8'))
    dtidx = pd.to_datetime(dtidx, format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time(UTC)', axis=1)
    data = pd.DataFrame(data, dtype=float)
    data.index = dtidx
    # finally there's some meta data
    meta = [line.decode('utf-8').strip() for line in src.readlines()]
    return data, months_selected, inputs, meta


def _parse_pvgis_tmy_basic(src):
    data = pd.read_csv(src)
    data.index = pd.to_datetime(
        data['time(UTC)'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time(UTC)', axis=1)
    return data, None, None, None