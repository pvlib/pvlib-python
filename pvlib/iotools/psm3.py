"""
Get PSM3 TMY
see https://developer.nrel.gov/docs/solar/nsrdb/psm3_data_download/
"""

import io
import csv
import datetime
import requests
import pandas as pd
import pytz

URL = "http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv"

# 'relative_humidity', 'total_precipitable_water' are not available
ATTRIBUTES = [
    'air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 'surface_albedo',
    'surface_pressure', 'wind_direction', 'wind_speed']


def get_psm3(latitude, longitude, names='tmy', interval=60):
    """
    Get PSM3 data

    Parameters
    ----------
    latitude : float or int
        in decimal degrees, between -90 and 90, north is positive
    longitude : float or int
        in decimal degrees, between -180 and 180, east is positive
    names : str
        PSM3 API parameter specifing year or TMY variant to download, see notes
        below for options, default: ``'tmy'``
    interval : int
        interval size in minutes, can be only either 30 or 60, default: 60

    Returns
    -------
    headers : dict
        metadata from NREL PSM3 about the record, see notes for fields
    data : pandas.DataFrame
        timeseries data from NREL PSM3

    Raises
    ------
    requests.HTTPError
        if the request return status is not ok then the ``'errors'`` from the
        JSON response will be returned as an exception

    Notes
    -----
    The PSM3 API `names` parameter must be a single value from the following
    list::

        ['1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
         '2014', '2015', '2016', '2017', 'tmy', 'tmy-2016', 'tmy-2017',
         'tdy-2017', 'tgy-2017']

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
    * Dew Point Units
    * DHI Units
    * DNI Units
    * GHI Units
    * Temperature Units
    * Pressure Units
    * Wind Direction Units
    * Wind Speed
    * Surface Albedo Units
    * Version

    The second item is dataframe with the timeseries data downloaded. The index
    is a :py:class:`pandas.DatetimeIndex` of the 

    See Also
    --------
    pvlib.iotools.read_tmy2, pvlib.iotools.read_tmy3

    References
    ----------

    * `NREL Developer Network - Physical Solar Model (PSM) v3 <https://developer.nrel.gov/docs/solar/nsrdb/psm3_data_download/>`_
    * `NREL National Solar Radiation Database (NSRDB) <https://nsrdb.nrel.gov/>`_

    """
    longitude = ('%9.4f' % longitude).strip()
    latitude = ('%8.4f' % latitude).strip()
    params = {
        'api_key': 'DEMO_KEY',
        'full_name': 'Sample User',
        'email': 'sample@email.com',
        'affiliation': 'Test Organization',
        'reason': 'Example',
        'mailing_list': 'true',
        'wkt': 'POINT(%s %s)' % (longitude, latitude),
        'names': names,
        'attributes':  ','.join(ATTRIBUTES),
        'leap_day': 'false',
        'utc': 'false',
        'interval': interval
    }

    s = requests.get(URL, params=params)
    if s.ok:
        f = io.StringIO(s.content.decode('utf-8'))
        x = csv.reader(f, delimiter=',', lineterminator='\n')
        y = list(x)
        z = dict(zip(y[0], y[1]))
        # in USA all timezones are integers
        z['Local Time Zone'] = int(z['Local Time Zone'])
        z['Time Zone'] = int(z['Time Zone'])
        z['Latitude'] = float(z['Latitude'])
        z['Longitude'] = float(z['Longitude'])
        z['Elevation'] = int(z['Elevation'])
        tz = pytz.timezone('Etc/GMT%+d' % -z['Time Zone'])
        w = pd.DataFrame(y[3:], columns=y[2], dtype=float)
        w.Year = w.Year.astype(int)
        w.Month = w.Month.astype(int)
        w.Day = w.Day.astype(int)
        w.Hour = w.Hour.astype(int)
        w.Minute = w.Minute.astype(int)
        w.index = w.apply(
            lambda dt: tz.localize(datetime.datetime(
                int(dt.Year), int(dt.Month), int(dt.Day),
                int(dt.Hour), int(dt.Minute))), axis=1)
        return z, w
    raise requests.HTTPError(s.json()['errors'])
