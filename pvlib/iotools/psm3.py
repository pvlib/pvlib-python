"""
Get PSM3 TMY
see https://developer.nrel.gov/docs/solar/nsrdb/psm3_data_download/
"""

import requests
import pandas
import io
import csv

URL = "http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv"

# 'relative_humidity', 'total_precipitable_water' are note available
ATTRIBUTES = [
    'air_temperature', 'dew_point', 'dhi', 'dni', 'ghi', 
    'surface_albedo', 'surface_pressure', 'wind_direction', 'wind_speed'
]


def get_psm3(latitude, longitiude, tmy='tmy', interval=60):
    """get psm3"""
    params = {
        'api_key': 'DEMO_KEY',
        'full_name': 'Sample User',
        'email': 'sample@email.com',
        'affiliation': 'Test Organization',
        'reason': 'Example',
        'mailing_list': 'true',
        'wkt': 'POINT(%9.4f %8.4)' % (latitude, longitude),
        'names': tmy,
        'attributes':  ','.join(attributes),
        'leap_day': 'false',
        'utc': 'false',
        'interval': interval
    }

    s = requests.get(url, params=params)
    if s.ok:
        f = io.StringIO(s.content.decode('utf-8'))
        x = csv.reader(f2, delimiter=',', lineterminator='\n')
        y = list[x]
        z = dict(zip(y[0], y[1]))
        w = pd.DataFrame(y[3:], columns=y[2])
        return z, w
    raise requests.HTTPError(s.json())
