import inspect
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
from numpy import dtype, nan

from pvlib.iotools import crn


test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
testfile = os.path.join(test_dir,
                        '../data/CRNS0101-05-2019-AZ_Tucson_11_W.txt')


def test_read_crn():
    columns = [
        'WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN',
        'longitude', 'latitude', 'temp_air', 'PRECIPITATION', 'ghi',
        'ghi_flag',
        'SURFACE_TEMPERATURE', 'ST_TYPE', 'ST_FLAG', 'relative_humidity',
        'relative_humidity_flag', 'SOIL_MOISTURE_5', 'SOIL_TEMPERATURE_5',
        'WETNESS', 'WET_FLAG', 'wind_speed', 'wind_speed_flag']
    index = pd.DatetimeIndex(['2019-01-01 16:10:00',
                              '2019-01-01 16:15:00',
                              '2019-01-01 16:20:00',
                              '2019-01-01 16:25:00'],
                             freq=None).tz_localize('UTC')
    values = np.array([
        [53131, 20190101, 1610, 20190101, 910, 3, -111.17, 32.24, nan,
         0.0, 296.0, 0, 4.4, 'C', 0, 90.0, 0, nan, nan, 24, 0, 0.78, 0],
        [53131, 20190101, 1615, 20190101, 915, 3, -111.17, 32.24, 3.3,
         0.0, 183.0, 0, 4.0, 'C', 0, 87.0, 0, nan, nan, 1182, 0, 0.36, 0],
        [53131, 20190101, 1620, 20190101, 920, 3, -111.17, 32.24, 3.5,
         0.0, 340.0, 0, 4.3, 'C', 0, 83.0, 0, nan, nan, 1183, 0, 0.53, 0],
        [53131, 20190101, 1625, 20190101, 925, 3, -111.17, 32.24, 4.0,
         0.0, 393.0, 0, 4.8, 'C', 0, 81.0, 0, nan, nan, 1223, 0, 0.64, 0]])
    dtypes = [
        dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64'),
        dtype('int64'), dtype('int64'), dtype('float64'), dtype('float64'),
        dtype('float64'), dtype('float64'), dtype('float64'),
        dtype('int64'), dtype('float64'), dtype('O'), dtype('int64'),
        dtype('float64'), dtype('int64'), dtype('float64'),
        dtype('float64'), dtype('int64'), dtype('int64'), dtype('float64'),
        dtype('int64')]
    expected = pd.DataFrame(values, columns=columns, index=index)
    for (col, _dtype) in zip(expected.columns, dtypes):
        expected[col] = expected[col].astype(_dtype)
    out = crn.read_crn(testfile)
    assert_frame_equal(out, expected)
