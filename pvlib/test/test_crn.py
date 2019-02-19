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
        'longitude', 'latitude', 'temp_air', 'PRECIPITATION', 'ghi', 'ghi_flag',
        'SURFACE_TEMPERATURE', 'ST_TYPE', 'ST_FLAG', 'relative_humidity',
        'relative_humidity_flag', 'SOIL_MOISTURE_5', 'SOIL_TEMPERATURE_5',
        'WETNESS', 'WET_FLAG', 'wind_speed', 'wind_speed_flag']
    index = pd.DatetimeIndex(['2019-01-01 00:05:00+00:00',
                              '2019-01-01 00:10:00+00:00',
                              '2019-01-01 00:15:00+00:00'],
                             dtype='datetime64[ns, UTC]', freq=None)
    values = np.array([
       [53131, 20190101, 5, 20181231, 1705, 3, -111.17, 32.24, 10.4, 0.0,
        10.0, 0, 9.0, 'C', 0, 52.0, 0, nan, nan, 1144, 0, 2.2, 0],
       [53131, 20190101, 10, 20181231, 1710, 3, -111.17, 32.24, 10.5,
        0.0, nan, 0, 9.0, 'C', 0, 52.0, 0, nan, nan, 19, 0, 2.95, 0],
       [53131, 20190101, 15, 20181231, 1715, 3, -111.17, 32.24, nan, 0.0,
        9.0, 0, 8.9, 'C', 0, 52.0, 0, nan, nan, 19, 0, 3.25, 0]])
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
