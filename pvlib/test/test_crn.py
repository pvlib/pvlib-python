import inspect
import os

import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
from numpy import dtype

from pvlib.iotools import crn


test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
testfile = os.path.join(test_dir,
                        '../data/CRNS0101-05-2019-AZ_Tucson_11_W.txt')


def test_read_crn():
    columns = [
        'longitude', 'latitude', 'temp_air', 'ghi', 'ghi_flag',
        'relative_humidity', 'relative_humidity_flag', 'wind_speed',
        'wind_speed_flag']
    index = pd.DatetimeIndex(['2019-01-01 00:05:00+00:00',
                              '2019-01-01 00:10:00+00:00',
                              '2019-01-01 00:15:00+00:00'],
                             dtype='datetime64[ns, UTC]', freq=None)
    values = np.array([
        [-111.17, 32.24, 10.4, 10., 0, 52., 0, 2.2, 0],
        [-111.17, 32.24, 10.5, np.nan, 0, 52, 0, 2.95, 0],
        [-111.17, 32.24, np.nan, 9., 0, 52., 0, 3.25, 0]])
    dtypes = [
        dtype('float64'), dtype('float64'), dtype('float64'),
        dtype('float64'), dtype('int64'), dtype('float64'), dtype('int64'),
        dtype('float64'), dtype('int64')]
    expected = pd.DataFrame(values, columns=columns, index=index)
    for (col, _dtype) in zip(expected.columns, dtypes):
        expected[col] = expected[col].astype(_dtype)
    out = crn.read_crn(testfile)
    assert_frame_equal(out, expected)
