import pandas as pd
from conftest import assert_frame_equal
import numpy as np
from numpy import nan

import pytest

from pvlib.iotools import solrad
from conftest import DATA_DIR


testfile = DATA_DIR / 'abq19056.dat'
testfile_mad = DATA_DIR / 'msn19056.dat'


columns = [
    'year', 'julian_day', 'month', 'day', 'hour', 'minute', 'decimal_time',
    'solar_zenith', 'ghi', 'ghi_flag', 'dni', 'dni_flag', 'dhi', 'dhi_flag',
    'uvb', 'uvb_flag', 'uvb_temp', 'uvb_temp_flag', 'std_dw_psp', 'std_direct',
    'std_diffuse', 'std_uvb']
index = pd.DatetimeIndex(['2019-02-25 00:00:00',
                          '2019-02-25 00:01:00',
                          '2019-02-25 00:02:00',
                          '2019-02-25 00:03:00'],
                         freq=None).tz_localize('UTC')
values = np.array([
    [2.019e+03, 5.600e+01, 2.000e+00, 2.500e+01, 0.000e+00, 0.000e+00,
        0.000e+00, 7.930e+01, 1.045e+02, 0.000e+00, 6.050e+01, 0.000e+00,
        9.780e+01, 0.000e+00, 5.900e+00, 0.000e+00, 4.360e+01, 0.000e+00,
        3.820e-01, 2.280e+00, 4.310e-01, 6.000e-02],
    [2.019e+03, 5.600e+01, 2.000e+00, 2.500e+01, 0.000e+00, 1.000e+00,
        1.700e-02, 7.949e+01, 1.026e+02, 0.000e+00, 5.970e+01, 0.000e+00,
        9.620e+01, 0.000e+00, 5.700e+00, 0.000e+00, 4.360e+01, 0.000e+00,
        7.640e-01, 1.800e+00, 4.310e-01, 6.000e-02],
    [2.019e+03, 5.600e+01, 2.000e+00, 2.500e+01, 0.000e+00, 2.000e+00,
        3.300e-02, 7.968e+01, 1.021e+02, 0.000e+00, 6.580e+01, 0.000e+00,
        9.480e+01, 0.000e+00, 5.500e+00, 0.000e+00, 4.360e+01, 0.000e+00,
        3.820e-01, 4.079e+00, 3.230e-01, 6.000e-02],
    [2.019e+03, 5.600e+01, 2.000e+00, 2.500e+01, 0.000e+00, 3.000e+00,
        5.000e-02, 7.987e+01, 1.026e+02, 0.000e+00, 7.630e+01, 0.000e+00,
        nan, 0.000e+00, 5.300e+00, 0.000e+00, 4.360e+01, 0.000e+00,
        5.090e-01, 1.920e+00, 2.150e-01, 5.000e-02]])
dtypes = [
    'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64',
    'float64', 'float64', 'int64', 'float64', 'int64', 'float64', 'int64',
    'float64', 'int64', 'float64', 'int64', 'float64', 'float64',
    'float64', 'float64']

columns_mad = [
    'year', 'julian_day', 'month', 'day', 'hour', 'minute', 'decimal_time',
    'solar_zenith', 'ghi', 'ghi_flag', 'dni', 'dni_flag', 'dhi', 'dhi_flag',
    'uvb', 'uvb_flag', 'uvb_temp', 'uvb_temp_flag', 'dpir', 'dpir_flag',
    'dpirc', 'dpirc_flag', 'dpird', 'dpird_flag', 'std_dw_psp',
    'std_direct', 'std_diffuse', 'std_uvb', 'std_dpir', 'std_dpirc',
    'std_dpird']
values_mad = np.array([
    [ 2.019e+03,  5.600e+01,  2.000e+00,  2.500e+01,  0.000e+00,
      0.000e+00,  0.000e+00,  9.428e+01, -2.300e+00,  0.000e+00,
      0.000e+00,  0.000e+00,  4.000e-01,  0.000e+00,        nan,
      1.000e+00,        nan,  1.000e+00,  1.872e+02,  0.000e+00,
      2.656e+02,  0.000e+00,  2.653e+02,  0.000e+00,  0.000e+00,
      0.000e+00,  0.000e+00,        nan,  2.000e-03,  2.600e+01,
      2.700e+01],
    [ 2.019e+03,  5.600e+01,  2.000e+00,  2.500e+01,  0.000e+00,
      1.000e+00,  1.700e-02,  9.446e+01, -2.300e+00,  0.000e+00,
      0.000e+00,  0.000e+00,  1.000e-01,  0.000e+00,        nan,
      1.000e+00,        nan,  1.000e+00,  1.882e+02,  0.000e+00,
      2.656e+02,  0.000e+00,  2.653e+02,  0.000e+00,  1.330e-01,
      1.280e-01,  2.230e-01,        nan,  1.000e-03,  2.600e+01,
      7.200e+01],
    [ 2.019e+03,  5.600e+01,  2.000e+00,  2.500e+01,  0.000e+00,
      2.000e+00,  3.300e-02,  9.464e+01, -2.700e+00,  0.000e+00,
     -2.000e-01,  0.000e+00,  0.000e+00,  0.000e+00,        nan,
      1.000e+00,        nan,  1.000e+00,  1.876e+02,  0.000e+00,
      2.656e+02,  0.000e+00,  2.653e+02,  0.000e+00,  0.000e+00,
      2.570e-01,  0.000e+00,        nan,  1.000e-03,  2.400e+01,
      4.200e+01],
    [ 2.019e+03,  5.600e+01,  2.000e+00,  2.500e+01,  0.000e+00,
      3.000e+00,  5.000e-02,  9.482e+01, -2.500e+00,  0.000e+00,
      4.000e-01,  0.000e+00,  0.000e+00,  0.000e+00,        nan,
      1.000e+00,        nan,  1.000e+00,  1.873e+02,  0.000e+00,
      2.656e+02,  0.000e+00,  2.653e+02,  0.000e+00,  2.660e-01,
      3.850e-01,  0.000e+00,        nan,  1.000e-03,  2.600e+01,
      4.800e+01]])
dtypes_mad = [
    'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64',
    'float64', 'int64', 'float64', 'int64', 'float64', 'int64', 'float64',
    'int64', 'float64', 'int64', 'float64', 'int64', 'float64', 'int64',
    'float64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64',
    'float64', 'float64']


@pytest.mark.parametrize('testfile,index,columns,values,dtypes', [
    (testfile, index, columns, values, dtypes),
    (testfile_mad, index, columns_mad, values_mad, dtypes_mad)
])
def test_read_solrad(testfile, index, columns, values, dtypes):
    expected = pd.DataFrame(values, columns=columns, index=index)
    for (col, _dtype) in zip(expected.columns, dtypes):
        expected[col] = expected[col].astype(_dtype)
    out = solrad.read_solrad(testfile)
    assert_frame_equal(out, expected)
