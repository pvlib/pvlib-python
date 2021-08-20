"""
tests for :mod:`pvlib.iotools.merra2`
"""

import pandas as pd
import numpy as np
import datetime as dt
import pytest
import os
from pvlib.iotools import read_merra2, get_merra2
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal,
                        requires_merra2_credentials, requires_xarray)


@pytest.fixture(scope="module")
def merra2_credentials():
    """Supplies pvlib-python's EarthData login credentials.
    Users should obtain their own credentials as described in the `get_merra2`
    documentation."""
    return (os.environ["MERRA2_USERNAME"], os.environ["MERRA2_PASSWORD"])


@pytest.fixture
def expected_index():
    index = pd.date_range('2020-1-1-00:30', periods=24*2, freq='1h', tz='UTC')
    index.name = 'time'
    return index


@requires_xarray
def test_read_merra2(expected_index):
    filenames = [DATA_DIR / 'MERRA2_400.tavg1_2d_rad_Nx.20200101.SUB.nc',
                 DATA_DIR / 'MERRA2_400.tavg1_2d_rad_Nx.20200102.SUB.nc']

    data, meta = read_merra2(filenames, map_variables=False)
    assert_index_equal(data.index, expected_index)
    assert meta['lat'] == {'name': 'lat', 'long_name': 'latitude',
                          'units': 'degrees_north'}
    assert np.isclose(data.loc['2020-01-01 12:30:00+00:00', 'SWGDN'], 130.4375)


@requires_xarray
def test_read_merra2_dataset(expected_index):
    filenames = [DATA_DIR / 'MERRA2_400.tavg1_2d_rad_Nx.20200101.SUB.nc',
                 DATA_DIR / 'MERRA2_400.tavg1_2d_rad_Nx.20200102.SUB.nc']

    data, meta = read_merra2(filenames, output_format='dataset',
                             map_variables=False)
    import xarray as xr
    assert isinstance(data, xr.Dataset)
    assert meta['lat'] == {'name': 'lat', 'long_name': 'latitude',
                          'units': 'degrees_north'}
    assert np.all([v in ['time', 'lon', 'lat', 'ALBEDO', 'EMIS', 'SWGDN',
                        'SWGDNCLR', 'SWTDN'] for v in list(data.variables)])


@requires_xarray
def test_read_merra2_map_variables():
    filename = DATA_DIR / 'MERRA2_400.tavg1_2d_rad_Nx.20200101.SUB.nc'
    data, meta = read_merra2(filename, map_variables=True)
    assert meta['ghi'] == {
        'name': 'ghi', 'long_name': 'surface_incoming_shortwave_flux',
        'units': 'W m-2'}


@requires_xarray
@requires_merra2_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2(merra2_credentials):
    username, password = merra2_credentials()
    data, meta = get_merra2(
        55, 15, dt.datetime(2020,1,1), dt.datetime(2020,1,2),
        dataset='M2T1NXRAD', variables=['TAUHGH', 'SWGNT'],
        username=username, password=password, map_variables=True)
    assert True
