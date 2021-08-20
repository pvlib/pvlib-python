"""
tests for :mod:`pvlib.iotools.era5`
"""

import pandas as pd
import numpy as np
import pytest
import os
from pvlib.iotools import read_era5, get_era5
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal,
                        requires_cds_credentials, requires_xarray)


@pytest.fixture(scope="module")
def cds_api_key():
    """Supplies pvlib-python's CDS API key.

    Users should obtain their own credentials as described in the `get_era5`
    documentation."""
    return os.environ["CDSAPI_KEY"]


@pytest.fixture
def expected_index():
    index = pd.date_range('2020-1-1', freq='1h', periods=8832, tz='UTC')
    index.name = 'time'
    return index


@pytest.fixture
def expected_columns():
    return ['t2m', 'u10', 'v10', 'sp', 'msdwswrf', 'msdwswrfcs', 'msdrswrf',
            'msdrswrfcs']


@pytest.fixture
def expected_columns_mapped():
    return ['temp_air', 'u10', 'v10', 'pressure', 'ghi', 'ghi_clear', 'bhi',
            'bhi_clear']


@requires_xarray
def test_read_era5(expected_index, expected_columns):
    data, meta = read_era5(DATA_DIR / 'era5_testfile.nc', map_variables=False)
    assert (expected_columns == data.columns).all()
    assert_index_equal(data.index, expected_index[:8784])
    # Test meta
    assert meta['msdwswrf'] == {
        'name': 'msdwswrf',
        'long_name': 'Mean surface downward short-wave radiation flux',
        'units': 'W m**-2'}
    assert 'dims' in meta.keys()
    # Test conversion of K to C
    assert meta['t2m']['units'] == 'C'
    assert np.isclose(data['t2m'].iloc[0], 2.8150635)  # temperature in deg C


@requires_xarray
def test_read_era5_variable_mapped(expected_index, expected_columns_mapped):
    data, meta = read_era5(DATA_DIR / 'era5_testfile.nc')
    assert (expected_columns_mapped == data.columns).all()
    assert_index_equal(data.index, expected_index[:8784])
    assert data.notna().all().all()
    assert meta['temp_air'] == {
        'name': 'temp_air', 'long_name': '2 metre temperature', 'units': 'C'}


@requires_xarray
def test_read_era5_output_format():
    import xarray as xr
    data, meta = read_era5(DATA_DIR / 'era5_testfile.nc',
                           output_format='dataset')
    assert isinstance(data, xr.Dataset)


@requires_xarray
def test_read_era5_multiple_files(expected_index):
    filenames = \
        [DATA_DIR / f for f in ['era5_testfile.nc', 'era5_testfile_1day.nc']]
    data, meta = read_era5(filenames)
    assert_index_equal(data.index, expected_index)


@requires_xarray
@requires_cds_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5(cds_api_key, expected_index):
    data, meta = get_era5(
        latitude=55.7,
        longitude=12.5,
        start=pd.Timestamp(2020, 1, 1),
        end=pd.Timestamp(2020, 1, 1),
        variables=['mean_surface_downward_short_wave_radiation_flux_clear_sky',
                   '2m_temperature'],
        api_key=cds_api_key,
        save_path='era5_test_data.nc',
        map_variables=True)
    assert 'temp_air' in data.columns
    assert 'ghi_clear' in data.columns
    assert_index_equal(data.index, expected_index[:24])
    assert data.notna().all().all()


@requires_xarray
@requires_cds_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5_area(cds_api_key, expected_index):
    data, meta = get_era5(
        latitude=[55.7, 55.7+0.25*2],
        longitude=[12.5, 55.7+0.25*2],
        start=pd.Timestamp(2020, 1, 1),
        end=pd.Timestamp(2020, 1, 1),
        variables=['mean_surface_downward_short_wave_radiation_flux_clear_sky',
                   '2m_temperature'],
        api_key=cds_api_key,
        save_path='era5_test_data.nc',
        map_variables=True)
    assert 'temp_air' in data.variables.mapping.keys()
    assert 'time' in data.variables.mapping.keys()
    assert 'longitude' in data.variables.mapping.keys()
    assert np.isclose(data.latitude.values, [56.2, 55.95, 55.7]).all()
    assert (data.time.values ==
            expected_index[:24].to_pydatetime().astype('datetime64[ns]')).all()
