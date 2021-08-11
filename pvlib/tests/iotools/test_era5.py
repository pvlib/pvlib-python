"""
tests for :mod:`pvlib.iotools.era5`
"""

import pandas as pd
import pytest
import os
from pvlib.iotools import read_era5, get_era5
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal,
                        requires_cds_credentials)


@pytest.fixture(scope="module")
def cds_api_key():
    """Supplies pvlib-python's CDS API key.

    Users should obtain their own credentials as described in the `get_bsrn`
    documentation."""
    return '98568:' + os.environ["CDSAPI_KEY"]


@pytest.fixture
def expected_index():
    index = pd.date_range('2020-1-1', freq='1h', periods=8784)
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


def test_read_era5(expected_index, expected_columns):
    data, meta = read_era5(DATA_DIR / 'era5_testfile.nc', map_variables=False)
    assert (expected_columns == data.columns).all()
    assert_index_equal(data.index, expected_index)


def test_read_era5_variable_mapped(expected_index, expected_columns_mapped):
    data, meta = read_era5(DATA_DIR / 'era5_testfile.nc')
    assert (expected_columns_mapped == data.columns).all()
    assert_index_equal(data.index, expected_index)
    assert data.notna().all().all()


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
        local_filename='era5_test_data.nc',  # save file
        map_variables=True)
    assert 'temp_air' in data.columns
    assert 'ghi_clear' in data.columns
    assert_index_equal(data.index, expected_index[:24])
    assert data.notna().all().all()
