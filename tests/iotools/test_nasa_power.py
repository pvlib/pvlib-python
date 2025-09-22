import pandas as pd
import pytest
import pvlib
from requests.exceptions import HTTPError
from tests.conftest import RERUNS, RERUNS_DELAY


@pytest.fixture
def data_index():
    index = pd.date_range(start='2025-02-02 00:00+00:00',
                          end='2025-02-02 23:00+00:00', freq='1h')
    return index


@pytest.fixture
def ghi_series(data_index):
    ghi = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.25, 184.2, 281.55, 368.3, 406.48,
        386.45, 316.05, 210.1, 109.05, 12.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    return pd.Series(data=ghi, index=data_index, name='ghi')


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power(data_index, ghi_series):
    data, meta = pvlib.iotools.get_nasa_power(latitude=44.76,
                                              longitude=7.64,
                                              start=data_index[0],
                                              end=data_index[-1],
                                              parameters=['ALLSKY_SFC_SW_DWN'],
                                              map_variables=False)
    # Check that metadata is correct
    assert meta['latitude'] == 44.76
    assert meta['longitude'] == 7.64
    assert meta['altitude'] == 705.88
    assert meta['start'] == '20250202'
    assert meta['end'] == '20250202'
    assert meta['time_standard'] == 'UTC'
    assert meta['title'] == 'NASA/POWER Source Native Resolution Hourly Data'
    # Assert that the index is parsed correctly
    pd.testing.assert_index_equal(data.index, data_index)
    # Test one column
    pd.testing.assert_series_equal(data['ALLSKY_SFC_SW_DWN'], ghi_series,
                                   check_freq=False, check_names=False)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_pvlib_params_naming(data_index, ghi_series):
    data, meta = pvlib.iotools.get_nasa_power(latitude=44.76,
                                              longitude=7.64,
                                              start=data_index[0],
                                              end=data_index[-1],
                                              parameters=['ghi'])
    # Assert that the index is parsed correctly
    pd.testing.assert_index_equal(data.index, data_index)
    # Test one column
    pd.testing.assert_series_equal(data['ghi'], ghi_series,
                                   check_freq=False)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_map_variables(data_index):
    # Check that variables are mapped by default to pvlib names
    data, meta = pvlib.iotools.get_nasa_power(latitude=44.76,
                                              longitude=7.64,
                                              start=data_index[0],
                                              end=data_index[-1])
    mapped_column_names = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
    for c in mapped_column_names:
        assert c in data.columns
    assert meta['latitude'] == 44.76
    assert meta['longitude'] == 7.64
    assert meta['altitude'] == 705.88


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_wrong_parameter_name(data_index):
    # Test if HTTPError is raised if a wrong parameter name is asked
    with pytest.raises(HTTPError, match=r"ALLSKY_SFC_SW_DLN"):
        pvlib.iotools.get_nasa_power(latitude=44.76,
                                     longitude=7.64,
                                     start=data_index[0],
                                     end=data_index[-1],
                                     parameters=['ALLSKY_SFC_SW_DLN'])


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_duplicate_parameter_name(data_index):
    # Test if HTTPError is raised if a duplicate parameter is asked
    with pytest.raises(HTTPError, match=r"ALLSKY_SFC_SW_DWN"):
        pvlib.iotools.get_nasa_power(latitude=44.76,
                                     longitude=7.64,
                                     start=data_index[0],
                                     end=data_index[-1],
                                     parameters=2*['ALLSKY_SFC_SW_DWN'])
