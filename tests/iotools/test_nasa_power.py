import pandas as pd
import pytest
import pvlib
from requests.exceptions import HTTPError


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
    assert meta['header']['start'] == '20250202'
    assert meta['header']['end'] == '20250202'
    assert meta['header']['time_standard'] == 'UTC'
    assert meta['header']['title'] == 'NASA/POWER Source Native Resolution Hourly Data'  # noqa: E501
    assert meta['header']['sources'][0] == 'SYN1DEG'
    # Check that columns are parsed correctly
    assert 'ALLSKY_SFC_SW_DWN' in data.columns
    # Assert that the index is parsed correctly
    pd.testing.assert_index_equal(data.index, data_index)
    # Test one column
    pd.testing.assert_series_equal(data['ALLSKY_SFC_SW_DWN'], ghi_series,
                                   check_freq=False, check_names=False)


def test_get_nasa_power_map_variables(data_index):
    # Check that variables are mapped by default to pvlib names
    data, meta = pvlib.iotools.get_nasa_power(latitude=44.76,
                                              longitude=7.64,
                                              start=data_index[0],
                                              end=data_index[-1],
                                              parameters=['ALLSKY_SFC_SW_DWN',
                                                          'ALLSKY_SFC_SW_DIFF',
                                                          'ALLSKY_SFC_SW_DNI',
                                                          'CLRSKY_SFC_SW_DWN',
                                                          'T2M', 'WS2M',
                                                          'WS10M'
                                                          ])
    mapped_column_names = ['ghi', 'dni', 'dhi', 'temp_air_2m', 'wind_speed_2m',
                           'wind_speed_10m', 'ghi_clear']
    for c in mapped_column_names:
        assert c in data.columns
    assert meta['latitude'] == 44.76
    assert meta['longitude'] == 7.64
    assert meta['altitude'] == 705.88


def test_get_nasa_power_wrong_parameter_name(data_index):
    # Test if HTTPError is raised if a wrong parameter name is asked
    with pytest.raises(HTTPError, match=r"ALLSKY_SFC_SW_DLN"):
        pvlib.iotools.get_nasa_power(latitude=44.76,
                                     longitude=7.64,
                                     start=data_index[0],
                                     end=data_index[-1],
                                     parameters=['ALLSKY_SFC_SW_DLN'])


def test_get_nasa_power_duplicate_parameter_name(data_index):
    # Test if HTTPError is raised if a duplicate parameter is asked
    with pytest.raises(HTTPError, match=r"ALLSKY_SFC_SW_DWN"):
        pvlib.iotools.get_nasa_power(latitude=44.76,
                                     longitude=7.64,
                                     start=data_index[0],
                                     end=data_index[-1],
                                     parameters=2*['ALLSKY_SFC_SW_DWN'])
