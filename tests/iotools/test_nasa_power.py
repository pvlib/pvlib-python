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


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_all_variable_map_parameters_valid():
    """
    Every NASA POWER parameter name in VARIABLE_MAP must be accepted by the
    live API. A typo or stale name (e.g. CLRSKY_DIFF vs CLRSKY_SFC_SW_DIFF)
    causes the API to return an HTTPError, which would fail this test.

    NASA POWER allows max 15 parameters per request; VARIABLE_MAP fits within
    that. If the map grows past 15, split this test into batches.
    """
    from pvlib.iotools.nasa_power import VARIABLE_MAP
    nasa_params = list(VARIABLE_MAP.keys())
    assert len(nasa_params) <= 15, (
        "VARIABLE_MAP exceeds NASA POWER's 15-parameter-per-request limit; "
        "split this test into batches."
    )

    data, meta = pvlib.iotools.get_nasa_power(
        latitude=44.76,
        longitude=7.64,
        start='2025-02-02',
        end='2025-02-02',
        parameters=nasa_params,
        map_variables=False,
    )

    # Every requested NASA parameter must come back as a column.
    missing = set(nasa_params) - set(data.columns)
    assert not missing, f"NASA POWER did not return: {sorted(missing)}"


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_all_variable_map_renamed():
    """
    With map_variables=True every NASA name must be renamed to its pvlib
    equivalent. Catches duplicate-target collisions (e.g. two entries both
    mapping to 'temp_dew', which silently drops a column under pandas rename).
    """
    from pvlib.iotools.nasa_power import VARIABLE_MAP
    nasa_params = list(VARIABLE_MAP.keys())
    pvlib_names = set(VARIABLE_MAP.values())

    data, _ = pvlib.iotools.get_nasa_power(
        latitude=44.76,
        longitude=7.64,
        start='2025-02-02',
        end='2025-02-02',
        parameters=nasa_params,
        map_variables=True,
    )

    missing = pvlib_names - set(data.columns)
    assert not missing, (
        f"map_variables=True dropped pvlib columns: {sorted(missing)}. "
        "Likely cause: duplicate target names in VARIABLE_MAP."
    )


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_pressure_unit_conversion():
    """
    NASA POWER returns PS in kPa; pvlib convention is Pa.
    Sea-level surface pressure should be ~101 kPa = ~101325 Pa, not ~101.
    """
    data, _ = pvlib.iotools.get_nasa_power(
        latitude=0.0,  # sea-level equatorial point
        longitude=-30.0,
        start='2025-02-02',
        end='2025-02-02',
        parameters=['PS'],
        map_variables=True,
    )
    mean_pressure = data['pressure'].mean()
    # Anywhere on earth, surface pressure in Pa is between ~50k and ~110k.
    # If the conversion is missing, value will be ~100 (kPa) and fail this.
    assert 50_000 < mean_pressure < 110_000, (
        f"PS not converted from kPa to Pa. Got mean={mean_pressure}"
    )


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_nasa_power_precipitable_water_unit_conversion():
    """
    NASA POWER returns TQV in kg/m^2 (= mm of water column);
    pvlib convention is cm. Typical atmospheric column is 1-5 cm,
    never above ~7 cm. If the /10 conversion is missing, values will
    be 10-50 and fail this bound.
    """
    data, _ = pvlib.iotools.get_nasa_power(
        latitude=0.0,  # tropics: high water vapor, worst case for bounds
        longitude=-60.0,
        start='2025-02-02',
        end='2025-02-02',
        parameters=['TQV'],
        map_variables=True,
    )
    mean_pw = data['precipitable_water'].mean()
    # pvlib precipitable_water is in cm. Tropical column is <~7 cm.
    # Missing /10 conversion would give 10-70 (kg/m^2 = mm).
    assert 0 < mean_pw < 10, (
        f"TQV not converted from kg/m^2 to cm. Got mean={mean_pw}"
    )
