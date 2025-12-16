import pandas as pd
import numpy as np
import pytest
import pvlib
from tests.conftest import RERUNS, RERUNS_DELAY
from requests.exceptions import HTTPError


@pytest.fixture
def demo_api_key():
    # Demo locations:
    # lat=50, lon=10 (Germany)
    # lat=21, lon=79 (India)
    # lat=-3, lon=-60 (Brazil)
    # lat=51, lon=-114 (Canada)
    # lat=24, lon=33 (Egypt)
    demo_api_key = 'demo0000-0000-0000-0000-000000000000'
    return demo_api_key


@pytest.fixture
def demo_url():
    demo_url = 'https://demo.meteonorm.com/v1/'
    return demo_url


@pytest.fixture
def expected_meta():
    meta = {
        'altitude': 290,
        'frequency': '1_hour',
        'parameters': [
           {'aggregation_method': 'average',
            'description': 'Global horizontal irradiance',
            'name': 'global_horizontal_irradiance',
            'unit': {
                'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Global horizontal irradiance with shading taken into account',  # noqa: E501
            'name': 'global_horizontal_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter',
                     'name': 'W/m**2'}},
        ],
        'surface_azimuth': 180,
        'surface_tilt': 0,
        'time_zone': 0,
        'latitude': 50,
        'longitude': 10,
    }
    return meta


@pytest.fixture
def expected_meteonorm_index():
    expected_meteonorm_index = \
        pd.date_range('2023-01-01', '2023-12-31 23:59', freq='1h', tz='UTC') \
        + pd.Timedelta(minutes=30)
    expected_meteonorm_index.freq = None
    return expected_meteonorm_index


@pytest.fixture
def expected_meteonorm_data():
    # The first 12 rows of data
    columns = ['ghi', 'global_horizontal_irradiance_with_shading']
    expected = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [2.5, 2.68],
        [77.5, 77.48],
        [165.0, 164.99],
        [210.75, 210.75],
        [221.0, 220.99],
    ]
    index = pd.date_range('2023-01-01 00:30', periods=12, freq='1h', tz='UTC')
    index.freq = None
    expected = pd.DataFrame(expected, index=index, columns=columns)
    return expected


@pytest.fixture
def expected_columns_all():
    columns = [
        'diffuse_horizontal_irradiance',
        'diffuse_horizontal_irradiance_with_shading',
        'diffuse_tilted_irradiance',
        'diffuse_tilted_irradiance_with_shading',
        'direct_horizontal_irradiance',
        'direct_horizontal_irradiance_with_shading',
        'direct_normal_irradiance',
        'direct_normal_irradiance_with_shading',
        'direct_tilted_irradiance',
        'direct_tilted_irradiance_with_shading',
        'global_clear_sky_irradiance',
        'global_horizontal_irradiance',
        'global_horizontal_irradiance_with_shading',
        'global_tilted_irradiance',
        'global_tilted_irradiance_with_shading',
        'pv_production',
        'pv_production_with_shading',
        'snow_depth',
        'temperature',
    ]
    return columns


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_training(
        demo_api_key, demo_url, expected_meta, expected_meteonorm_index,
        expected_meteonorm_data):
    data, meta = pvlib.iotools.get_meteonorm_observation_training(
        latitude=50, longitude=10,
        start='2023-01-01', end='2024-01-01',
        api_key=demo_api_key,
        parameters=['ghi', 'global_horizontal_irradiance_with_shading'],
        time_step='1h',
        url=demo_url)

    assert meta.items() >= expected_meta.items()  # check stable subset
    for key in ['version', 'commit']:
        assert key in meta  # value changes, so only check presence
    pd.testing.assert_index_equal(data.index, expected_meteonorm_index)
    # meteonorm API only guarantees similar, not identical, results between
    # calls.  so we allow a small amount of variation with atol.
    pd.testing.assert_frame_equal(data.iloc[:12], expected_meteonorm_data,
                                  check_exact=False, atol=1)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_realtime(demo_api_key, demo_url, expected_columns_all):
    data, meta = pvlib.iotools.get_meteonorm_observation_realtime(
        latitude=21, longitude=79,
        start=pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=5),
        end=pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1),
        surface_tilt=20, surface_azimuth=10,
        parameters=['all'],
        api_key=demo_api_key,
        time_step='1min',
        horizon='flat',
        map_variables=False,
        interval_index=True,
        url=demo_url,
    )
    assert meta['frequency'] == '1_minute'
    assert meta['lat'] == 21
    assert meta['lon'] == 79
    assert meta['surface_tilt'] == 20
    assert meta['surface_azimuth'] == 10

    assert list(data.columns) == expected_columns_all
    assert data.shape == (241, 19)
    # can't test the specific index as it varies due to the
    # use of pd.Timestamp.now
    assert type(data.index) is pd.core.indexes.interval.IntervalIndex


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_forecast_basic(demo_api_key, demo_url):
    data, meta = pvlib.iotools.get_meteonorm_forecast_basic(
        latitude=50, longitude=10,
        start='+1hours',
        end=pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=6),
        api_key=demo_api_key,
        parameters='ghi',
        url=demo_url)

    assert data.shape == (6, 1)
    assert data.columns == pd.Index(['ghi'])
    assert data.index[1] - data.index[0] == pd.Timedelta(hours=1)
    assert meta['frequency'] == '1_hour'


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_forecast_precision(demo_api_key, demo_url):
    data, meta = pvlib.iotools.get_meteonorm_forecast_precision(
        latitude=50, longitude=10,
        start='now',
        end='+3hours',
        api_key=demo_api_key,
        parameters='ghi',
        time_step='15min',
        url=demo_url)

    assert data.index[1] - data.index[0] == pd.Timedelta(minutes=15)
    assert data.shape == (60/15*3+1, 1)
    assert meta['frequency'] == '15_minutes'


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_custom_horizon(demo_api_key, demo_url):
    data, meta = pvlib.iotools.get_meteonorm_forecast_basic(
        latitude=50, longitude=10,
        start=pd.Timestamp.now(tz='UTC'),
        end=pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=5),
        api_key=demo_api_key,
        parameters='ghi',
        horizon=list(np.ones(360).astype(int)*80),
        url=demo_url)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_forecast_HTTPError(demo_api_key, demo_url):
    with pytest.raises(
            HTTPError, match='invalid parameter "not_a_real_parameter"'):
        _ = pvlib.iotools.get_meteonorm_forecast_basic(
            latitude=50, longitude=10,
            start=pd.Timestamp.now(tz='UTC'),
            end=pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=5),
            api_key=demo_api_key,
            parameters='not_a_real_parameter',
            url=demo_url)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_tmy_HTTPError(demo_api_key, demo_url):
    with pytest.raises(
            HTTPError, match='parameter "surface_azimuth"'):
        _ = pvlib.iotools.get_meteonorm_tmy(
            latitude=50, longitude=10,
            api_key=demo_api_key,
            parameters='dhi',
            # Infeasible surface_tilt
            surface_azimuth=400,
            url=demo_url)


@pytest.fixture
def expected_meteonorm_tmy_meta():
    meta = {
        'altitude': 290,
        'frequency': '1_hour',
        'parameters': [{
            'aggregation_method': 'average',
            'description': 'Diffuse horizontal irradiance',
            'name': 'diffuse_horizontal_irradiance',
            'unit': {'description': 'Watt per square meter',
                     'name': 'W/m**2'},
        }],
        'surface_azimuth': 90,
        'surface_tilt': 20,
        'time_zone': 1,
        'lat': 50,
        'lon': 10,
    }
    return meta


@pytest.fixture
def expected_meteonorm_tmy_data():
    # The first 12 rows of data
    columns = ['diffuse_horizontal_irradiance']
    expected = [
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [9.07],
        [8.44],
        [86.64],
        [110.44],
    ]
    index = pd.date_range(
        '2030-01-01', periods=12, freq='1h', tz=3600)
    index.freq = None
    interval_index = pd.IntervalIndex.from_arrays(
        index, index + pd.Timedelta(hours=1), closed='left')
    expected = pd.DataFrame(expected, index=interval_index, columns=columns)
    return expected


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_tmy(
        demo_api_key, demo_url, expected_meteonorm_tmy_meta,
        expected_meteonorm_tmy_data):
    data, meta = pvlib.iotools.get_meteonorm_tmy(
        latitude=50, longitude=10,
        api_key=demo_api_key,
        parameters='dhi',
        surface_tilt=20,
        surface_azimuth=90,
        time_step='1h',
        horizon=list(np.ones(360).astype(int)*2),
        terrain_situation='open',
        albedo=0.5,
        turbidity=[5.2, 4, 3, 3.1, 3.0, 2.8, 3.14, 3.0, 3, 3, 4, 5],
        random_seed=100,
        clear_sky_radiation_model='solis',
        data_version='v9.0',  # fix version
        future_scenario='ssp1_26',
        future_year=2030,
        interval_index=True,
        map_variables=False,
        url=demo_url)
    assert meta.items() >= expected_meteonorm_tmy_meta.items()
    for key in ['version', 'commit']:
        assert key in meta  # value changes, so only check presence
    # meteonorm API only guarantees similar, not identical, results between
    # calls.  so we allow a small amount of variation with atol.
    pd.testing.assert_frame_equal(data.iloc[:12], expected_meteonorm_tmy_data,
                                  check_exact=False, atol=1)
