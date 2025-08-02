import pandas as pd
import numpy as np
import pytest
import pvlib
from tests.conftest import RERUNS, RERUNS_DELAY


@pytest.fixture
def demo_api_key():
    # Demo locations:
    # lat=50, lon=10 (Germany)
    # lat=21, lon=79 (India)
    # lat=-3, lon=-60 (Brazil)
    # lat=51, lon=-114 (Canada)
    # lat=24, lon=33 (Egypt)
    return 'demo0000-0000-0000-0000-000000000000'


@pytest.fixture
def demo_url():
    return 'https://demo.meteonorm.com/v1/'


@pytest.fixture
def expected_meta():
    meta = {
        'altitude': 290,
        'frequency': '1_hour',
        'parameters': [
           {'aggregation_method': 'average',
            'description': 'Diffuse horizontal irradiance',
            'name': 'diffuse_horizontal_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Diffuse horizontal irradiance with shading taken into account',
            'name': 'diffuse_horizontal_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Diffuse tilted irradiance',
            'name': 'diffuse_tilted_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Diffuse tilted irradiance with shading taken into account',
            'name': 'diffuse_tilted_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Direct horizontal irradiance',
            'name': 'direct_horizontal_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Direct horizontal irradiance with shading taken into account',
            'name': 'direct_horizontal_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Direct normal irradiance',
            'name': 'direct_normal_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Direct normal irradiance with shading taken into account',
            'name': 'direct_normal_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Direct tilted irradiance',
            'name': 'direct_tilted_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Direct tilted irradiance with shading taken into account',
            'name': 'direct_tilted_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Global horizontal clear sky irradiance',
            'name': 'global_clear_sky_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Global horizontal irradiance',
            'name': 'global_horizontal_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Global horizontal irradiance with shading taken into account',
            'name': 'global_horizontal_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Global tilted irradiance',
            'name': 'global_tilted_irradiance',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Global tilted irradiance with shading taken into account',
            'name': 'global_tilted_irradiance_with_shading',
            'unit': {'description': 'Watt per square meter', 'name': 'W/m**2'}},
           {'aggregation_method': 'average',
            'description': 'Power output per kWp installed',
            'name': 'pv_production',
            'unit': {'description': 'Watts per kilowatt peak', 'name': 'W/kWp'}},
           {'aggregation_method': 'average',
            'description': 'Power output per kWp installed, with shading taken into account',
            'name': 'pv_production_with_shading',
            'unit': {'description': 'Watts per kilowatt peak', 'name': 'W/kWp'}},
           {'aggregation_method': 'average',
            'description': 'Snow depth',
            'name': 'snow_depth',
            'unit': {'description': 'millimeters', 'name': 'mm'}},
           {'aggregation_method': 'average',
            'description': 'Air temperature, 2 m above ground.',
            'name': 'temperature',
            'unit': {'description': 'degrees Celsius', 'name': '°C'}}],
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
        pd.date_range('2023-01-01', '2024-12-31 23:59', freq='1h', tz='UTC')
    expected_meteonorm_index.freq = None
    return expected_meteonorm_index


@pytest.fixture
def expected_metenorm_data():
    # The first 12 rows of data
    columns = ['dhi', 'diffuse_horizontal_irradiance_with_shading', 'poa_diffuse',
               'diffuse_tilted_irradiance_with_shading', 'bhi',
               'direct_horizontal_irradiance_with_shading', 'dni',
               'direct_normal_irradiance_with_shading', 'poa_direct',
               'direct_tilted_irradiance_with_shading', 'ghi_clear', 'ghi',
               'global_horizontal_irradiance_with_shading', 'poa',
               'global_tilted_irradiance_with_shading', 'pv_production',
               'pv_production_with_shading', 'snow_depth', 'temp_air']
    expected = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 12.25],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11.75],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11.75],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11.5],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11.25],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 11.],
        [2.5, 2.68309898, 2.67538201, 2.68309898, 0., 0., 0., 0., 0., 0., 0., 2.5,
         2.68309898, 2.67538201, 2.68309898, 2.34649978, 2.35326557, 0., 11.],
        [40.43632435, 40.41304027, 40.43632435, 40.41304027, 37.06367565, 37.06367565,
         288.7781947, 288.7781947, 37.06367565, 37.06367565, 98.10113439, 77.5,
         77.47671591, 77.5, 77.47671591, 67.02141875, 67.00150474, 0., 11.75],
        [60.52591348, 60.51498257, 60.52591348, 60.51498257, 104.47408652, 104.47408652,
         478.10101591, 478.10101591, 104.47408652, 104.47408652, 191.27910925, 165.,
         164.98906908, 165., 164.98906908, 140.23845, 140.22938131, 0., 12.75],
        [71.90169306, 71.89757085, 71.90169306, 71.89757085, 138.84830694, 138.84830694,
         508.02986044, 508.02986044, 138.84830694, 138.84830694, 253.85597777, 210.75,
         210.7458778, 210.75, 210.7458778, 177.07272956, 177.06937293, 0., 13.75],
        [78.20403711, 78.19681926, 78.20403711, 78.19681926, 142.79596289, 142.79596289,
         494.06576548, 494.06576548, 142.79596289, 142.79596289, 272.34275335, 221.,
         220.99278214, 221., 220.99278214, 185.179657, 185.17380523, 0., 14.],
    ]
    index = pd.date_range('2023-01-01', periods=12, freq='1h', tz='UTC')
    index.freq = None
    expected = pd.DataFrame(expected, index=index, columns=columns)
    expected['snow_depth'] = expected['snow_depth'].astype(np.int64)
    return expected


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_training(
        demo_api_key, demo_url, expected_meta, expected_meteonorm_index,
        expected_metenorm_data):
    data, meta = pvlib.iotools.get_meteonorm(
        latitude=50, longitude=10,
        start='2023-01-01', end='2025-01-01',
        api_key=demo_api_key,
        endpoint='observation/training',
        time_step='1h',
        url=demo_url)

    assert meta == expected_meta
    pd.testing.assert_index_equal(data.index, expected_meteonorm_index)
    pd.testing.assert_frame_equal(data.iloc[:12], expected_metenorm_data)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_meteonorm_realtime(
        demo_api_key, demo_url, expected_meta, expected_meteonorm_index,
        expected_metenorm_data):
    data, meta = pvlib.iotools.get_meteonorm(
        latitude=21, longitude=79,
        start=pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=5),
        end=pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1),
        surface_tilt=20, surface_azimuth=10,
        parameters=['ghi', 'global_horizontal_irradiance_with_shading'],
        api_key=demo_api_key,
        endpoint='/observation/realtime',
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

    assert all(data.columns == pd.Index([
        'global_horizontal_irradiance',
        'global_horizontal_irradiance_with_shading']))
    assert data.shape == (241, 2)
    # can't test the specific index as it varies due to the
    # use of pd.Timestamp.now
    assert type(data.index) is pd.core.indexes.interval.IntervalIndex
