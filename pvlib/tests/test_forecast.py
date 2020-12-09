from datetime import datetime, timedelta, timezone
import warnings

import pandas as pd

import pytest
from numpy.testing import assert_allclose

from conftest import (
    requires_siphon,
    has_siphon,
    skip_windows,
    requires_recent_cftime
)
from conftest import RERUNS, RERUNS_DELAY

pytestmark = pytest.mark.skipif(not has_siphon, reason='requires siphon')


if has_siphon:
    with warnings.catch_warnings():
        # don't emit import warning
        warnings.simplefilter("ignore")
        from pvlib.forecast import GFS, HRRR_ESRL, HRRR, NAM, NDFD, RAP

    # setup times and location to be tested. Tucson, AZ
    _latitude = 32.2
    _longitude = -110.9
    _tz = 'US/Arizona'
    _start = pd.Timestamp.now(tz=_tz)
    _end = _start + pd.Timedelta(days=1)
    _modelclasses = [
        GFS, NAM, HRRR, NDFD, RAP,
        pytest.param(
            HRRR_ESRL, marks=[
                skip_windows,
                pytest.mark.xfail(reason="HRRR_ESRL is unreliable"),
                pytest.mark.timeout(timeout=60),
                pytest.mark.filterwarnings('ignore:.*experimental')])]
    _working_models = []
    _variables = ['temp_air', 'wind_speed', 'total_clouds', 'low_clouds',
                  'mid_clouds', 'high_clouds', 'dni', 'dhi', 'ghi']
    _nonnan_variables = ['temp_air', 'wind_speed', 'total_clouds', 'dni',
                         'dhi', 'ghi']
else:
    _modelclasses = []


# make a model object for each model class
# get the data for that model and store it in an
# attribute for further testing
@requires_siphon
@pytest.fixture(scope='module', params=_modelclasses)
def model(request):
    amodel = request.param()
    try:
        raw_data = amodel.get_data(_latitude, _longitude, _start, _end)
    except Exception as e:
        warnings.warn('Exception getting data for {}.\n'
                      'latitude, longitude, start, end = {} {} {} {}\n{}'
                      .format(amodel, _latitude, _longitude, _start, _end, e))
        raw_data = pd.DataFrame()  # raw_data.empty will be used later
    amodel.raw_data = raw_data
    return amodel


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_process_data(model):
    for how in ['campbell_norman', 'clearsky_scaling']:
        if model.raw_data.empty:
            warnings.warn('Could not test {} process_data with how={} '
                          'because raw_data was empty'.format(model, how))
            continue
        data = model.process_data(model.raw_data, how=how)
        for variable in _nonnan_variables:
            try:
                assert not data[variable].isnull().values.any()
            except AssertionError:
                warnings.warn('{}, {}, data contained null values'
                              .format(model, variable))


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_bad_kwarg_get_data():
    # For more information on why you would want to pass an unknown keyword
    # argument, see Github issue #745.
    amodel = NAM()
    data = amodel.get_data(_latitude, _longitude, _start, _end,
                           bad_kwarg=False)
    assert not data.empty


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_bad_kwarg_get_processed_data():
    # For more information on why you would want to pass an unknown keyword
    # argument, see Github issue #745.
    amodel = NAM()
    data = amodel.get_processed_data(_latitude, _longitude, _start, _end,
                                     bad_kwarg=False)
    assert not data.empty


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_how_kwarg_get_processed_data():
    amodel = NAM()
    data = amodel.get_processed_data(_latitude, _longitude, _start, _end,
                                     how='clearsky_scaling')
    assert not data.empty


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_vert_level():
    amodel = NAM()
    vert_level = 5000
    amodel.get_processed_data(_latitude, _longitude, _start, _end,
                              vert_level=vert_level)


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_datetime():
    amodel = NAM()
    start = datetime.now(tz=timezone.utc)
    end = start + timedelta(days=1)
    amodel.get_processed_data(_latitude, _longitude, start, end)


@requires_siphon
@requires_recent_cftime
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_queryvariables():
    amodel = GFS()
    new_variables = ['u-component_of_wind_height_above_ground']
    data = amodel.get_data(_latitude, _longitude, _start, _end,
                           query_variables=new_variables)
    data['u-component_of_wind_height_above_ground']


@requires_siphon
def test_latest():
    GFS(set_type='latest')


@requires_siphon
def test_full():
    GFS(set_type='full')


def test_temp_convert():
    amodel = GFS()
    data = pd.DataFrame({'temp_air': [273.15]})
    data['temp_air'] = amodel.kelvin_to_celsius(data['temp_air'])

    assert_allclose(data['temp_air'].values, 0.0)


# @requires_siphon
# def test_bounding_box():
#     amodel = GFS()
#     latitude = [31.2,32.2]
#     longitude = [-111.9,-110.9]
#     new_variables = {'temperature':'Temperature_surface'}
#     data = amodel.get_query_data(latitude, longitude, _start, _end,
#                                  variables=new_variables)


def test_set_location():
    amodel = GFS()
    latitude, longitude = 32.2, -110.9
    time = 'UTC'
    amodel.set_location(time, latitude, longitude)


def test_set_query_time_range_tzfail():
    amodel = GFS()
    with pytest.raises(TypeError):
        amodel.set_query_time_range(datetime.now(), datetime.now())


def test_cloud_cover_to_transmittance_linear():
    amodel = GFS()
    assert_allclose(amodel.cloud_cover_to_transmittance_linear(0), 0.75)
    assert_allclose(amodel.cloud_cover_to_transmittance_linear(100), 0.0)
    assert_allclose(amodel.cloud_cover_to_transmittance_linear(0, 0.5), 0.5)


def test_cloud_cover_to_ghi_linear():
    amodel = GFS()
    ghi_clear = 1000
    offset = 25
    out = amodel.cloud_cover_to_ghi_linear(0, ghi_clear, offset=offset)
    assert_allclose(out, 1000)
    out = amodel.cloud_cover_to_ghi_linear(100, ghi_clear, offset=offset)
    assert_allclose(out, 250)
