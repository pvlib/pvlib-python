"""
Created on Wed Oct  2 10:14:16 2019

@author: cwhanse
"""

import numpy as np
import pandas as pd

import pytest
from conftest import assert_series_equal
from numpy.testing import assert_allclose

from pvlib import iam as _iam


def test_ashrae():
    thetas = np.array([-90., -67.5, -45., -22.5, 0., 22.5, 45., 67.5, 89., 90.,
                       np.nan])
    expected = np.array([0, 0.9193437, 0.97928932, 0.99588039, 1., 0.99588039,
                         0.97928932, 0.9193437, 0, 0, np.nan])
    iam = _iam.ashrae(thetas, .05)
    assert_allclose(iam, expected, equal_nan=True)
    iam_series = _iam.ashrae(pd.Series(thetas))
    assert_series_equal(iam_series, pd.Series(expected))


def test_ashrae_scalar():
    thetas = -45.
    iam = _iam.ashrae(thetas, .05)
    expected = 0.97928932
    assert_allclose(iam, expected, equal_nan=True)
    thetas = np.nan
    iam = _iam.ashrae(thetas, .05)
    expected = np.nan
    assert_allclose(iam, expected, equal_nan=True)


def test_physical():
    aoi = np.array([-90., -67.5, -45., -22.5, 0., 22.5, 45., 67.5, 90.,
                    np.nan])
    expected = np.array([0, 0.8893998, 0.98797788, 0.99926198, 1, 0.99926198,
                         0.98797788, 0.8893998, 0, np.nan])
    iam = _iam.physical(aoi, 1.526, 0.002, 4)
    assert_allclose(iam, expected, equal_nan=True)

    # GitHub issue 397
    aoi = pd.Series(aoi)
    iam = _iam.physical(aoi, 1.526, 0.002, 4)
    expected = pd.Series(expected)
    assert_series_equal(iam, expected)


def test_physical_scalar():
    aoi = -45.
    iam = _iam.physical(aoi, 1.526, 0.002, 4)
    expected = 0.98797788
    assert_allclose(iam, expected, equal_nan=True)
    aoi = np.nan
    iam = _iam.physical(aoi, 1.526, 0.002, 4)
    expected = np.nan
    assert_allclose(iam, expected, equal_nan=True)


def test_martin_ruiz():

    aoi = 45.
    a_r = 0.16
    expected = 0.98986965

    # will fail if default values change
    iam = _iam.martin_ruiz(aoi)
    assert_allclose(iam, expected)

    # will fail if parameter names change
    iam = _iam.martin_ruiz(aoi=aoi, a_r=a_r)
    assert_allclose(iam, expected)

    a_r = 0.18
    aoi = [-100, -60, 0, 60, 100, np.nan, np.inf]
    expected = [0.0, 0.9414631, 1.0, 0.9414631, 0.0, np.nan, 0.0]

    # check out of range of inputs as list
    iam = _iam.martin_ruiz(aoi, a_r)
    assert_allclose(iam, expected, equal_nan=True)

    # check out of range of inputs as array
    iam = _iam.martin_ruiz(np.array(aoi), a_r)
    assert_allclose(iam, expected, equal_nan=True)

    # check out of range of inputs as Series
    aoi = pd.Series(aoi)
    expected = pd.Series(expected)
    iam = _iam.martin_ruiz(aoi, a_r)
    assert_series_equal(iam, expected)


def test_martin_ruiz_exception():

    with pytest.raises(ValueError):
        _iam.martin_ruiz(0.0, a_r=0.0)


def test_martin_ruiz_diffuse():

    surface_tilt = 30.
    a_r = 0.16
    expected = (0.9549735, 0.7944426)

    # will fail if default values change
    iam = _iam.martin_ruiz_diffuse(surface_tilt)
    assert_allclose(iam, expected)

    # will fail if parameter names change
    iam = _iam.martin_ruiz_diffuse(surface_tilt=surface_tilt, a_r=a_r)
    assert_allclose(iam, expected)

    a_r = 0.18
    surface_tilt = [0, 30, 90, 120, 180, np.nan, np.inf]
    expected_sky = [0.9407678, 0.9452250, 0.9407678, 0.9055541, 0.0000000,
                    np.nan, np.nan]
    expected_gnd = [0.0000000, 0.7610849, 0.9407678, 0.9483508, 0.9407678,
                    np.nan, np.nan]

    # check various inputs as list
    iam = _iam.martin_ruiz_diffuse(surface_tilt, a_r)
    assert_allclose(iam[0], expected_sky, atol=1e-7, equal_nan=True)
    assert_allclose(iam[1], expected_gnd, atol=1e-7, equal_nan=True)

    # check various inputs as array
    iam = _iam.martin_ruiz_diffuse(np.array(surface_tilt), a_r)
    assert_allclose(iam[0], expected_sky, atol=1e-7, equal_nan=True)
    assert_allclose(iam[1], expected_gnd, atol=1e-7, equal_nan=True)

    # check various inputs as Series
    surface_tilt = pd.Series(surface_tilt)
    expected_sky = pd.Series(expected_sky, name='iam_sky')
    expected_gnd = pd.Series(expected_gnd, name='iam_ground')
    iam = _iam.martin_ruiz_diffuse(surface_tilt, a_r)
    assert_series_equal(iam[0], expected_sky)
    assert_series_equal(iam[1], expected_gnd)


def test_iam_interp():

    aoi_meas = [0.0, 45.0, 65.0, 75.0]
    iam_meas = [1.0,  0.9,  0.8,  0.6]

    # simple default linear method
    aoi = 55.0
    expected = 0.85
    iam = _iam.interp(aoi, aoi_meas, iam_meas)
    assert_allclose(iam, expected)

    # simple non-default method
    aoi = 55.0
    expected = 0.8878062
    iam = _iam.interp(aoi, aoi_meas, iam_meas, method='cubic')
    assert_allclose(iam, expected)

    # check with all reference values
    aoi = aoi_meas
    expected = iam_meas
    iam = _iam.interp(aoi, aoi_meas, iam_meas)
    assert_allclose(iam, expected)

    # check normalization and Series
    aoi = pd.Series(aoi)
    expected = pd.Series(expected)
    iam_mult = np.multiply(0.9, iam_meas)
    iam = _iam.interp(aoi, aoi_meas, iam_mult, normalize=True)
    assert_series_equal(iam, expected)

    # check beyond reference values
    aoi = [-45, 0, 45, 85, 90, 95, 100, 105, 110]
    expected = [0.9, 1.0, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]
    iam = _iam.interp(aoi, aoi_meas, iam_meas)
    assert_allclose(iam, expected)

    # check exception clause
    with pytest.raises(ValueError):
        _iam.interp(0.0, [0], [1])

    # check exception clause
    with pytest.raises(ValueError):
        _iam.interp(0.0, [0, 90], [1, -1])


@pytest.mark.parametrize('aoi,expected', [
    (45, 0.9975036250000002),
    (np.array([[-30, 30, 100, np.nan]]),
     np.array([[0, 1.007572, 0, np.nan]])),
    (pd.Series([80]), pd.Series([0.597472]))
])
def test_sapm(sapm_module_params, aoi, expected):

    out = _iam.sapm(aoi, sapm_module_params)

    if isinstance(aoi, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-4)


def test_sapm_limits():
    module_parameters = {'B0': 5, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    assert _iam.sapm(1, module_parameters) == 5

    module_parameters = {'B0': 5, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    assert _iam.sapm(1, module_parameters, upper=1) == 1

    module_parameters = {'B0': -5, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    assert _iam.sapm(1, module_parameters) == 0


def test_marion_diffuse_model(mocker):
    # 1: return values are correct
    # 2: the underlying models are called appropriately
    ashrae_expected = {
        'sky': 0.9596085829811408,
        'horizon': 0.8329070417832541,
        'ground': 0.719823559106309
    }
    physical_expected = {
        'sky': 0.9539178294437575,
        'horizon': 0.7652650139134007,
        'ground': 0.6387140117795903
    }
    ashrae_spy = mocker.spy(_iam, 'ashrae')
    physical_spy = mocker.spy(_iam, 'physical')

    ashrae_actual = _iam.marion_diffuse('ashrae', 20)
    assert ashrae_spy.call_count == 3  # one call for each of the 3 regions
    assert physical_spy.call_count == 0
    physical_actual = _iam.marion_diffuse('physical', 20)
    assert ashrae_spy.call_count == 3
    assert physical_spy.call_count == 3

    for k, v in ashrae_expected.items():
        np.testing.assert_allclose(ashrae_actual[k], v)

    for k, v in physical_expected.items():
        np.testing.assert_allclose(physical_actual[k], v)


def test_marion_diffuse_kwargs():
    # kwargs get passed to underlying model
    expected = {
        'sky': 0.967489994422575,
        'horizon': 0.8647842827418412,
        'ground': 0.7700443455928433
    }
    actual = _iam.marion_diffuse('ashrae', 20, b=0.04)

    for k, v in expected.items():
        np.testing.assert_allclose(actual[k], v)


def test_marion_diffuse_invalid():
    with pytest.raises(ValueError):
        _iam.marion_diffuse('not_a_model', 20)


@pytest.mark.parametrize('region,N,expected', [
    ('sky', 180, 0.9596085829811408),
    ('horizon', 1800, 0.8329070417832541),
    ('ground', 180, 0.719823559106309)
])
def test_marion_integrate_scalar(region, N, expected):
    actual = _iam.marion_integrate(_iam.ashrae, 20, region, N)
    assert_allclose(actual, expected)

    with np.errstate(invalid='ignore'):
        actual = _iam.marion_integrate(_iam.ashrae, np.nan, region, N)
    expected = np.nan
    assert_allclose(actual, expected)


@pytest.mark.parametrize('region,N,expected', [
    ('sky', 180, [0.9523611991069362, 0.9596085829811408, 0.9619811198105501]),
    ('horizon', 1800, [0.0, 0.8329070417832541, 0.8987287652347437]),
    ('ground', 180, [0.0, 0.719823559106309, 0.8186360238536674])
])
def test_marion_integrate_list(region, N, expected):
    actual = _iam.marion_integrate(_iam.ashrae, [0, 20, 30], region, N)
    assert_allclose(actual, expected)

    with np.errstate(invalid='ignore'):
        actual = _iam.marion_integrate(_iam.ashrae, [0, 20, np.nan], region, N)
    assert_allclose(actual, [expected[0], expected[1], np.nan])


@pytest.mark.parametrize('region,N,expected', [
    ('sky', 180, [0.9523611991069362, 0.9596085829811408, 0.9619811198105501]),
    ('horizon', 1800, [0.0, 0.8329070417832541, 0.8987287652347437]),
    ('ground', 180, [0.0, 0.719823559106309, 0.8186360238536674])
])
def test_marion_integrate_series(region, N, expected):
    idx = pd.date_range('2019-01-01', periods=3, freq='h')
    tilt = pd.Series([0, 20, 30], index=idx)
    expected = pd.Series(expected, index=idx)
    actual = _iam.marion_integrate(_iam.ashrae, tilt, region, N)
    assert_series_equal(actual, expected)

    tilt.iloc[1] = np.nan
    expected.iloc[1] = np.nan
    with np.errstate(invalid='ignore'):
        actual = _iam.marion_integrate(_iam.ashrae, tilt, region, N)
    assert_allclose(actual, expected)


def test_marion_integrate_ground_flat():
    iam = _iam.marion_integrate(_iam.ashrae, 0, 'horizon', 1800)
    assert_allclose(iam, 0)


def test_marion_integrate_invalid():
    # check for invalid region string.  this actually gets checked twice,
    # with the difference being whether `num` is specified or not.
    with pytest.raises(ValueError):
        _iam.marion_integrate(_iam.ashrae, 0, 'bad')

    with pytest.raises(ValueError):
        _iam.marion_integrate(_iam.ashrae, 0, 'bad', 180)
