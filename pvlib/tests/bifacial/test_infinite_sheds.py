"""
test infinite sheds
"""

import numpy as np
import pandas as pd
from pvlib.bifacial import infinite_sheds
from pvlib.tools import cosd
from ..conftest import assert_series_equal

import pytest


@pytest.fixture
def test_system():
    syst = {'height': 1.0,
            'pitch': 2.,
            'surface_tilt': 30.,
            'surface_azimuth': 180.,
            'rotation': -30.}  # rotation of right edge relative to horizontal
    syst['gcr'] = 1.0 / syst['pitch']
    pts = np.linspace(0, 1, num=3)
    sqr3 = np.sqrt(3) / 4
    # c_i,j = cos(angle from point i to edge of row j), j=0 is row = -1
    # c_i,j = cos(angle from point i to edge of row j), j=0 is row = -1
    c00 = (-2 - sqr3) / np.sqrt(1.25**2 + (2 + sqr3)**2)  # right edge row -1
    c01 = -sqr3 / np.sqrt(1.25**2 + sqr3**2)  # right edge row 0
    c02 = sqr3 / np.sqrt(0.75**2 + sqr3**2)  # left edge of row 0
    c03 = (2 - sqr3) / np.sqrt(1.25**2 + (2 - sqr3)**2)  # right edge of row 1
    vf_0 = 0.5 * (c03 - c02 + c01 - c00)  # vf at point 0
    c10 = (-3 - sqr3) / np.sqrt(1.25**2 + (3 + sqr3)**2)  # right edge row -1
    c11 = (-1 - sqr3) / np.sqrt(1.25**2 + (1 + sqr3)**2)  # right edge row 0
    c12 = (-1 + sqr3) / np.sqrt(0.75**2 + (-1 + sqr3)**2)  # left edge row 0
    c13 = (1 - sqr3) / np.sqrt(1.25**2 + (1 - sqr3)**2)  # right edge row
    vf_1 = 0.5 * (c13 - c12 + c11 - c10)  # vf at point 1
    c20 = -(4 + sqr3) / np.sqrt(1.25**2 + (4 + sqr3)**2)  # right edge row -1
    c21 = (-2 + sqr3) / np.sqrt(0.75**2 + (-2 + sqr3)**2)  # left edge row 0
    c22 = (-2 - sqr3) / np.sqrt(1.25**2 + (2 + sqr3)**2)  # right edge row 0
    c23 = (0 - sqr3) / np.sqrt(1.25**2 + (0 - sqr3)**2)  # right edge row 1
    vf_2 = 0.5 * (c23 - c22 + c21 - c20)  # vf at point 1
    vfs_ground_sky = np.array([vf_0, vf_1, vf_2])
    return syst, pts, vfs_ground_sky


@pytest.mark.parametrize("vectorize", [True, False])
def test__vf_ground_sky_integ(test_system, vectorize):
    ts, pts, vfs_gnd_sky = test_system
    # pass rotation here since max_rows=1 for the hand-solved case in
    # the fixture test_system, which means the ground-to-sky view factor
    # isn't summed over enough rows for symmetry to hold.
    vf_integ = infinite_sheds._vf_ground_sky_integ(
        ts['rotation'], ts['surface_azimuth'],
        ts['gcr'], ts['height'], ts['pitch'],
        max_rows=1, npoints=3, vectorize=vectorize)
    expected_vf_integ = np.trapz(vfs_gnd_sky, pts)
    assert np.isclose(vf_integ, expected_vf_integ, rtol=0.1)


def test__vf_row_sky_integ(test_system):
    ts, _, _ = test_system
    gcr = ts['gcr']
    surface_tilt = ts['surface_tilt']
    f_x = np.array([0., 0.5, 1.])
    shaded = []
    noshade = []
    for x in f_x:
        s, ns = infinite_sheds._vf_row_sky_integ(
            x, surface_tilt, gcr, npoints=100)
        shaded.append(s)
        noshade.append(ns)

    def analytic(gcr, surface_tilt, x):
        c = cosd(surface_tilt)
        a = 1. / gcr
        dx = np.sqrt(a**2 - 2 * a * c * x + x**2)
        return - a * (c**2 - 1) * np.arctanh((x - a * c) / dx) - c * dx

    expected_shade = 0.5 * (f_x * cosd(surface_tilt)
                            - analytic(gcr, surface_tilt, 1 - f_x)
                            + analytic(gcr, surface_tilt, 1.))
    expected_noshade = 0.5 * ((1 - f_x) * cosd(surface_tilt)
                              + analytic(gcr, surface_tilt, 1. - f_x)
                              - analytic(gcr, surface_tilt, 0.))
    shaded = np.array(shaded)
    noshade = np.array(noshade)
    assert np.allclose(shaded, expected_shade)
    assert np.allclose(noshade, expected_noshade)


def test__poa_sky_diffuse_pv():
    dhi = np.array([np.nan, 0.0, 500.])
    f_x = np.array([0.2, 0.2, 0.5])
    vf_shade_sky_integ = np.array([1.0, 0.5, 0.2])
    vf_noshade_sky_integ = np.array([0.0, 0.5, 0.8])
    poa = infinite_sheds._poa_sky_diffuse_pv(
        f_x, dhi, vf_shade_sky_integ, vf_noshade_sky_integ)
    expected_poa = np.array([np.nan, 0.0, 500 * (0.5 * 0.2 + 0.5 * 0.8)])
    assert np.allclose(poa, expected_poa, equal_nan=True)


def test__ground_angle(test_system):
    ts, _, _ = test_system
    x = np.array([0., 0.5, 1.0])
    angles = infinite_sheds._ground_angle(
        x, ts['surface_tilt'], ts['gcr'])
    expected_angles = np.array([0., 5.866738789543952, 9.896090638982903])
    assert np.allclose(angles, expected_angles)


def test__ground_angle_zero_gcr():
    surface_tilt = 30.0
    x = np.array([0.0, 0.5, 1.0])
    angles = infinite_sheds._ground_angle(x, surface_tilt, 0)
    expected_angles = np.array([0, 0, 0])
    assert np.allclose(angles, expected_angles)


def test__vf_row_ground(test_system):
    ts, _, _ = test_system
    x = np.array([0., 0.5, 1.0])
    sqr3 = np.sqrt(3)
    vfs = infinite_sheds._vf_row_ground(
        x, ts['surface_tilt'], ts['gcr'])
    expected_vfs = np.array([
        0.5 * (1. - sqr3 / 2),
        0.5 * ((4 + sqr3 / 2) / np.sqrt(17 + 4 * sqr3) - sqr3 / 2),
        0.5 * ((4 + sqr3) / np.sqrt(20 + 8 * sqr3) - sqr3 / 2)])
    assert np.allclose(vfs, expected_vfs)


def test__vf_row_ground_integ(test_system):
    ts, _, _ = test_system
    gcr = ts['gcr']
    surface_tilt = ts['surface_tilt']
    f_x = np.array([0., 0.5, 1.0])
    shaded, noshade = infinite_sheds._vf_row_ground_integ(
        f_x, surface_tilt, gcr)

    def analytic(x, surface_tilt, gcr):
        c = cosd(surface_tilt)
        a = 1. / gcr
        dx = np.sqrt(a**2 + 2 * a * c * x + x**2)
        return c * dx - a * (c**2 - 1) * np.arctanh((a * c + x) / dx)

    expected_shade = 0.5 * (analytic(f_x, surface_tilt, gcr)
                            - analytic(0., surface_tilt, gcr)
                            - f_x * cosd(surface_tilt))
    expected_noshade = 0.5 * (analytic(1., surface_tilt, gcr)
                              - analytic(f_x, surface_tilt, gcr)
                              - (1. - f_x) * cosd(surface_tilt))
    assert np.allclose(shaded, expected_shade)
    assert np.allclose(noshade, expected_noshade)


def test__poa_ground_shadows():
    poa_ground, f_gnd_beam, df, vf_gnd_sky = (300., 0.5, 0.5, 0.2)
    result = infinite_sheds._poa_ground_shadows(
        poa_ground, f_gnd_beam, df, vf_gnd_sky)
    expected = 300. * (0.5 * 0.5 + 0.5 * 0.2)
    assert np.isclose(result, expected)
    # vector inputs
    poa_ground = np.array([300., 300.])
    f_gnd_beam = np.array([0.5, 0.5])
    df = np.array([0.5, 0.])
    vf_gnd_sky = np.array([0.2, 0.2])
    result = infinite_sheds._poa_ground_shadows(
        poa_ground, f_gnd_beam, df, vf_gnd_sky)
    expected_vec = np.array([expected, 300. * 0.5])
    assert np.allclose(result, expected_vec)


def test__shaded_fraction_floats():
    result = infinite_sheds._shaded_fraction(
        solar_zenith=60., solar_azimuth=180., surface_tilt=60.,
        surface_azimuth=180., gcr=1.0)
    assert np.isclose(result, 0.5)


def test__shaded_fraction_array():
    solar_zenith = np.array([0., 60., 90., 60.])
    solar_azimuth = np.array([180., 180., 180., 180.])
    surface_azimuth = np.array([180., 180., 180., 210.])
    surface_tilt = np.array([30., 60., 0., 30.])
    gcr = 1.0
    result = infinite_sheds._shaded_fraction(
        solar_zenith, solar_azimuth, surface_tilt, surface_azimuth, gcr)
    x = 0.75 + np.sqrt(3) / 2
    expected = np.array([0.0, 0.5, 0., (x - 1) / x])
    assert np.allclose(result, expected)


def test_get_irradiance_poa():
    # singleton inputs
    solar_zenith = 0.
    solar_azimuth = 180.
    surface_tilt = 0.
    surface_azimuth = 180.
    gcr = 0.5
    height = 1.
    pitch = 1
    ghi = 1000
    dhi = 300
    dni = 700
    albedo = 0
    iam = 1.0
    npoints = 100
    res = infinite_sheds.get_irradiance_poa(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni,
        albedo, iam=iam, npoints=npoints)
    expected_diffuse = np.array([300.])
    expected_direct = np.array([700.])
    expected_global = expected_diffuse + expected_direct
    assert np.isclose(res['poa_global'], expected_global)
    assert np.isclose(res['poa_diffuse'], expected_diffuse)
    assert np.isclose(res['poa_direct'], expected_direct)
    # vector inputs
    surface_tilt = np.array([0., 0., 0., 0.])
    height = 1.
    surface_azimuth = np.array([180., 180., 180., 180.])
    gcr = 0.5
    pitch = 1
    solar_zenith = np.array([0., 45., 45., 90.])
    solar_azimuth = np.array([180., 180., 135., 180.])
    expected_diffuse = np.array([300., 300., 300., 300.])
    expected_direct = np.array(
        [700., 350. * np.sqrt(2), 350. * np.sqrt(2), 0.])
    expected_global = expected_diffuse + expected_direct
    res = infinite_sheds.get_irradiance_poa(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni,
        albedo, iam=iam, npoints=npoints)
    assert np.allclose(res['poa_global'], expected_global)
    assert np.allclose(res['poa_diffuse'], expected_diffuse)
    assert np.allclose(res['poa_direct'], expected_direct)
    # series inputs
    surface_tilt = pd.Series(surface_tilt)
    surface_azimuth = pd.Series(data=surface_azimuth, index=surface_tilt.index)
    solar_zenith = pd.Series(solar_zenith, index=surface_tilt.index)
    solar_azimuth = pd.Series(data=solar_azimuth, index=surface_tilt.index)
    expected_diffuse = pd.Series(
        data=expected_diffuse, index=surface_tilt.index)
    expected_direct = pd.Series(
        data=expected_direct, index=surface_tilt.index)
    expected_global = expected_diffuse + expected_direct
    expected_global.name = 'poa_global'  # to match output Series
    res = infinite_sheds.get_irradiance_poa(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni,
        albedo, iam=iam, npoints=npoints)
    assert isinstance(res, pd.DataFrame)
    assert_series_equal(res['poa_global'], expected_global)
    assert all(k in res.columns for k in [
        'poa_global', 'poa_diffuse', 'poa_direct', 'poa_ground_diffuse',
        'poa_sky_diffuse'])


def test__backside_tilt():
    tilt = np.array([0., 30., 30., 180.])
    system_azimuth = np.array([180., 150., 270., 0.])
    back_tilt, back_az = infinite_sheds._backside(tilt, system_azimuth)
    assert np.allclose(back_tilt, np.array([180., 150., 150., 0.]))
    assert np.allclose(back_az, np.array([0., 330., 90., 180.]))


@pytest.mark.parametrize("vectorize", [True, False])
def test_get_irradiance(vectorize):
    # singleton inputs
    solar_zenith = 0.
    solar_azimuth = 180.
    surface_tilt = 0.
    surface_azimuth = 180.
    gcr = 0.5
    height = 1.
    pitch = 1.
    ghi = 1000.
    dhi = 300.
    dni = 700.
    albedo = 0.
    iam_front = 1.0
    iam_back = 1.0
    npoints = 100
    result = infinite_sheds.get_irradiance(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni, albedo, iam_front, iam_back,
        bifaciality=0.8, shade_factor=-0.02, transmission_factor=0,
        npoints=npoints, vectorize=vectorize)
    expected_front_diffuse = np.array([300.])
    expected_front_direct = np.array([700.])
    expected_front_global = expected_front_diffuse + expected_front_direct
    assert np.isclose(result['poa_front'], expected_front_global)
    assert np.isclose(result['poa_front_diffuse'], expected_front_diffuse)
    assert np.isclose(result['poa_front_direct'], expected_front_direct)
    assert np.isclose(result['poa_global'], result['poa_front'])
    # series inputs
    ghi = pd.Series([1000., 500., 500., np.nan])
    dhi = pd.Series([300., 500., 500., 500.], index=ghi.index)
    dni = pd.Series([700., 0., 0., 700.], index=ghi.index)
    solar_zenith = pd.Series([0., 0., 0., 135.], index=ghi.index)
    surface_tilt = pd.Series([0., 0., 90., 0.], index=ghi.index)
    result = infinite_sheds.get_irradiance(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni, albedo, iam_front, iam_back,
        bifaciality=0.8, shade_factor=-0.02, transmission_factor=0,
        npoints=npoints, vectorize=vectorize)
    result_front = infinite_sheds.get_irradiance_poa(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni,
        albedo, iam=iam_front, vectorize=vectorize)
    assert isinstance(result, pd.DataFrame)
    expected_poa_global = pd.Series(
        [1000., 500., result_front['poa_global'][2] * (1 + 0.8 * 0.98),
         np.nan], index=ghi.index, name='poa_global')
    assert_series_equal(result['poa_global'], expected_poa_global)


def test_get_irradiance_limiting_gcr():
    # test confirms that irradiance on widely spaced rows is approximately
    # the same as for a single row array
    solar_zenith = 0.
    solar_azimuth = 180.
    surface_tilt = 90.
    surface_azimuth = 180.
    gcr = 0.00001
    height = 1.
    pitch = 100.
    ghi = 1000.
    dhi = 300.
    dni = 700.
    albedo = 1.
    iam_front = 1.0
    iam_back = 1.0
    npoints = 100
    result = infinite_sheds.get_irradiance(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni, albedo, iam_front, iam_back,
        bifaciality=1., shade_factor=-0.00, transmission_factor=0.,
        npoints=npoints)
    expected_ground_diffuse = np.array([500.])
    expected_sky_diffuse = np.array([150.])
    expected_direct = np.array([0.])
    expected_diffuse = expected_ground_diffuse + expected_sky_diffuse
    expected_poa = expected_diffuse + expected_direct
    assert np.isclose(result['poa_front'], expected_poa, rtol=0.01)
    assert np.isclose(result['poa_front_diffuse'], expected_diffuse, rtol=0.01)
    assert np.isclose(result['poa_front_direct'], expected_direct)
    assert np.isclose(result['poa_front_sky_diffuse'], expected_sky_diffuse,
                      rtol=0.01)
    assert np.isclose(result['poa_front_ground_diffuse'],
                      expected_ground_diffuse, rtol=0.01)
    assert np.isclose(result['poa_front'], result['poa_back'])
    assert np.isclose(result['poa_front_diffuse'], result['poa_back_diffuse'])
    assert np.isclose(result['poa_front_direct'], result['poa_back_direct'])
    assert np.isclose(result['poa_front_sky_diffuse'],
                      result['poa_back_sky_diffuse'])
    assert np.isclose(result['poa_front_ground_diffuse'],
                      result['poa_back_ground_diffuse'])


def test_get_irradiance_with_haydavies():
    # singleton inputs
    solar_zenith = 0.
    solar_azimuth = 180.
    surface_tilt = 0.
    surface_azimuth = 180.
    gcr = 0.5
    height = 1.
    pitch = 1.
    ghi = 1000.
    dhi = 300.
    dni = 700.
    albedo = 0.
    dni_extra = 1413.
    model = 'haydavies'
    iam_front = 1.0
    iam_back = 1.0
    npoints = 100
    result = infinite_sheds.get_irradiance(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni, albedo, model, dni_extra,
        iam_front, iam_back, bifaciality=0.8, shade_factor=-0.02,
        transmission_factor=0, npoints=npoints)
    expected_front_diffuse = np.array([151.38])
    expected_front_direct = np.array([848.62])
    expected_front_global = expected_front_diffuse + expected_front_direct
    assert np.isclose(result['poa_front'], expected_front_global)
    assert np.isclose(result['poa_front_diffuse'], expected_front_diffuse)
    assert np.isclose(result['poa_front_direct'], expected_front_direct)
    assert np.isclose(result['poa_global'], result['poa_front'])
    # test for when dni_extra is not supplied
    with pytest.raises(ValueError, match='supply dni_extra for haydavies'):
        result = infinite_sheds.get_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            gcr, height, pitch, ghi, dhi, dni, albedo, model, None,
            iam_front, iam_back, bifaciality=0.8, shade_factor=-0.02,
            transmission_factor=0, npoints=npoints)
