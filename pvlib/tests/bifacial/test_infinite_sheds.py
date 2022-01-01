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
            'axis_azimuth': None,
            'rotation': -30.}
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


def test__tilt_to_rotation():
    surface_tilt = np.array([0., 20., 90.])
    surface_azimuth = np.array([180., 90., 270.])
    result = infinite_sheds._tilt_to_rotation(
        surface_tilt, surface_azimuth, 180.)
    assert np.allclose(result, np.array([0., -20., 90.]))
    res = infinite_sheds._tilt_to_rotation(surface_tilt, 180., None)
    assert np.allclose(res, np.array([0., -20., -90.]))


def test__sky_angle(test_system):
    ts, _, _ = test_system
    x = np.array([0., 1.0])
    angle, tan_angle = infinite_sheds._sky_angle(
        ts['gcr'], ts['surface_tilt'], x)
    exp_tan_angle = np.array([1. / (4 - np.sqrt(3)), 0.])
    exp_angle = np.array([23.79397689, 0.])
    assert np.allclose(angle, exp_angle)
    assert np.allclose(tan_angle, exp_tan_angle)


def test__vf_ground_sky_integ(test_system):
    ts, pts, vfs_gnd_sky = test_system
    vf_integ, z, vfs = infinite_sheds._vf_ground_sky_integ(
        ts['gcr'], ts['height'], ts['surface_tilt'],
        ts['surface_azimuth'], ts['pitch'],
        ts['axis_azimuth'], max_rows=1, npoints=3)
    expected_vf_integ = np.trapz(vfs_gnd_sky, pts)
    assert np.allclose(z, pts)
    assert np.allclose(vfs, vfs_gnd_sky, rtol=0.1)
    assert np.isclose(vf_integ, expected_vf_integ, rtol=0.1)


def test__vf_row_sky_integ(test_system):
    ts, _, _ = test_system
    gcr = ts['gcr']
    surface_tilt = ts['surface_tilt']
    f_x = np.array([0., 0.5, 1.])
    shaded = []
    noshade = []
    for x in f_x:
        s, ns = infinite_sheds._vf_row_sky_integ(gcr, surface_tilt, x,
                                                 npoints=100)
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
        dhi, f_x, vf_shade_sky_integ, vf_noshade_sky_integ)
    expected_poa = np.array([np.nan, 0.0, 500 * (0.5 * 0.2 + 0.5 * 0.8)])
    assert np.allclose(poa, expected_poa, equal_nan=True)


def test__ground_angle(test_system):
    ts, _, _ = test_system
    x = np.array([0., 0.5, 1.0])
    angles, tan_angles = infinite_sheds._ground_angle(
        ts['gcr'], ts['surface_tilt'], x)
    expected_angles = np.array([0., 5.866738789543952, 9.896090638982903])
    expected_tan_angles = np.array([0., 0.5 / (4 + np.sqrt(3) / 2.),
                                    1. / (4 + np.sqrt(3))])
    assert np.allclose(angles, expected_angles)
    assert np.allclose(tan_angles, expected_tan_angles)


def test__vf_row_ground(test_system):
    ts, _, _ = test_system
    x = np.array([0., 0.5, 1.0])
    sqr3 = np.sqrt(3)
    vfs = infinite_sheds._vf_row_ground(
        ts['gcr'], ts['surface_tilt'], x)
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
    shaded = []
    noshade = []
    for x in f_x:
        s, ns = infinite_sheds._vf_row_ground_integ(
            gcr, surface_tilt, x)
        shaded.append(s)
        noshade.append(ns)

    def analytic(gcr, surface_tilt, x):
        c = cosd(surface_tilt)
        a = 1. / gcr
        dx = np.sqrt(a**2 + 2 * a * c * x + x**2)
        return c * dx - a * (c**2 - 1) * np.arctanh((a * c + x) / dx)

    shaded = np.array(shaded)
    noshade = np.array(noshade)
    expected_shade = 0.5 * (analytic(gcr, surface_tilt, f_x)
                            - analytic(gcr, surface_tilt, 0.)
                            - f_x * cosd(surface_tilt))
    expected_noshade = 0.5 * (analytic(gcr, surface_tilt, 1.)
                              - analytic(gcr, surface_tilt, f_x)
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
    df = np.array([0.5, np.inf])
    vf_gnd_sky = np.array([0.2, 0.2])
    result = infinite_sheds._poa_ground_shadows(
        poa_ground, f_gnd_beam, df, vf_gnd_sky)
    expected_vec = np.array([expected, 300. * 0.5])
    assert np.allclose(result, expected_vec)


def test_get_irradiance_poa():
    # singleton inputs
    surface_tilt = 0.
    height = 1.
    surface_azimuth = 180.
    gcr = 0.5
    pitch = 1
    solar_zenith = 0.
    solar_azimuth = 180.
    ghi = 1000
    dhi = 300
    dni = 700
    albedo = 0
    iam = 1.0
    npoints = 100
    all_output = True
    res = infinite_sheds.get_irradiance_poa(
        solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, gcr, height, pitch, ghi, dhi, dni,
        albedo, iam, npoints, all_output)
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
        solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, gcr, height, pitch, ghi, dhi, dni,
        albedo, iam, npoints, all_output)
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
        solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, gcr, height, pitch, ghi, dhi, dni,
        albedo, iam, npoints, all_output)
    assert isinstance(res, pd.DataFrame)
    assert_series_equal(res['poa_global'], expected_global)
