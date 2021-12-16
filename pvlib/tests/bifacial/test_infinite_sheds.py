"""
test infinite sheds
"""

import numpy as np
from pvlib.bifacial import infinite_sheds
from pvlib.tools import cosd

import pytest


def test__gcr_prime():
    result = infinite_sheds._gcr_prime(gcr=0.5, height=1, surface_tilt=20,
                                       pitch=4)
    assert np.isclose(result, 1.2309511000407718)


@pytest.fixture
def test_system():
    syst = {'height': 1.,
            'pitch': 4.,
            'surface_tilt': 30}
    syst['gcr'] = 2.0 / syst['pitch']
    return syst


def test__ground_sky_angles(test_system):
    x = np.array([0.0, 0.5, 1.0])
    psi0, psi1 = infinite_sheds._ground_sky_angles(x, **test_system)
    expected_psi0 = np.array([150., 126.2060231, 75.])
    expected_psi1 = np.array([15., 20.10390936, 30.])
    assert np.allclose(psi0, expected_psi0)
    assert np.allclose(psi1, expected_psi1)


FZ0_LIMIT = 1.4619022000815438  # infinite_sheds.f_z0_limit(*ARGS)
# np.arctan2(GCR * np.sin(TILT_RAD), (1.0 - GCR * np.cos(TILT_RAD)))
PSI_TOP = 0.3120297392978313


def test__ground_sky_angles_prev(test_system):
    x = np.array([0.0, 1.0])
    psi0, psi1 = infinite_sheds._ground_sky_angles_prev(x, **test_system)
    expected_psi0 = np.array([75., 23.7939769])
    expected_psi1 = np.array([30., 180. - 23.7939769])
    assert np.allclose(psi0, expected_psi0)
    assert np.allclose(psi1, expected_psi1)


FZ1_LIMIT = 1.4619022000815427  # infinite_sheds.f_z1_limit(*ARGS)
# np.arctan2(GCR * np.sin(BACK_TILT_RAD), (1.0 - GCR * np.cos(BACK_TILT_RAD)))
PSI_TOP_BACK = 0.11582480672702507


def test__ground_sky_angles_next(test_system):
    x = np.array([0., 1.0])
    psi0, psi1 = infinite_sheds._ground_sky_angles_next(x, **test_system)
    expected_psi0 = np.array([180 - 9.8960906389, 150.])
    expected_psi1 = np.array([9.8960906389, 15.])
    assert np.allclose(psi0, expected_psi0)
    assert np.allclose(psi1, expected_psi1)


def test__sky_angle(test_system):
    x = np.array([0., 1.0])
    angle, tan_angle = infinite_sheds._sky_angle(
        test_system['gcr'], test_system['surface_tilt'], x)
    exp_tan_angle = np.array([1. / (4 - np.sqrt(3)), 0.])
    exp_angle = np.array([23.79397689, 0.])
    assert np.allclose(angle, exp_angle)
    assert np.allclose(tan_angle, exp_tan_angle)


def test__f_z0_limit(test_system):
    result = infinite_sheds._f_z0_limit(**test_system)
    expected = 1.0
    assert np.isclose(result, expected)


def test__f_z1_limit(test_system):
    result = infinite_sheds._f_z1_limit(**test_system)
    expected = 1.0
    assert np.isclose(result, expected)


def test__vf_sky():
    result = infinite_sheds._vf_sky(np.array([0.0, 30.0, 90.]),
                                    np.array([0.0, 30.0, 90.]))
    expected = np.array([1., np.sqrt(3) / 2, 0.])
    assert np.allclose(result, expected)


def test__vf_ground_sky(test_system):
    pts, vfs = infinite_sheds._vf_ground_sky(**test_system, npoints=3)
    expected_pts = np.linspace(0, 1, num=3)
    sqr3 = np.sqrt(3)
    # at f_z=0
    vf_0 = 0.5 * ((2 + sqr3) / np.sqrt(8 + 4 * sqr3) - sqr3 / 2)
    # at f_z=0.5
    vf_05_pre = 0.5 * ((3 - sqr3) / np.sqrt(13 - 6 * sqr3)
                       - np.sqrt(2 - sqr3) / 2)
    vf_05_mid = 0.5 * ((sqr3 + 1) / np.sqrt(5 + 2 * sqr3)
                       - (sqr3 - 1) / np.sqrt(5 - 2 * sqr3))
    vf_05_nex = 0.5 * ((2 + sqr3) / (2 * np.sqrt(2 + sqr3))
                       - (3 + sqr3) / np.sqrt(13 + 6 * sqr3))
    vf_05 = vf_05_pre + vf_05_mid + vf_05_nex
    # at f_z=1.0
    vf_1 = 0.5 * ((4 - 2 * sqr3) / 4 / np.sqrt(2 - sqr3) + sqr3 / 2)
    expected_vfs = np.array([vf_0 + vf_1, vf_05, vf_0 + vf_1])
    assert np.allclose(vfs, expected_vfs, rtol=0.1)  # middle point vf is off
    assert np.allclose(pts, expected_pts)


def test__vf_ground_sky_integ(test_system):
    vf_integ, pts, vf_pts = infinite_sheds._vf_ground_sky_integ(
        **test_system, npoints=3)
    expected_pts = np.linspace(0, 1, num=3)
    sqr3 = np.sqrt(3)
    # at f_z=0
    vf_0 = 0.5 * ((2 + sqr3) / np.sqrt(8 + 4 * sqr3) - sqr3 / 2)
    # at f_z=0.5
    vf_05_pre = 0.5 * ((3 - sqr3) / np.sqrt(13 - 6 * sqr3)
                       - np.sqrt(2 - sqr3) / 2)
    vf_05_mid = 0.5 * ((sqr3 + 1) / np.sqrt(5 + 2 * sqr3)
                       - (sqr3 - 1) / np.sqrt(5 - 2 * sqr3))
    vf_05_nex = 0.5 * ((2 + sqr3) / (2 * np.sqrt(2 + sqr3))
                       - (3 + sqr3) / np.sqrt(13 + 6 * sqr3))
    vf_05 = vf_05_pre + vf_05_mid + vf_05_nex
    # at f_z=1.0
    vf_1 = 0.5 * ((4 - 2 * sqr3) / 4 / np.sqrt(2 - sqr3) + sqr3 / 2)
    expected_vfs = np.array([vf_0 + vf_1, vf_05, vf_0 + vf_1])
    expected_vf_integ = np.trapz(expected_vfs, pts)
    assert np.allclose(pts, expected_pts)
    assert np.allclose(vf_pts, expected_vfs, rtol=0.1)
    assert np.isclose(vf_integ, expected_vf_integ, rtol=0.1)


def test__vf_row_sky_integ(test_system):
    gcr = test_system['gcr']
    surface_tilt = test_system['surface_tilt']
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
    x = np.array([0., 0.5, 1.0])
    angles, tan_angles = infinite_sheds._ground_angle(
        test_system['gcr'], test_system['surface_tilt'], x)
    expected_angles = np.array([0., 5.866738789543952, 9.896090638982903])
    expected_tan_angles = np.array([0., 0.5 / (4 + np.sqrt(3) / 2.),
                                    1. / (4 + np.sqrt(3))])
    assert np.allclose(angles, expected_angles)
    assert np.allclose(tan_angles, expected_tan_angles)


def test__vf_row_ground(test_system):
    x = np.array([0., 0.5, 1.0])
    sqr3 = np.sqrt(3)
    vfs = infinite_sheds._vf_row_ground(
        test_system['gcr'], test_system['surface_tilt'], x)
    expected_vfs = np.array([
        0.5 * (1. - sqr3 / 2),
        0.5 * ((4 + sqr3 / 2) / np.sqrt(17 + 4 * sqr3) - sqr3 / 2),
        0.5 * ((4 + sqr3) / np.sqrt(20 + 8 * sqr3) - sqr3 / 2)])
    assert np.allclose(vfs, expected_vfs)


def test__vf_row_ground_integ(test_system):
    gcr = test_system['gcr']
    surface_tilt = test_system['surface_tilt']
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


def test_get_irradiance_poa():
    # front side irradiance
    surface_tilt = np.array([0., 0., 0., 0.])
    height = 1.
    surface_azimuth = np.array([180., 180., 180., 180.])
    gcr = 0.5
    pitch = 1
    solar_zenith = np.array([0., 45., 45., 90.])
    solar_azimuth = np.array([180., 180., 135., 180.])
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
    expected_diffuse = np.array([300., 300., 300., 300.])
    expected_direct = np.array(
        [700., 350. * np.sqrt(2), 350. * np.sqrt(2), 0.])
    expected_global = expected_diffuse + expected_direct
    assert np.allclose(res['poa_global'], expected_global)
    assert np.allclose(res['poa_diffuse'], expected_diffuse)
    assert np.allclose(res['poa_direct'], expected_direct)
    # horizontal and elevated but gcr=1, so expect no rear irradiance
    height = 1
    gcr = 1.0
    res = infinite_sheds.get_irradiance_poa(
        solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, gcr, height, pitch, ghi, dhi, dni,
        albedo, iam, npoints, all_output)
    assert np.allclose(res['poa_global'], expected_global)
    assert np.allclose(res['poa_diffuse'], expected_diffuse)
    assert np.allclose(res['poa_direct'], expected_direct)
    
test_get_irradiance_poa()
