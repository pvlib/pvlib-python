"""
test infinite sheds
"""

import os
import numpy as np
import pandas as pd
from pvlib.bifacial import infinite_sheds
from pvlib.tools import cosd

import pytest
from ..conftest import DATA_DIR


TESTDATA = os.path.join(DATA_DIR, 'infinite_sheds.csv')

# location and irradiance
LAT, LON, TZ = 37.85, -122.25, -8  # global coordinates

# PV module orientation
#   tilt: positive = facing toward sun, negative = backward
#   system-azimuth: positive counter-clockwise from north
TILT, SYSAZ = 20.0, 250.0
GCR = 0.5  # ground coverage ratio
HEIGHT = 1  # height above ground
PITCH = 4  # row spacing

# IAM parameters
B0 = 0.05
MAXAOI = 85

# backside
BACKSIDE = {'tilt': 180.0 - TILT, 'sysaz': (180.0 + SYSAZ) % 360.0}

# TESTDATA
TESTDATA = pd.read_csv(TESTDATA, parse_dates=True)
GHI, DHI = TESTDATA.ghi, TESTDATA.dhi
# convert #DIV/0 to np.inf, 0/0 to NaN, then convert to float
DF = np.where(GHI > 0, TESTDATA.df, np.inf)
DF = np.where(DHI > 0, DF, np.nan).astype(np.float64)
TESTDATA.df = DF
F_GND_BEAM = TESTDATA['Fsky-gnd']
BACK_POA_GND_SKY = TESTDATA['POA_gnd-sky_b']
FRONT_POA_GND_SKY = TESTDATA['POA_gnd-sky_f']
BACK_TAN_PSI_TOP = np.tan(TESTDATA.psi_top_b)
FRONT_TAN_PSI_TOP = np.tan(TESTDATA.psi_top_f)

TAN_PSI_TOP0_F = np.tan(0.312029739)  # GCR*SIN(TILT_f)/(1-GCR*COS(TILT_f))
TAN_PSI_TOP0_B = np.tan(0.115824807)  # GCR*SIN(TILT_b)/(1-GCR*COS(TILT_b))
TAN_PSI_BOT1_F = np.tan(0.115824807)  # SIN(TILT_f) / (COS(TILT_f) + 1/GCR)
TAN_PSI_BOT1_B = np.tan(0.312029739)  # SIN(TILT_b) / (COS(TILT_b) + 1/GCR)

# radians
SOLAR_ZENITH_RAD = np.radians(TESTDATA.apparent_zenith)
SOLAR_AZIMUTH_RAD = np.radians(TESTDATA.azimuth)
SYSAZ_RAD = np.radians(SYSAZ)
BACK_SYSAZ_RAD = np.radians(BACKSIDE['sysaz'])
TILT_RAD = np.radians(TILT)
BACK_TILT_RAD = np.radians(BACKSIDE['tilt'])

ARGS = (GCR, HEIGHT, TILT, PITCH)


gcr, height, surface_tilt, pitch = ARGS


def test__gcr_prime():
    result = infinite_sheds._gcr_prime(gcr=0.5, height=1, surface_tilt=20,
                                       pitch=4)
    assert np.isclose(result, 1.2309511000407718)


# calculated ground-sky angles at panel edges
# gcr_prime = infinite_sheds._gcr_prime(*ARGS)
# back_tilt_rad = np.pi - tilt_rad
# psi_0_x0 = back_tilt_rad
# psi_1_x0 = np.arctan2(
#     gcr_prime * np.sin(back_tilt_rad), 1 - gcr_prime * np.cos(back_tilt_rad))
PSI_0_X0, PSI_1_X0 = 2.792526803190927, 0.19278450775754705
# psi_0_x1 = np.arctan2(
#     gcr_prime * np.sin(tilt_rad), 1 - gcr_prime * np.cos(tilt_rad))
# psi_1_x1 = tilt_rad
PSI_0_X1, PSI_1_X1 = 1.9271427336418656, 0.3490658503988659


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


def test__vf_row_ground(gcr, surface_tilt, x):
    x = np.array([0., 0.5, 1.0])
    sqr3 = np.sqrt(3)
    vfs = infinite_sheds._vf_row_ground(
        test_system['gcr'], test_system['surface_tilt'], x)
    expected_vfs = np.array([
        1., 0.5 * ((4 + sqr3 / 2) / np.sqrt(17 + 4 * sqr3) - sqr3 / 2),
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
        return  c * dx - a * (c**2 - 1) * np.arctanh((a * c + x) / dx)

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


VF_GND_SKY = 0.5184093800689326
FZ_SKY = np.array([
    0.37395996, 0.37985504, 0.38617593, 0.39294621, 0.40008092,
    0.40760977, 0.41546240, 0.42363368, 0.43209234, 0.44079809,
    0.44974664, 0.45887908, 0.46819346, 0.47763848, 0.48719477,
    0.49682853, 0.50650894, 0.51620703, 0.52589332, 0.53553353,
    0.54510461, 0.55457309, 0.56391157, 0.57309977, 0.58209408,
    0.59089589, 0.59944489, 0.60775144, 0.61577071, 0.62348812,
    0.63089212, 0.63793327, 0.64463809, 0.65092556, 0.65683590,
    0.66231217, 0.66735168, 0.67194521, 0.67603859, 0.67967459,
    0.68274901, 0.68532628, 0.68733124, 0.68876957, 0.68962743,
    0.68984316, 0.68953528, 0.68867052, 0.68716547, 0.68492226,
    0.68196156, 0.67826724, 0.67378014, 0.66857561, 0.66252116,
    0.65574207, 0.64814205, 0.63978082, 0.63066636, 0.62078878,
    0.61025517, 0.59900195, 0.58719184, 0.57481610, 0.56199241,
    0.54879229, 0.53530254, 0.52163859, 0.50789053, 0.49417189,
    0.48059555, 0.46725727, 0.45425705, 0.44170686, 0.42964414,
    0.41822953, 0.40742909, 0.39738731, 0.38808373, 0.37957663,
    0.37191014, 0.36503340, 0.35906878, 0.35388625, 0.34959679,
    0.34610681, 0.34343945, 0.34158818, 0.34047992, 0.34019127,
    0.34058737, 0.34174947, 0.34357674, 0.34608321, 0.34924749,
    0.35300886, 0.35741583, 0.36235918, 0.36789933, 0.37394838])


def test__poa_ground_sky():
    # front side
    poa_gnd_sky_f = infinite_sheds._poa_ground_sky(
        TESTDATA.poa_ground_diffuse_f, F_GND_BEAM, DF, 1.0)
    # CSV file decimals are truncated
    assert np.allclose(
        poa_gnd_sky_f, FRONT_POA_GND_SKY, equal_nan=True, atol=1e-6)
    # backside
    poa_gnd_sky_b = infinite_sheds._poa_ground_sky(
        TESTDATA.poa_ground_diffuse_b, F_GND_BEAM, DF, 1.0)
    assert np.allclose(poa_gnd_sky_b, BACK_POA_GND_SKY, equal_nan=True)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.ion()
    plt.plot(*infinite_sheds.ground_sky_diffuse_view_factor(*ARGS))
    plt.title(
        'combined sky view factor, not including horizon and first/last row')
    plt.xlabel('fraction of pitch from front to back')
    plt.ylabel('view factor')
    plt.grid()
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    fx = np.linspace(0, 1, 100)
    fskyz = [infinite_sheds.calc_fgndpv_zsky(x, *ARGS) for x in fx]
    fskyz, fgnd_pv = zip(*fskyz)
    ax[0].plot(fx, fskyz/(1-np.cos(TILT_RAD))*2)
    ax[0].plot(fx, fgnd_pv/(1-np.cos(TILT_RAD))*2)
    ax[0].grid()
    ax[0].set_title('frontside integrated ground reflected')
    ax[0].set_xlabel('fraction of PV surface from bottom ($F_x$)')
    ax[0].set_ylabel('adjustment to $\\frac{1-\\cos(\\beta)}{2}$')
    ax[0].legend(('blocked', 'all sky'))
    fskyz = [
        infinite_sheds.calc_fgndpv_zsky(
            x, GCR, HEIGHT, BACK_TILT_RAD, PITCH) for x in fx]
    fskyz, fgnd_pv = zip(*fskyz)
    ax[1].plot(fx, fskyz/(1-np.cos(BACK_TILT_RAD))*2)
    ax[1].plot(fx, fgnd_pv/(1-np.cos(BACK_TILT_RAD))*2)
    ax[1].grid()
    ax[1].set_title('backside integrated ground reflected')
    ax[1].set_xlabel('fraction of PV surface from bottom ($F_x$)')
    ax[1].set_ylabel('adjustment to $\\frac{1-\\cos(\\beta)}{2}$')
    ax[1].legend(('blocked', 'all sky'))
    plt.tight_layout()
