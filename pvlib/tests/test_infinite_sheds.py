"""
test infinite sheds
"""

import os
import numpy as np
import pandas as pd
import pvlib

BASEDIR = os.path.dirname(__file__)
PROJDIR = os.path.dirname(BASEDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TESTDATA = os.path.join(DATADIR, 'infinite_sheds.csv')

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
F_GND_SKY = TESTDATA['Fsky-gnd']
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


def test_solar_projection_tangent():
    tan_phi_f = pvlib.infinite_sheds.solar_projection_tangent(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, SYSAZ_RAD)
    backside_sysaz_rad = np.radians(BACKSIDE['sysaz'])
    tan_phi_b = pvlib.infinite_sheds.solar_projection_tangent(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, backside_sysaz_rad)
    assert np.allclose(tan_phi_f, -tan_phi_b)


def test_solar_projection():
    # frontside
    phi, tan_phi = pvlib.infinite_sheds.solar_projection(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_f)
    # backside
    phi, tan_phi = pvlib.infinite_sheds.solar_projection(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, BACK_SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_b)


def test_frontside_solar_projection_tangent():
    # frontside
    tan_phi = pvlib.infinite_sheds.solar_projection_tangent(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_f)
    # backside
    tan_phi = pvlib.infinite_sheds.solar_projection_tangent(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, BACK_SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_b)


def test_unshaded_ground_fraction():
    # frontside, same for both sides
    f_sky_gnd = pvlib.infinite_sheds.unshaded_ground_fraction(
        GCR, TILT_RAD, TESTDATA.tan_phi_f)
    assert np.allclose(f_sky_gnd, F_GND_SKY)
    # backside, should be the same as frontside
    f_sky_gnd = pvlib.infinite_sheds.unshaded_ground_fraction(
        GCR, BACK_TILT_RAD, TESTDATA.tan_phi_b)
    assert np.allclose(f_sky_gnd, F_GND_SKY)


ARGS = (GCR, HEIGHT, TILT_RAD, PITCH)
GCR_PRIME = pvlib.infinite_sheds._gcr_prime(*ARGS)


def calc_ground_sky_angles_at_edges(tilt_rad=TILT_RAD, gcr_prime=GCR_PRIME):
    back_tilt_rad = np.pi - tilt_rad
    psi_0_x0 = back_tilt_rad
    opposite_side = gcr_prime * np.sin(back_tilt_rad)
    adjacent_side = 1 - gcr_prime * np.cos(back_tilt_rad)
    # tan_psi_1_x0 = opposite_side / adjacent_side
    psi_1_x0 = np.arctan2(opposite_side, adjacent_side)
    opposite_side = gcr_prime * np.sin(tilt_rad)
    adjacent_side = 1 - gcr_prime * np.cos(tilt_rad)
    # tan_psi_0_x1 = opposite_side / adjacent_side
    psi_0_x1 = np.arctan2(opposite_side, adjacent_side)
    psi_1_x1 = tilt_rad
    return psi_0_x0, psi_1_x0, psi_0_x1, psi_1_x1


PSI_0_X0, PSI_1_X0, PSI_0_X1, PSI_1_X1 = calc_ground_sky_angles_at_edges()


def test_ground_sky_angles():
    # check limit at x=0, these are the same as the back edge of the row beyond
    assert np.allclose(
        pvlib.infinite_sheds.ground_sky_angles(0, *ARGS), (PSI_0_X0, PSI_1_X0))
    assert np.allclose(
        pvlib.infinite_sheds.ground_sky_angles(1, *ARGS), (PSI_0_X1, PSI_1_X1))


FZ0_LIMIT = pvlib.infinite_sheds.f_z0_limit(*ARGS)
PSI_TOP = np.arctan2(GCR * np.sin(TILT_RAD), (1.0 - GCR * np.cos(TILT_RAD)))


def test_ground_sky_angles_prev():
    if HEIGHT > 0:
        # check limit at z=0, these are the same as z=1 of the previous row
        assert np.allclose(
            pvlib.infinite_sheds.ground_sky_angles_prev(0, *ARGS),
            (PSI_0_X1, PSI_1_X1))
        # check limit at z=z0_limit, angles must sum to 180
        assert np.isclose(
            sum(pvlib.infinite_sheds.ground_sky_angles_prev(FZ0_LIMIT, *ARGS)),
            np.pi)
        # directly under panel, angle should be 90 straight upward!
        z_panel = HEIGHT / PITCH / np.tan(TILT_RAD)
        assert np.isclose(
            pvlib.infinite_sheds.ground_sky_angles_prev(z_panel, *ARGS)[1],
            np.pi / 2.)
    # angles must be the same as psi_top
    assert np.isclose(
        pvlib.infinite_sheds.ground_sky_angles_prev(FZ0_LIMIT, *ARGS)[0],
        PSI_TOP)


FZ1_LIMIT = pvlib.infinite_sheds.f_z1_limit(*ARGS)
PSI_TOP_BACK = np.arctan2(
    GCR * np.sin(BACK_TILT_RAD), (1.0 - GCR * np.cos(BACK_TILT_RAD)))


def test_ground_sky_angles_next():
    if HEIGHT > 0:
        # check limit at z=1, these are the same as z=0 of the next row beyond
        assert np.allclose(
            pvlib.infinite_sheds.ground_sky_angles_next(1, *ARGS),
            (PSI_0_X0, PSI_1_X0))
        # check limit at zprime=z1_limit, angles must sum to 180
        sum_angles_z1_limit = sum(
            pvlib.infinite_sheds.ground_sky_angles_next(1 - FZ1_LIMIT, *ARGS))
        assert np.isclose(sum_angles_z1_limit, np.pi)
        # directly under panel, angle should be 90 straight upward!
        z_panel = 1 + HEIGHT / PITCH / np.tan(TILT_RAD)
        assert np.isclose(
            pvlib.infinite_sheds.ground_sky_angles_next(z_panel, *ARGS)[0],
            np.pi / 2.)
    # angles must be the same as psi_top
    assert np.isclose(
        pvlib.infinite_sheds.ground_sky_angles_next(1 - FZ1_LIMIT, *ARGS)[1],
        PSI_TOP_BACK)


def test_bigz():
    _, height, tilt, pitch = ARGS
    psi_x0_bottom = 0
    bigz_x0 = pvlib.infinite_sheds._big_z(psi_x0_bottom, height, tilt, pitch)
    assert np.isinf(bigz_x0)


def test_diffuse_fraction():
    df = pvlib.infinite_sheds.diffuse_fraction(GHI, DHI)
    assert np.allclose(df, DF, equal_nan=True)


def test_poa_ground_sky():
    # front side
    poa_gnd_sky = pvlib.infinite_sheds.poa_ground_sky(
        TESTDATA.poa_ground_diffuse_f, F_GND_SKY, DF)
    # CSV file decimals are truncated
    assert np.allclose(
        poa_gnd_sky, FRONT_POA_GND_SKY, equal_nan=True, atol=1e-6)
    # backside
    poa_gnd_sky = pvlib.infinite_sheds.poa_ground_sky(
        TESTDATA.poa_ground_diffuse_b, F_GND_SKY, DF)
    assert np.allclose(poa_gnd_sky, BACK_POA_GND_SKY, equal_nan=True)


def test_shade_line():
    # front side
    f_x = pvlib.infinite_sheds.shade_line(GCR, TILT_RAD, TESTDATA.tan_phi_f)
    assert np.allclose(f_x, TESTDATA.Fx_f)
    # backside
    f_x = pvlib.infinite_sheds.shade_line(
        GCR, BACK_TILT_RAD, TESTDATA.tan_phi_b)
    assert np.allclose(f_x, TESTDATA.Fx_b)


def test_sky_angles():
    # frontside
    psi_top, tan_psi_top = pvlib.infinite_sheds.sky_angle(
        GCR, TILT_RAD, TESTDATA.Fx_f)
    assert np.allclose(psi_top, TESTDATA.psi_top_f)
    assert np.allclose(tan_psi_top, FRONT_TAN_PSI_TOP)
    # backside
    psi_top, tan_psi_top = pvlib.infinite_sheds.sky_angle(
        GCR, BACK_TILT_RAD, TESTDATA.Fx_b)
    assert np.allclose(psi_top, TESTDATA.psi_top_b)
    assert np.allclose(tan_psi_top, BACK_TAN_PSI_TOP)


def test_sky_angle_tangent():
    # frontside
    tan_psi_top = pvlib.infinite_sheds.sky_angle_tangent(
        GCR, TILT_RAD, TESTDATA.Fx_f)
    assert np.allclose(tan_psi_top, FRONT_TAN_PSI_TOP)
    # backside
    tan_psi_top = pvlib.infinite_sheds.sky_angle_tangent(
        GCR, BACK_TILT_RAD, TESTDATA.Fx_b)
    assert np.allclose(tan_psi_top, BACK_TAN_PSI_TOP)


def test_sky_angle_0_tangent():
    # frontside
    tan_psi_top = pvlib.infinite_sheds.sky_angle_0_tangent(GCR, TILT_RAD)
    assert np.allclose(tan_psi_top, TAN_PSI_TOP0_F)
    # backside
    tan_psi_top = pvlib.infinite_sheds.sky_angle_0_tangent(GCR, BACK_TILT_RAD)
    assert np.allclose(tan_psi_top, TAN_PSI_TOP0_B)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.ion()
    plt.plot(*pvlib.infinite_sheds.ground_sky_diffuse_view_factor(*ARGS))
    plt.title(
        'combined sky view factor, not including horizon and first/last row')
    plt.xlabel('fraction of pitch from front to back')
    plt.ylabel('view factor')
    plt.grid()
    plt.figure()
    fskyz = [
        pvlib.infinite_sheds.calc_fgndpv_zsky(x, *ARGS)
        for x in np.linspace(0, 1, 100)]
    fskyz, fgnd_pv = zip(*fskyz)
    plt.plot(fskyz/(1-np.cos(TILT_RAD))*2)
    plt.plot(fgnd_pv/(1-np.cos(TILT_RAD))*2)
    plt.grid()
    plt.title('frontside integrated ground reflected')
    plt.legend(('blocked', 'all sky'))
    plt.figure()
    fskyz = [
        pvlib.infinite_sheds.calc_fgndpv_zsky(
            x, GCR, HEIGHT, BACK_TILT_RAD, PITCH)
        for x in np.linspace(0, 1, 100)]
    fskyz, fgnd_pv = zip(*fskyz)
    plt.plot(fskyz/(1-np.cos(BACK_TILT_RAD))*2)
    plt.plot(fgnd_pv/(1-np.cos(BACK_TILT_RAD))*2)
    plt.grid()
    plt.title('backside integrated ground reflected')
    plt.legend(('blocked', 'all sky'))
