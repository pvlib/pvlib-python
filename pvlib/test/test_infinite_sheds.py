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

# IAM parameters
B0 = 0.05
MAXAOI = 85

# backside
BACKSIDE = {'tilt': 180.0 - TILT, 'sysaz': (180.0 + SYSAZ) % 360.0}

# TESTDATA
TESTDATA = pd.read_csv(TESTDATA, parse_dates=True)
GHI, DHI = TESTDATA.ghi, TESTDATA.dhi
DF = np.where(DHI > 0, TESTDATA.df, np.nan).astype(np.float64)

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


def test_frontside_solar_projection():
    phi, tan_phi = pvlib.infinite_sheds.solar_projection(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_f)


def test_frontside_solar_projection_tangent():
    tan_phi = pvlib.infinite_sheds.solar_projection_tangent(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_f)


def test_backside_solar_projection():
    phi, tan_phi = pvlib.infinite_sheds.solar_projection(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, BACK_SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_b)


def test_backside_solar_projection_tangent():
    tan_phi = pvlib.infinite_sheds.solar_projection_tangent(
        SOLAR_ZENITH_RAD, SOLAR_AZIMUTH_RAD, BACK_SYSAZ_RAD)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_b)


def test_ground_illumination():
    f_sky_gnd = pvlib.infinite_sheds.ground_illumination(
        GCR, TILT_RAD, TESTDATA.tan_phi_f)
    assert np.allclose(f_sky_gnd, TESTDATA['Fsky-gnd'])


def test_backside_ground_illumination():
    f_sky_gnd = pvlib.infinite_sheds.ground_illumination(
        GCR, BACK_TILT_RAD, TESTDATA.tan_phi_b)
    assert np.allclose(f_sky_gnd, TESTDATA['Fsky-gnd'])


def test_diffuse_fraction():
    df = pvlib.infinite_sheds.diffuse_fraction(GHI, DHI)
    assert np.allclose(df, DF, equal_nan=True)


def test_frontside_poa_ground_sky():
    poa_gnd_sky = pvlib.infinite_sheds.poa_ground_sky(
        TESTDATA['poa_ground_diffuse_f'], TESTDATA['Fsky-gnd'], DF)
    # CSV file decimals are truncated
    assert np.allclose(
        poa_gnd_sky, TESTDATA['POA_gnd-sky_f'], equal_nan=True, atol=1e-6)


def test_backside_poa_ground_sky():
    poa_gnd_sky = pvlib.infinite_sheds.poa_ground_sky(
        TESTDATA['poa_ground_diffuse_b'], TESTDATA['Fsky-gnd'], DF)
    assert np.allclose(poa_gnd_sky, TESTDATA['POA_gnd-sky_b'], equal_nan=True)