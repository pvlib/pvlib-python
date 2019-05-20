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
# convert #DIV/0 to np.inf, 0/0 to NaN, then convert to float
DF = np.where(GHI > 0, TESTDATA.df, np.inf)
DF = np.where(DHI > 0, DF, np.nan).astype(np.float64)
TESTDATA.df = DF
F_GND_SKY = TESTDATA['Fsky-gnd']
BACK_POA_GND_SKY = TESTDATA['POA_gnd-sky_b']
FRONT_POA_GND_SKY = TESTDATA['POA_gnd-sky_f']

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


def test_ground_illumination():
    # frontside, same for both sides
    f_sky_gnd = pvlib.infinite_sheds.ground_illumination(
        GCR, TILT_RAD, TESTDATA.tan_phi_f)
    assert np.allclose(f_sky_gnd, F_GND_SKY)
    # backside, should be the same as frontside
    f_sky_gnd = pvlib.infinite_sheds.ground_illumination(
        GCR, BACK_TILT_RAD, TESTDATA.tan_phi_b)
    assert np.allclose(f_sky_gnd, F_GND_SKY)


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
