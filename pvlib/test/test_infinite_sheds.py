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
TESTDATA = pd.read_csv(TESTDATA, parse_dates=True)

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


def test_solar_projection_tangent():
    solar_zenith_rad = np.radians(TESTDATA.apparent_zenith)
    solar_azimuth_rad = np.radians(TESTDATA.azimuth)
    system_azimuth_rad = np.radians(SYSAZ)
    tan_phi_f = pvlib.infinite_sheds.solar_projection_tangent(
        solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad)
    backside_sysaz_rad = np.radians(BACKSIDE['sysaz'])
    tan_phi_b = pvlib.infinite_sheds.solar_projection_tangent(
        solar_zenith_rad, solar_azimuth_rad, backside_sysaz_rad)
    assert np.allclose(tan_phi_f, -tan_phi_b)


def test_frontside_solar_projection():
    solar_zenith_rad = np.radians(TESTDATA.apparent_zenith)
    solar_azimuth_rad = np.radians(TESTDATA.azimuth)
    system_azimuth_rad = np.radians(SYSAZ)
    phi, tan_phi = pvlib.infinite_sheds.solar_projection(
        solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_f)


def test_frontside_solar_projection_tangent():
    solar_zenith_rad = np.radians(TESTDATA.apparent_zenith)
    solar_azimuth_rad = np.radians(TESTDATA.azimuth)
    system_azimuth_rad = np.radians(SYSAZ)
    tan_phi = pvlib.infinite_sheds.solar_projection_tangent(
        solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_f)


def test_backside_solar_projection():
    solar_zenith_rad = np.radians(TESTDATA.apparent_zenith)
    solar_azimuth_rad = np.radians(TESTDATA.azimuth)
    system_azimuth_rad = np.radians(BACKSIDE['sysaz'])
    phi, tan_phi = pvlib.infinite_sheds.solar_projection(
        solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_b)


def test_backside_solar_projection_tangent():
    solar_zenith_rad = np.radians(TESTDATA.apparent_zenith)
    solar_azimuth_rad = np.radians(TESTDATA.azimuth)
    system_azimuth_rad = np.radians(BACKSIDE['sysaz'])
    tan_phi = pvlib.infinite_sheds.solar_projection_tangent(
        solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad)
    assert np.allclose(tan_phi, TESTDATA.tan_phi_b)