"""
test infinite sheds
"""

import os
import numpy as np
import pytest
from pvlib.bifacial import utils


@pytest.fixture
def test_system_fixed_tilt():
    syst = {'height': 1.0,
            'pitch': 2.,
            'surface_tilt': 30.,
            'surface_azimuth': 180.,
            'axis_azimuth': None,
            'rotation': -30.}
    syst['gcr'] = 1.0 / syst['pitch']
    return syst


def test_solar_projection_tangent():
    tan_phi_f = utils.solar_projection_tangent(
        30, 150, 180)
    tan_phi_b = utils.solar_projection_tangent(
        30, 150, 0)
    assert np.allclose(tan_phi_f, 0.5)
    assert np.allclose(tan_phi_b, -0.5)
    assert np.allclose(tan_phi_f, -tan_phi_b)


@pytest.mark.parametrize(
    "gcr,surface_tilt,surface_azimuth,solar_zenith,solar_azimuth,expected",
    [(np.sqrt(2) / 2, 45, 180, 0, 180, 0.5),
     (np.sqrt(2) / 2, 45, 180, 45, 180, 0.0),
     (np.sqrt(2) / 2, 45, 180, 45, 90, 0.5),
     (np.sqrt(2) / 2, 45, 180, 45, 0, 1.0),
     (np.sqrt(2) / 2, 45, 180, 45, 135, 0.5 * (1 - np.sqrt(2) / 2)),
     ])
def test_unshaded_ground_fraction(
        gcr, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        expected):
    # frontside, same for both sides
    f_sky_beam_f = utils.unshaded_ground_fraction(
        gcr, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    assert np.allclose(f_sky_beam_f, expected)
    # backside, should be the same as frontside
    f_sky_beam_b = utils.unshaded_ground_fraction(
        gcr, surface_tilt + 90, surface_azimuth - 180, solar_zenith,
        solar_azimuth)
    assert np.allclose(f_sky_beam_b, expected)


def test__vf_ground_sky(test_system_fixed_tilt):
    # singleton inputs
    pts = np.linspace(0, 1, num=3)
    vfs, _ = utils.vf_ground_sky_2d(
        pts, test_system_fixed_tilt['rotation'], test_system_fixed_tilt['gcr'], 
        test_system_fixed_tilt['pitch'], test_system_fixed_tilt['height'],
        max_rows=1)
    sqr3 = np.sqrt(3) / 4
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
    c22 = (-2 - sqr3) / np.sqrt(1.75**2 + (2 + sqr3)**2)  # right edge row 0
    c23 = (0 - sqr3) / np.sqrt(1.25**2 + (0 - sqr3)**2)  # right edge row 1 
    vf_2 = 0.5 * (c23 - c22 + c21 - c20)  # vf at point 1
    expected_vfs = np.array([vf_0, vf_1, vf_2])
    assert np.allclose(vfs, expected_vfs, rtol=0.1)  # middle point vf is off
