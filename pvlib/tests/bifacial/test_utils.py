"""
test bifical.utils
"""
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
    # view factors from 3 points on the ground between rows to the sky
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
    vfs_ground_sky = np.array([[vf_0], [vf_1], [vf_2]])
    return syst, pts, vfs_ground_sky


def test__solar_projection_tangent():
    tan_phi_f = utils._solar_projection_tangent(
        30, 150, 180)
    tan_phi_b = utils._solar_projection_tangent(
        30, 150, 0)
    assert np.allclose(tan_phi_f, 0.5)
    assert np.allclose(tan_phi_b, -0.5)
    assert np.allclose(tan_phi_f, -tan_phi_b)


@pytest.mark.parametrize(
    "gcr,surface_tilt,surface_azimuth,solar_zenith,solar_azimuth,expected",
    [(0.5, 0., 180., 0., 180., 0.5),
     (1.0, 0., 180., 0., 180., 0.0),
     (1.0, 90., 180., 0., 180., 1.0),
     (0.5, 45., 180., 45., 270., 1.0 - np.sqrt(2) / 4),
     (0.5, 45., 180., 90., 180., 0.),
     (np.sqrt(2) / 2, 45, 180, 0, 180, 0.5),
     (np.sqrt(2) / 2, 45, 180, 45, 180, 0.0),
     (np.sqrt(2) / 2, 45, 180, 45, 90, 0.5),
     (np.sqrt(2) / 2, 45, 180, 45, 0, 1.0),
     (np.sqrt(2) / 2, 45, 180, 45, 135, 0.5 * (1 - np.sqrt(2) / 2)),
     ])
def test__unshaded_ground_fraction(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, gcr,
        expected):
    # frontside, same for both sides
    f_sky_beam_f = utils._unshaded_ground_fraction(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, gcr)
    assert np.allclose(f_sky_beam_f, expected)
    # backside, should be the same as frontside
    f_sky_beam_b = utils._unshaded_ground_fraction(
        180. - surface_tilt, surface_azimuth - 180., solar_zenith,
        solar_azimuth, gcr)
    assert np.allclose(f_sky_beam_b, expected)


def test__vf_ground_sky_2d(test_system_fixed_tilt):
    # vector input
    ts, pts, vfs_gnd_sky = test_system_fixed_tilt
    vfs = utils._vf_ground_sky_2d(pts, ts['rotation'], ts['gcr'],
                                  ts['pitch'], ts['height'], max_rows=1)
    assert np.allclose(vfs, vfs_gnd_sky, rtol=0.1)  # middle point vf is off
    # test with singleton x
    vf = utils._vf_ground_sky_2d(pts[0], ts['rotation'], ts['gcr'],
                                 ts['pitch'], ts['height'], max_rows=1)
    assert np.isclose(vf, vfs_gnd_sky[0])
