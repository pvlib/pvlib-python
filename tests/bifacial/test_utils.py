"""
test bifical.utils
"""
import numpy as np
import pytest
from pvlib.bifacial import utils
from pvlib.shading import masking_angle, ground_angle
from pvlib.tools import cosd
from scipy.integrate import trapezoid

from pvlib._deprecation import pvlibDeprecationWarning


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
    "gcr,surface_tilt,phi,expected",
    [(0.5, 0., 0., 0.5),
     (1.0, 0., 0., 0.0),
     (1.0, 90., 0., 1.0),
     (0.5, 45., 0., 1.0 - np.sqrt(2) / 4),
     (0.5, 45., 90., 0.),
     (np.sqrt(2) / 2, 45, 0, 0.5),
     (np.sqrt(2) / 2, 45, 45, 0.0),
     (np.sqrt(2) / 2, 45, 0, 0.5),
     (np.sqrt(2) / 2, 45, -45, 1.0),
     (np.sqrt(2) / 2, 45, 35.264389682754654, 0.5 * (1 - np.sqrt(2) / 2)),
     ])
def test__unshaded_ground_fraction(surface_tilt, phi, gcr, expected):
    # frontside, same for both sides
    f_sky_beam_f = utils._unshaded_ground_fraction(surface_tilt, phi, gcr)
    assert np.allclose(f_sky_beam_f, expected)
    # backside, should be the same as frontside
    f_sky_beam_b = utils._unshaded_ground_fraction(surface_tilt - 180., phi,
                                                   gcr)
    assert np.allclose(f_sky_beam_b, expected)


def test__unshaded_ground_fraction_horizontal():
    # test that zero angles don't mess things up. specifically, check
    # for continuity with small positive and negative angles
    params = dict(gcr=0.5, pitch=4, height=2.5,
                  g0=[0, 0.25, 0.5, 0.75], g1=[0.25, 0.5, 0.75, 1])
    zero = utils._unshaded_ground_fraction(0.0, 0.0, **params)
    pos = utils._unshaded_ground_fraction(0.01, 0.01, **params)
    neg = utils._unshaded_ground_fraction(-0.01, -0.01, **params)
    np.testing.assert_allclose(pos, neg, atol=0.01)
    np.testing.assert_allclose(pos, zero, atol=0.01)
    np.testing.assert_allclose(neg, zero, atol=0.01)


def test_vf_ground_sky_2d(test_system_fixed_tilt):
    # vector input
    ts, pts, vfs_gnd_sky = test_system_fixed_tilt
    vfs = utils.vf_ground_sky_2d(ts['rotation'], ts['gcr'], pts,
                                 ts['pitch'], ts['height'], max_rows=1)
    assert np.allclose(vfs, vfs_gnd_sky, rtol=0.1)  # middle point vf is off
    # test with singleton x
    vf = utils.vf_ground_sky_2d(ts['rotation'], ts['gcr'], pts[0],
                                ts['pitch'], ts['height'], max_rows=1)
    assert np.isclose(vf, vfs_gnd_sky[0])


@pytest.mark.parametrize("vectorize", [True, False])
def test_vf_ground_sky_2d_integ(test_system_fixed_tilt, vectorize):
    ts, pts, vfs_gnd_sky = test_system_fixed_tilt
    # pass rotation here since max_rows=1 for the hand-solved case in
    # the fixture test_system, which means the ground-to-sky view factor
    # isn't summed over enough rows for symmetry to hold.
    match = '`npoints` and `vectorize` parameters have no effect'
    with pytest.warns(pvlibDeprecationWarning, match=match):
        vf_integ = utils.vf_ground_sky_2d_integ(
            ts['rotation'], ts['gcr'], ts['height'], ts['pitch'],
            max_rows=1, npoints=3, vectorize=vectorize)
    expected_vf_integ = trapezoid(vfs_gnd_sky, pts, axis=0)
    assert np.isclose(vf_integ, expected_vf_integ, rtol=0.1)


def test_vf_row_sky_2d(test_system_fixed_tilt):
    ts, _, _ = test_system_fixed_tilt
    # with float input, fx at top of row
    vf = utils.vf_row_sky_2d(ts['surface_tilt'], ts['gcr'], 1.)
    expected = 0.5 * (1 + cosd(ts['surface_tilt']))
    assert np.isclose(vf, expected)
    # with array input
    fx = np.array([0., 0.5, 1.])
    vf = utils.vf_row_sky_2d(ts['surface_tilt'], ts['gcr'], fx)
    phi = masking_angle(ts['surface_tilt'], ts['gcr'], fx)
    expected = 0.5 * (1 + cosd(ts['surface_tilt'] + phi))
    assert np.allclose(vf, expected)


def test_vf_row_sky_2d_integ(test_system_fixed_tilt):
    ts, _, _ = test_system_fixed_tilt
    # with float input, check end position
    with np.errstate(invalid='ignore'):
        vf = utils.vf_row_sky_2d_integ(ts['surface_tilt'], ts['gcr'], 1., 1.)
    expected = utils.vf_row_sky_2d(ts['surface_tilt'], ts['gcr'], 1.)
    assert np.isclose(vf, expected)
    # with array input
    fx0 = np.array([0., 0.5])
    fx1 = np.array([0., 0.8])
    with np.errstate(invalid='ignore'):
        vf = utils.vf_row_sky_2d_integ(ts['surface_tilt'], ts['gcr'], fx0, fx1)
    phi = masking_angle(ts['surface_tilt'], ts['gcr'], fx0[0])
    y0 = 0.5 * (1 + cosd(ts['surface_tilt'] + phi))
    x = np.arange(fx0[1], fx1[1], 1e-4)
    phi_y = masking_angle(ts['surface_tilt'], ts['gcr'], x)
    y = 0.5 * (1 + cosd(ts['surface_tilt'] + phi_y))
    y1 = trapezoid(y, x) / (fx1[1] - fx0[1])
    expected = np.array([y0, y1])
    assert np.allclose(vf, expected, rtol=1e-3)
    # with defaults (0, 1)
    vf = utils.vf_row_sky_2d_integ(ts['surface_tilt'], ts['gcr'])
    x = np.arange(0, 1, 1e-4)
    phi_y = masking_angle(ts['surface_tilt'], ts['gcr'], x)
    y = 0.5 * (1 + cosd(ts['surface_tilt'] + phi_y))
    y1 = trapezoid(y, x) / (1 - 0)
    assert np.allclose(vf, y1, rtol=1e-3)


def test_vf_ground_sky_2d_integ_horizontal_large_max_rows():
    # with large max_rows, rows far out towards the horizon are considered.
    # obstructed string lengths are then calculated using angles very close
    # to zero.  however, numerical roundoff requires some amount of slop being
    # allowed in the comparison.  this case previously failed when the slop
    # allowance was too large and prevented accurate resolution of those
    # far-away rows.
    inputs = dict(tracker_rotation=0, gcr=0.518, height=1.5, pitch=3.5,
                  g0=0, g1=0.05, max_rows=1000)
    vf_gnd_sky = utils.vf_ground_sky_2d_integ(**inputs)
    assert np.max(vf_gnd_sky) < 1


def test_vf_row_ground_2d(test_system_fixed_tilt):
    ts, _, _ = test_system_fixed_tilt
    # with float input, fx at bottom of row
    vf = utils.vf_row_ground_2d(ts['surface_tilt'], ts['gcr'], 0.)
    expected = 0.5 * (1. - cosd(ts['surface_tilt']))
    assert np.isclose(vf, expected)
    # with array input
    fx = np.array([0., 0.5, 1.0])
    vf = utils.vf_row_ground_2d(ts['surface_tilt'], ts['gcr'], fx)
    phi = ground_angle(ts['surface_tilt'], ts['gcr'], fx)
    expected = 0.5 * (1 - cosd(phi - ts['surface_tilt']))
    assert np.allclose(vf, expected)


def test_vf_row_ground_2d_integ(test_system_fixed_tilt):
    ts, _, _ = test_system_fixed_tilt
    # with float input, check end position
    with np.errstate(invalid='ignore'):
        vf = utils.vf_row_ground_2d_integ(ts['surface_tilt'], ts['gcr'],
                                          ts['height'], ts['pitch'],
                                          x0=0., x1=0.00001, max_rows=1000)
    expected = utils.vf_row_ground_2d(ts['surface_tilt'], ts['gcr'], 0.)
    assert np.isclose(vf, expected, atol=1e-4)
    # with array input
    fx0 = np.array([0., 0.5])
    fx1 = np.array([0.00001, 0.8])
    with np.errstate(invalid='ignore'):
        vf = utils.vf_row_ground_2d_integ(ts['surface_tilt'], ts['gcr'],
                                          ts['height'], ts['pitch'],
                                          x0=fx0, x1=fx1, max_rows=1000)
    phi = ground_angle(ts['surface_tilt'], ts['gcr'], fx0[0])
    y0 = 0.5 * (1 - cosd(phi - ts['surface_tilt']))
    x = np.arange(fx0[1], fx1[1], 1e-4)
    phi_y = ground_angle(ts['surface_tilt'], ts['gcr'], x)
    y = 0.5 * (1 - cosd(phi_y - ts['surface_tilt']))
    y1 = trapezoid(y, x) / (fx1[1] - fx0[1])
    expected = np.array([y0, y1])
    assert np.allclose(vf, expected, rtol=1e-2)
    # with defaults (0, 1)
    vf = utils.vf_row_ground_2d_integ(ts['surface_tilt'], ts['gcr'],
                                      ts['height'], ts['pitch'], x0=0, x1=1,
                                      max_rows=1000)
    x = np.arange(0, 1, 1e-4)
    phi_y = ground_angle(ts['surface_tilt'], ts['gcr'], x)
    y = 0.5 * (1 - cosd(phi_y - ts['surface_tilt']))
    y1 = trapezoid(y, x) / (1 - 0)
    assert np.allclose(vf, y1, rtol=1e-2)


def test_vf_row_ground_2d_integ_upsidedown():
    # horizontal, rear-side
    inputs = {'surface_tilt': np.array([-180]),
              'gcr': 0.6, 'height': 1.5, 'pitch': 3.5,
              'x0': np.array([0.]), 'x1': np.array([1.]),
              'g0': np.array([0., 0.5]),
              'g1': np.array([0.5, 1.]),
              'max_rows': 7}
    vf_row_ground = utils.vf_row_ground_2d_integ(**inputs)
    assert all(vf_row_ground > 1e-3)
    inputs['surface_tilt'] = +180
    vf_row_ground = utils.vf_row_ground_2d_integ(**inputs)
    assert all(vf_row_ground > 1e-3)
