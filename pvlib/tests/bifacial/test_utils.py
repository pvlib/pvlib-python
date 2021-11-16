"""
test infinite sheds
"""

import os
import numpy as np
import pytest
from pvlib.bifacial import utils

BASEDIR = os.path.dirname(__file__)
PROJDIR = os.path.dirname(BASEDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TESTDATA = os.path.join(DATADIR, 'infinite_sheds.csv')


def test_solar_projection_tangent():
    tan_phi_f = utils.solar_projection_tangent(
        30, 150, 180)
    tan_phi_b = utils.solar_projection_tangent(
        30, 150, 0)
    assert np.allclose(tan_phi_f, 0.5)
    assert np.allclose(tan_phi_b, 0.5)
    assert np.allclose(tan_phi_f, -tan_phi_b)


@pytest.mark.parametrize(
    "gcr,surface_tilt,surface_azimuth,solar_zenith,solar_azimuth,expected",
    [(np.sqrt(2) / 2, 45, 180, 0, 180, 0.5),
     (np.sqrt(2) / 2, 45, 180, 45, 180, 0.0),
     (np.sqrt(2) / 2, 45, 180, 45, 90, 0.5),
     (np.sqrt(2) / 2, 45, 180, 45, 0, 1.0),
     (np.sqrt(2) / 2, 45, 180, 45, 135, 0.5 * (1 - np.sqrt(2) / 2)),
     ]
    )
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
