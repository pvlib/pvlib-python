import itertools

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose

from pvlib import atmosphere
from pvlib import solarposition

latitude, longitude, tz, altitude = 32.2, -111, 'US/Arizona', 700

times = pd.date_range(start='20140626', end='20140626', freq='6h', tz=tz)

ephem_data = solarposition.get_solarposition(times, latitude, longitude)


# need to add physical tests instead of just functional tests

def test_pres2alt():
    atmosphere.pres2alt(100000)


def test_alt2press():
    atmosphere.pres2alt(1000)


@pytest.mark.parametrize("model",
    ['simple', 'kasten1966', 'youngirvine1967', 'kastenyoung1989',
     'gueymard1993', 'young1994', 'pickering2002'])
def test_airmass(model):
    out = atmosphere.relativeairmass(ephem_data['zenith'], model)
    assert isinstance(out, pd.Series)
    out = atmosphere.relativeairmass(ephem_data['zenith'].values, model)
    assert isinstance(out, np.ndarray)


def test_airmass_scalar():
    assert not np.isnan(atmosphere.relativeairmass(10))


def test_airmass_scalar_nan():
    assert np.isnan(atmosphere.relativeairmass(100))


def test_airmass_invalid():
    with pytest.raises(ValueError):
        atmosphere.relativeairmass(ephem_data['zenith'], 'invalid')


def test_absoluteairmass():
    relative_am = atmosphere.relativeairmass(ephem_data['zenith'], 'simple')
    atmosphere.absoluteairmass(relative_am)
    atmosphere.absoluteairmass(relative_am, pressure=100000)


def test_absoluteairmass_numeric():
    atmosphere.absoluteairmass(2)


def test_absoluteairmass_nan():
    np.testing.assert_equal(np.nan, atmosphere.absoluteairmass(np.nan))


def test_gueymard94_pw():
    temp_air = np.array([0, 20, 40])
    relative_humidity = np.array([0, 30, 100])
    temps_humids = np.array(
        list(itertools.product(temp_air, relative_humidity)))
    pws = atmosphere.gueymard94_pw(temps_humids[:, 0], temps_humids[:, 1])

    expected = np.array(
        [  0.1       ,   0.33702061,   1.12340202,   0.1       ,
         1.12040963,   3.73469877,   0.1       ,   3.44859767,  11.49532557])

    assert_allclose(pws, expected, atol=0.01)


@pytest.mark.parametrize("module_type,expect", [
    ('cdte', np.array(
        [[ 0.9905102 ,  0.9764032 ,  0.93975028],
         [ 1.02928735,  1.01881074,  0.98578821],
         [ 1.04750335,  1.03814456,  1.00623986]])),
    ('monosi', np.array(
        [[ 0.9776977 ,  1.02043409,  1.03574032],
         [ 0.98630905,  1.03055092,  1.04736262],
         [ 0.98828494,  1.03299036,  1.05026561]])),
    ('polysi', np.array(
        [[ 0.9770408 ,  1.01705849,  1.02613202],
         [ 0.98992828,  1.03173953,  1.04260662],
         [ 0.99352435,  1.03588785,  1.04730718]])),
    ('cigs', np.array(
        [[ 0.9745919 ,  1.02821696,  1.05067895],
         [ 0.97529378,  1.02967497,  1.05289307],
         [ 0.97269159,  1.02730558,  1.05075651]])),   
    ('asi', np.array(
        [[ 1.0555275 ,  0.87707583,  0.72243772],
         [ 1.11225204,  0.93665901,  0.78487953],
         [ 1.14555295,  0.97084011,  0.81994083]]))     
])
def test_first_solar_spectral_correction(module_type, expect):
    ams = np.array([1, 3, 5])
    pws = np.array([1, 3, 5])
    ams, pws = np.meshgrid(ams, pws)
    out = atmosphere.first_solar_spectral_correction(pws, ams, module_type)
    assert_allclose(out, expect, atol=0.001)


def test_first_solar_spectral_correction_supplied():
    # use the cdte coeffs
    coeffs = (0.87102, -0.040543, -0.00929202, 0.10052, 0.073062, -0.0034187)
    out = atmosphere.first_solar_spectral_correction(1, 1, coefficients=coeffs)
    expected = 0.99134828
    assert_allclose(out, expected, atol=1e-3)


def test_first_solar_spectral_correction_ambiguous():
    with pytest.raises(TypeError):
        atmosphere.first_solar_spectral_correction(1, 1)


def test_kasten96_lt():
    """Test Linke turbidity factor calculated from AOD, Pwat and AM"""
    amp = np.array([1, 3, 5])
    pwat = np.array([0, 2.5, 5])
    aod_bb = np.array([0, 0.1, 1])
    lt_expected = np.array(
        [[[1.3802, 2.4102, 11.6802],
          [1.16303976, 2.37303976, 13.26303976],
          [1.12101907, 2.51101907, 15.02101907]],

         [[2.95546945, 3.98546945, 13.25546945],
          [2.17435443, 3.38435443, 14.27435443],
          [1.99821967, 3.38821967, 15.89821967]],

         [[3.37410769, 4.40410769, 13.67410769],
          [2.44311797, 3.65311797, 14.54311797],
          [2.23134152, 3.62134152, 16.13134152]]]
    )
    lt = atmosphere.kasten96_lt(*np.meshgrid(amp, pwat, aod_bb))
    assert np.allclose(lt, lt_expected, 1e-3)
    return lt


def test_angstrom_aod():
    """Test Angstrom turbidity model functions."""
    aod550 = 0.15
    aod1240 = 0.05
    alpha = atmosphere.angstrom_alpha(aod550, 550, aod1240, 1240)
    assert np.isclose(alpha, 1.3513924317859232)
    aod700 = atmosphere.angstrom_aod_at_lambda(aod550, 550, alpha)
    assert np.isclose(aod700, 0.10828110997681031)


def test_bird_hulstrom80_aod_bb():
    """Test Bird_Hulstrom broadband AOD."""
    aod380, aod500 = 0.22072480948195175, 0.1614279181106312
    bird_hulstrom = atmosphere.bird_hulstrom80_aod_bb(aod380, aod500)
    assert np.isclose(0.11738229553812768, bird_hulstrom)
