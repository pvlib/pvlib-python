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
        [[ 0.99134828,  0.97701063,  0.93975103],
         [ 1.02852847,  1.01874908,  0.98604776],
         [ 1.04722476,  1.03835703,  1.00656735]])),
    ('monosi', np.array(
        [[ 0.9782842 ,  1.02092726,  1.03602157],
         [ 0.9859024 ,  1.0302268 ,  1.04700244],
         [ 0.98885429,  1.03351495,  1.05062687]])),
    ('polysi', np.array(
        [[ 0.9774921 ,  1.01757872,  1.02649543],
         [ 0.98947361,  1.0314545 ,  1.04226547],
         [ 0.99403107,  1.03639082,  1.04758064]]))
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
    np.isclose(alpha, 1.3513924317859232)
    aod700 = atmosphere.angstrom_aod_at_lambda(aod550, 550, alpha)
    np.isclose(aod700, 0.10828110997681031)


def test_bird_hulstrom80_aod_bb():
    """Test Bird_Hulstrom broadband AOD."""
    aod380, aod500 = 0.22072480948195175, 0.1614279181106312
    bird_hulstrom = atmosphere.bird_hulstrom80_aod_bb(aod380, aod500)
    np.isclose(0.09823143641608373, bird_hulstrom)


MELBOURNE_FL = (
    ['1999-01-31T12:00:00-05:00', '2000-02-20T15:00:00-05:00',
     '2000-02-22T13:00:00-05:00', '2000-02-24T15:00:00-05:00',
     '1995-03-02T14:00:00-05:00', '1995-03-11T12:00:00-05:00',
     '1995-03-12T13:00:00-05:00', '1995-03-20T11:00:00-05:00',
     '1995-03-20T14:00:00-05:00', '1995-03-22T11:00:00-05:00',
     '1995-03-22T14:00:00-05:00', '1995-04-07T09:00:00-05:00',
     '1995-04-10T09:00:00-05:00', '1995-04-21T09:00:00-05:00',
     '2004-05-01T08:00:00-05:00', '2004-05-03T08:00:00-05:00',
     '2004-05-05T13:00:00-05:00', '2004-05-16T09:00:00-05:00',
     '2004-05-21T15:00:00-05:00', '2004-05-24T11:00:00-05:00',
     '2004-05-31T09:00:00-05:00', '2002-06-04T16:00:00-05:00',
     '2002-06-16T17:00:00-05:00', '2002-06-17T08:00:00-05:00',
     '2000-07-01T08:00:00-05:00', '2000-07-05T07:00:00-05:00',
     '2000-07-06T07:00:00-05:00', '2000-07-06T17:00:00-05:00',
     '2000-07-25T15:00:00-05:00', '2001-10-14T13:00:00-05:00',
     '2001-10-15T11:00:00-05:00', '2003-11-04T14:00:00-05:00',
     '1999-12-20T13:00:00-05:00'],
    {'AOD': [0.062,  0.082,  0.084,  0.087,  0.097,  0.110,  0.111,  0.119,
             0.119,  0.121,  0.121,  0.128,  0.129,  0.137,  0.150,  0.152,
             0.154,  0.164,  0.167,  0.169,  0.171,  0.173,  0.188,  0.191,
             0.237,  0.248,  0.251,  0.251,  0.244,  0.119,  0.118,  0.101,
             0.085],
     'Pwat': [2.7,  2.2,  1.6,  2.3,  1.9,  1.9,  1.9,  2.0,  2.0,  1.9,  1.9,
              2.6,  2.5,  2.6,  4.0,  4.1,  1.8,  3.6,  3.0,  2.3,  3.7,  4.1,
              4.9,  4.6,  4.2,  4.3,  4.3,  4.3,  4.6,  4.8,  4.9,  4.7,  2.7],
     'Pressure': [101500.,  101700.,  101200.,  101900.,  101700.,  102900.,
                  102800.,  101600.,  101400.,  101400.,  101200.,  101200.,
                  101400.,  101700.,  101700.,  101500.,  101900.,  102200.,
                  102000.,  101800.,  101500.,  101500.,  101200.,  101400.,
                  101800.,  102000.,  101900.,  101700.,  101700.,  101300.,
                  101700.,  101500.,  101400.],
     'DryBulb': [23.0,  19.0,  24.0,  23.0,  18.3,  21.6,  23.3,  22.7,  24.4,
                 24.4,  28.3,  19.4,  25.0,  27.2,  25.0,  25.0,  25.0,  25.0,
                 28.0,  28.0,  31.0,  29.0,  24.0,  24.0,  25.0,  26.0,  23.0,
                 32.0,  29.0,  29.0,  28.0,  26.6,  20.0]}
)
