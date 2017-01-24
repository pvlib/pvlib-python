import datetime
import itertools

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose

from pvlib.location import Location
from pvlib import solarposition
from pvlib import atmosphere


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


def test_linke_turbidity_kasten_pyrheliometric_formula():
    # Linke turbidity factor calculated from AOD, Pwat and AM

    # from datetime import datetime
    # import pvlib
    # from solar_utils import *
    # from matplotlib import pyplot as plt
    # import seaborn as sns
    # import os
    #
    # plt.ion()
    # sns.set_context(rc={'figure.figsize': (12, 8)})

    def demo_kasten_96lt():
        atmos_path = os.path.dirname(os.path.abspath(__file__))
        pvlib_path = os.path.dirname(atmos_path)
        melbourne_fl = pvlib.tmy.readtmy3(os.path.join(
            pvlib_path, 'data', '722040TYA.CSV')
        )
        aod700 = melbourne_fl[0]['AOD']
        pwat_cm = melbourne_fl[0]['Pwat']
        press_mbar = melbourne_fl[0]['Pressure']
        dry_temp = melbourne_fl[0]['DryBulb']
        timestamps = melbourne_fl[0].index
        location = (melbourne_fl[1]['latitude'],
                    melbourne_fl[1]['longitude'],
                    melbourne_fl[1]['TZ'])
        _, airmass = zip(*[solposAM(
            location, d.timetuple()[:6], (press_mbar.loc[d], dry_temp.loc[d])
        ) for d in timestamps])
        _, amp = zip(*airmass)
        amp = np.array(amp)
        filter = amp < 0
        amp[filter] = np.nan
        lt_molineaux = kasten_96lt(aod=[(700.0, aod700)], am=amp, pwat=pwat_cm)
        lt_bird_huldstrom = kasten_96lt(aod=[(700.0, aod700)], am=amp,
                                        pwat=pwat_cm,
                                        method='Bird-Huldstrom')
        t = pd.DatetimeIndex(
            [datetime.replace(d, year=2016) for d in timestamps])
        lt_molineaux.index = t
        lt_bird_huldstrom.index = t
        pvlib.clearsky.lookup_linke_turbidity(t, *location[:2]).plot()
        lt_molineaux.resample('D').mean().plot()
        lt_bird_huldstrom.resample('D').mean().plot()
        title = [
            'Linke turbidity factor comparison at Melbourne, FL (722040TYA),',
            'calculated using Kasten Pyrheliometric formula']
        plt.title(' '.join(title))
        plt.ylabel('Linke turbidity factor')
        plt.legend(['Linke Turbidity', 'Molineaux', 'Bird-Huldstrom'])
        return lt_molineaux, lt_bird_huldstrom

    lt_molineaux, lt_bird_huldstrom = demo_kasten_96lt()
