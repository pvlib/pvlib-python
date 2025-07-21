import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from .conftest import assert_series_equal
import pytest

from pvlib import pvarray


def test_pvefficiency_adr():
    g = [1000, 200, 1000, 200, 1000, 200, 0.0, np.nan]
    t = [25, 25, 50, 50, 75, 75, 25, 25]
    params = [1.0, -6.651460, 0.018736, 0.070679, 0.054170]

    # the expected values were calculated using the new function itself
    # hence this test is primarily a regression test
    eta = [1.0, 0.949125, 0.928148, 0.876472, 0.855759, 0.803281, 0.0, np.nan]

    result = pvarray.pvefficiency_adr(g, t, *params)
    assert_allclose(result, eta, atol=1e-6)


def test_fit_pvefficiency_adr():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    eta = [1.0, 0.949125, 0.928148, 0.876472, 0.855759, 0.803281]

    # the expected values were calculated using the new function itself
    # hence this test is primarily a regression test
    params = [1.0, -6.651460, 0.018736, 0.070679, 0.054170]

    result = pvarray.fit_pvefficiency_adr(g, t, eta, dict_output=False)
    # the fitted parameters vary somewhat by platform during the testing
    # so the tolerance is higher on the parameters than on the efficiencies
    # in the other tests
    assert_allclose(result, params, rtol=1e-3)

    result = pvarray.fit_pvefficiency_adr(g, t, eta, dict_output=True)
    assert 'k_a' in result


def test_pvefficiency_adr_round_trip():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    eta = [1.0, 0.949125, 0.928148, 0.876472, 0.855759, 0.803281]

    params = pvarray.fit_pvefficiency_adr(g, t, eta, dict_output=False)
    result = pvarray.pvefficiency_adr(g, t, *params)
    assert_allclose(result, eta, atol=1e-6)


def test_huld():
    pdc0 = 100
    res = pvarray.huld(1000, 25, pdc0, cell_type='cSi')
    assert np.isclose(res, pdc0)
    k = pvarray._infer_k_huld('cSi', pdc0, '2011')
    exp_sum = np.exp(1) * (np.sum(k) + pdc0)
    res = pvarray.huld(1000*np.exp(1), 26, pdc0, cell_type='cSi')
    assert np.isclose(res, exp_sum)
    res = pvarray.huld(100, 30, pdc0, k=(1, 1, 1, 1, 1, 1))
    exp_100 = 0.1 * (pdc0 + np.log(0.1) + np.log(0.1)**2 + 5 + 5*np.log(0.1)
                     + 5*np.log(0.1)**2 + 25)
    assert np.isclose(res, exp_100)
    # Series input, and irradiance = 0
    eff_irr = pd.Series([1000, 100, 0])
    tm = pd.Series([25, 30, 30])
    expected = pd.Series([pdc0, exp_100, 0])
    res = pvarray.huld(eff_irr, tm, pdc0, k=(1, 1, 1, 1, 1, 1))
    assert_series_equal(res, expected)
    with pytest.raises(ValueError,
                       match='Either k or cell_type must be specified'):
        res = pvarray.huld(1000, 25, 100)


def test_huld_params():
    """Test Huld with built-in coefficients."""
    pdc0 = 100
    # Use non-reference values so coefficients affect the result
    eff_irr = 800  # W/m^2 (not 1000)
    temp_mod = 35  # deg C (not 25)
    # calculated by C. Hansen using Excel, 2025
    expected = {'2011': {'csi': 76.405089,
                         'cis': 77.086016,
                         'cdte': 78.642762
                         },
                '2025': {'csi': 76.421390,
                         'cis': 77.095102,
                         'cdte': 78.648450
                         }
                }
    # Test with 2011 coefficients for all cell types
    for yr in expected:
        for cell_type in expected[yr]:
            result = pvarray.huld(eff_irr, temp_mod, pdc0, cell_type=cell_type,
                                  k_version=yr)
            assert np.isclose(result, expected[yr][cell_type])
    # Check errors for incorrect cell_type and incorrect k_version
    with pytest.raises(KeyError):
        pvarray.huld(
            eff_irr, temp_mod, pdc0, cell_type='invalid', k_version='2011'
        )
    with pytest.raises(ValueError, match='Invalid k_version="2021"'):
        pvarray.huld(
            eff_irr, temp_mod, pdc0, cell_type='csi', k_version='2021'
        )
