import numpy as np
from numpy import nan
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
    # tests with default k_version='pvgis5'
    pdc0 = 100
    res = pvarray.huld(1000, 25, pdc0, cell_type='cSi')
    assert np.isclose(res, pdc0)
    k = pvarray._infer_k_huld('cSi', pdc0, 'pvgis5')
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
                       match='Either k or cell_type must be specified'
                       ):
        pvarray.huld(1000, 25, 100)


def test_huld_params():
    """Test Huld with built-in coefficients."""
    pdc0 = 100
    # Use non-reference values so coefficients affect the result
    eff_irr = 800  # W/m^2 (not 1000)
    temp_mod = 35  # deg C (not 25)
    # calculated by C. Hansen using Excel, 2025
    expected = {'pvgis5': {'csi': 76.405089,
                           'cis': 77.086016,
                           'cdte': 78.642762
                           },
                'pvgis6': {'csi': 77.649421,
                           'cis': 77.723110,
                           'cdte': 77.500399
                           }
                }
    # Test with PVGIS5 coefficients for all cell types
    for yr in expected:
        for cell_type in expected[yr]:
            result = pvarray.huld(eff_irr, temp_mod, pdc0, cell_type=cell_type,
                                  k_version=yr)
            assert np.isclose(result, expected[yr][cell_type])


def test_huld_errors():
    # Check errors
    pdc0 = 100
    # Use non-reference values so coefficients affect the result
    eff_irr = 800  # W/m^2 (not 1000)
    temp_mod = 35  # deg C (not 25)
    # provide both cell_type and k_version
    with pytest.raises(KeyError):
        pvarray.huld(
            eff_irr, temp_mod, pdc0, cell_type='invalid', k_version='pvgis5'
        )
    # provide invalid k_version
    with pytest.raises(ValueError, match='Invalid k_version=2021'):
        pvarray.huld(
            eff_irr, temp_mod, pdc0, cell_type='csi', k_version='2021'
        )


def test_batzelis():
    params = {'i_sc': 15.98, 'v_oc': 50.26, 'i_mp': 15.27, 'v_mp': 42.57,
              'alpha_sc': 0.007351, 'beta_voc': -0.120624}
    g = np.array([1000, 500, 1200, 500, 1200, 0, nan, 1000])
    t = np.array([25, 20, 20, 50, 50, 25, 0, nan])
    expected = {  # these values were computed using pvarray.batzelis itself
        'p_mp': [650.044, 328.599, 789.136, 300.079, 723.401, 0, nan, nan],
        'i_mp': [ 15.270, 7.626, 18.302, 7.680, 18.433, 0, nan, nan],
        'v_mp': [ 42.570, 43.090, 43.117, 39.071, 39.246, 0, nan, nan],
        'i_sc': [ 15.980, 7.972, 19.132, 8.082, 19.397, 0, nan, nan],
        'v_oc': [ 50.260, 49.687, 51.172, 45.948, 47.585, 0, nan, nan],
    }

    # numpy array
    actual = pvarray.batzelis(g, t, **params)
    for key, exp in expected.items():
        np.testing.assert_allclose(actual[key], exp, atol=1e-3)

    # pandas series
    actual = pvarray.batzelis(pd.Series(g), pd.Series(t), **params)
    assert isinstance(actual, pd.DataFrame)
    for key, exp in expected.items():
        np.testing.assert_allclose(actual[key], pd.Series(exp), atol=1e-3)

    # scalar
    actual = pvarray.batzelis(g[1], t[1], **params)
    for key, exp in expected.items():
        assert pytest.approx(exp[1], abs=1e-3) == actual[key]


def test_batzelis_negative_voltage():
    params = {'i_sc': 15.98, 'v_oc': 50.26, 'i_mp': 15.27, 'v_mp': 42.57,
              'alpha_sc': 0.007351, 'beta_voc': -0.120624}
    actual = pvarray.batzelis(1e-10, 25, **params)
    assert actual['v_mp'] == 0
    assert actual['v_oc'] == 0
