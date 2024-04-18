import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from .conftest import assert_series_equal
import pytest
from pvlib.tests.conftest import requires_statsmodels
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
    exp_sum = np.exp(1) * (np.sum(pvarray._infer_k_huld('cSi', pdc0)) + pdc0)
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


@pytest.mark.parametrize('method', ['ols', 'robust'])
@requires_statsmodels
def test_fit_huld(method):
    # test is to recover the parameters in _infer_huld_k for each cell type
    # IEC61853 conditions to make data for fitting
    ee, tc = pvarray._build_iec61853()
    techs = ['csi', 'cis', 'cdte']
    pdc0 = 250
    for tech in techs:
        k0 = pvarray._infer_k_huld(tech, pdc0)
        pdc = pvarray.huld(ee, tc, pdc0, cell_type=tech)
        m_pdc0, k = pvarray.fit_huld(ee, tc, pdc, method=method)
        expected = np.array([pdc0, ] + [v for v in k0], dtype=float)
        modeled = np.hstack((m_pdc0, k))
        assert_allclose(expected, modeled, rtol=1e-8)
    # once more to check that NaNs are handled
    ee[7] = np.nan
    tc[9] = np.nan
    k0 = pvarray._infer_k_huld('csi', pdc0)
    pdc = pvarray.huld(ee, tc, pdc0, cell_type='csi')
    pdc[11] = np.nan
    m_pdc0, k = pvarray.fit_huld(ee, tc, pdc, method='ols')
    expected = np.array([pdc0, ] + [v for v in k0], dtype=float)
    modeled = np.hstack((m_pdc0, k))
    assert_allclose(expected, modeled, rtol=1e-8)


@requires_statsmodels
def test_fit_huld_method_error():
    ee, tc = pvarray._build_iec61853()
    pdc0 = 250
    pdc = pvarray.huld(ee, tc, pdc0, cell_type='csi')
    method = 'brute_force'
    with pytest.raises(ValueError, match="method must be ols or robust"):
        m_pdc0, k = pvarray.fit_huld(ee, tc, pdc, method=method)
