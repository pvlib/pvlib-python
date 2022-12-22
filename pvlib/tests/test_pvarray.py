import numpy as np
from numpy.testing import assert_allclose

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
