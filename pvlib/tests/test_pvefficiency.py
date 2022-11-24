from numpy.testing import assert_allclose

from pvlib import pvefficiency


def test_adr():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    p = [1.0, -6.684898, 0.018855, 0.069917, 0.054369]

    e = [1.0, 0.949154, 0.92812, 0.876472, 0.855699, 0.80325]

    result = pvefficiency.adr(g, t, *p)
    assert_allclose(result, e, rtol=1e-5)


def test_fit_pvefficiency_adr():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    e = [1.0, 0.949154, 0.92812, 0.876472, 0.855699, 0.80325]

    p = [1.0, -6.684898, 0.018855, 0.069917, 0.054369]

    result = pvefficiency.fit_pvefficiency_adr(g, t, e, dict_output=False)
    # the fitted parameters vary somewhat by platform during the testing
    # so the tolerance is higher on the parameters than on the efficiencies
    # in the other tests
    assert_allclose(result, p, rtol=1e-3)

    result = pvefficiency.fit_pvefficiency_adr(g, t, e, dict_output=True)
    assert 'k_a' in result


def test_adr_round_trip():
    g = [1000, 200, 1000, 200, 1000, 200]
    t = [25, 25, 50, 50, 75, 75]
    e = [1.0, 0.949154, 0.92812, 0.876472, 0.855699, 0.80325]

    p = pvefficiency.fit_pvefficiency_adr(g, t, e, dict_output=False)
    result = pvefficiency.adr(g, t, *p)
    assert_allclose(result, e, rtol=1e-5)
