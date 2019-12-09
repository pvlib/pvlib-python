import numpy as np

from pvlib.ivtools.fit_sdm import _update_rsh_fixed_pt


def test_answer():
    outrsh = _update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]))
    assert np.isnan(outrsh)


def test_answer1():
    outrsh1 = _update_rsh_fixed_pt(np.array([0.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]))
    assert np.isnan(outrsh1)


def test_answer2():
    outrsh2 = _update_rsh_fixed_pt(np.array([2.]), np.array([0.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]))
    assert np.isnan(outrsh2)


def test_answer3():
    outrsh3 = _update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                   np.array([0.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]))
    assert np.isnan(outrsh3)


def test_answer4():
    outrsh4 = _update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([0.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]))
    assert np.isnan(outrsh4)


def test_answer5():
    outrsh5 = _update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([0.]), np.array([2.]),
                                   np.array([2.]))
    assert np.isnan(outrsh5)


def test_answer6():
    outrsh6 = _update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([0.]),
                                   np.array([2.]))
    assert np.isnan(outrsh6)


def test_answer7():
    outrsh7 = _update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([2.]), np.array([2.]),
                                   np.array([0.]))
    np.testing.assert_allclose(outrsh7, np.array([502]), atol=.0001)


def test_answer8():
    outrsh8 = _update_rsh_fixed_pt(np.array([-1., 3, .5]),
                                   np.array([1., -.5, 2.]),
                                   np.array([.2, .3, -.4]),
                                    np.array([-.1, 1, 3]),
                                   np.array([4., -.2, .1]),
                                   np.array([.2, .2, -1]),
                                   np.array([0., -1, 0.]))
    assert np.isnan(outrsh8[0])
    assert np.isnan(outrsh8[1])
    assert np.isnan(outrsh8[2])
