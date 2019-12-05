from pvlib.ivtools.update_rsh_fixed_pt import update_rsh_fixed_pt
import numpy as np


def test_answer():
    outrsh = update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                 np.array([2.]), np.array([2.]),
                                 np.array([2.]), np.array([2.]),
                                 np.array([2.]))
    assert np.isnan(outrsh)


def test_answer1():
    outrsh1 = update_rsh_fixed_pt(np.array([0.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]))
    assert np.isnan(outrsh1)


def test_answer2():
    outrsh2 = update_rsh_fixed_pt(np.array([2.]), np.array([0.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]))
    assert np.isnan(outrsh2)


def test_answer3():
    outrsh3 = update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                  np.array([0.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]))
    assert np.isnan(outrsh3)


def test_answer4():
    outrsh4 = update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([0.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]))
    assert np.isnan(outrsh4)


def test_answer5():
    outrsh5 = update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([0.]), np.array([2.]),
                                  np.array([2.]))
    assert np.isnan(outrsh5)


def test_answer6():
    outrsh6 = update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([0.]),
                                  np.array([2.]))
    assert np.isnan(outrsh6)


def test_answer7():
    outrsh7 = update_rsh_fixed_pt(np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([2.]), np.array([2.]),
                                  np.array([0.]))
    np.testing.assert_allclose(outrsh7, np.array([502]), atol=.0001)


def test_answer8():
    outrsh8 = update_rsh_fixed_pt(np.array([-1., 3, .5]),
                                  np.array([1., -.5, 2.]),
                                  np.array([.2, .3, -.4]),
                                  np.array([-.1, 1, 3]),
                                  np.array([4., -.2, .1]),
                                  np.array([.2, .2, -1]),
                                  np.array([0., -1, 0.]))
    assert np.isnan(outrsh8[0])
    assert np.isnan(outrsh8[1])
    assert np.isnan(outrsh8[2])
