from PAN_File.scripts import update_rsh_fixed_pt
import numpy as np

outrsh = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                                 np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer():
    assert np.isnan(outrsh)

outrsh1 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                                  np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer1():
    assert np.isnan(outrsh1)

outrsh2 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]),
                                                  np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer2():
    assert np.isnan(outrsh2)

outrsh3 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]),
                                                  np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer3():
    assert np.isnan(outrsh3)

outrsh4 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]),
                                                  np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer4():
    assert np.isnan(outrsh4)

outrsh5 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                                  np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer5():
    assert np.isnan(outrsh5)

outrsh6 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                                  np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer6():
    assert np.isnan(outrsh6)

outrsh7 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                                  np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer7():
    np.testing.assert_array_almost_equal(outrsh7, np.array([502]), 4)

outrsh8 = update_rsh_fixed_pt.update_rsh_fixed_pt(np.array([-1., 3, .5]), np.array([1., -.5, 2.]),
                                                  np.array([.2, .3, -.4]), np.array([-.1, 1, 3]),
                                                  np.array([4., -.2, .1]), np.array([.2, .2, -1]),
                                                  np.array([0., -1, 0.]))


def test_answer8():
    assert np.isnan(outrsh8[0])
    assert np.isnan(outrsh8[1])
    assert np.isnan(outrsh8[2])
