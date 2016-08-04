from pvlib.calc_theta_phi_exact import calc_theta_phi_exact
import numpy as np

[theta, phi] = calc_theta_phi_exact(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                    np.array([2.]), np.array([2.]))


def test_answer():
    np.testing.assert_array_almost_equal(theta, np.array([1.8726]), 4)
    np.testing.assert_array_almost_equal(phi, np.array([2]), 4)

[theta1, phi1] = calc_theta_phi_exact(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                      np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer1():
    np.testing.assert_array_almost_equal(theta1, np.array([1.8726]), 4)
    np.testing.assert_array_almost_equal(phi1, np.array([3.4537]), 4)

[theta2, phi2] = calc_theta_phi_exact(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]),
                                      np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer2():
    np.testing.assert_array_almost_equal(theta2, np.array([1.2650]), 4)
    np.testing.assert_array_almost_equal(phi2, np.array([0.8526]), 4)

[theta3, phi3] = calc_theta_phi_exact(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]),
                                      np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer3():
    np.testing.assert_array_almost_equal(theta3, np.array([1.5571]), 4)
    np.testing.assert_array_almost_equal(phi3, np.array([2]), 4)

[theta4, phi4] = calc_theta_phi_exact(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]),
                                      np.array([2.]), np.array([2.]), np.array([2.]))


def test_answer4():
    assert np.isnan(theta4)
    assert np.isnan(phi4)

[theta5, phi5] = calc_theta_phi_exact(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                      np.array([0.]), np.array([2.]), np.array([2.]))


def test_answer5():
    assert np.isnan(theta5)
    assert np.isnan(phi5)

[theta6, phi6] = calc_theta_phi_exact(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                      np.array([2.]), np.array([0.]), np.array([2.]))


def test_answer6():
    assert np.isnan(theta6)
    np.testing.assert_array_almost_equal(phi6, np.array([2]), 4)

[theta7, phi7] = calc_theta_phi_exact(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                      np.array([2.]), np.array([2.]), np.array([0.]))


def test_answer7():
    assert np.isnan(theta7)
    assert np.isnan(phi7)

[theta8, phi8] = calc_theta_phi_exact(np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]),
                                      np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., 1.]),
                                      np.array([1., -1.]))


def test_answer8():
    assert np.isnan(theta8[0])
    assert np.isnan(theta8[1])
    assert np.isnan(phi8[0])
    np.testing.assert_array_almost_equal(phi8[1], np.array([2.2079]), 4)
