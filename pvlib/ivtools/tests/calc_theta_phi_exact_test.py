from pvlib.ivtools.calc_theta_phi_exact import calc_theta_phi_exact
import numpy as np


def test_answer():
    [theta, phi] = calc_theta_phi_exact(np.array([2.]), np.array([2.]),
                                        np.array([2.]), np.array([2.]),
                                        np.array([2.]), np.array([2.]),
                                        np.array([2.]))
    np.testing.assert_allclose(theta, np.array([1.8726]), atol=.0001)
    np.testing.assert_allclose(phi, np.array([2]), atol=.0001)


def test_answer1():
    [theta1, phi1] = calc_theta_phi_exact(np.array([0.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]))
    np.testing.assert_allclose(theta1, np.array([1.8726]), atol=.0001)
    np.testing.assert_allclose(phi1, np.array([3.4537]), atol=.0001)


def test_answer2():
    [theta2, phi2] = calc_theta_phi_exact(np.array([2.]), np.array([0.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]))
    np.testing.assert_allclose(theta2, np.array([1.2650]), atol=.0001)
    np.testing.assert_allclose(phi2, np.array([0.8526]), atol=.0001)


def test_answer3():
    [theta3, phi3] = calc_theta_phi_exact(np.array([2.]), np.array([2.]),
                                          np.array([0.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]))
    np.testing.assert_allclose(theta3, np.array([1.5571]), atol=.0001)
    np.testing.assert_allclose(phi3, np.array([2]), atol=.0001)


def test_answer4():
    [theta4, phi4] = calc_theta_phi_exact(np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([0.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]))
    assert np.isnan(theta4)
    assert np.isnan(phi4)


def test_answer5():
    [theta5, phi5] = calc_theta_phi_exact(np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([0.]), np.array([2.]),
                                          np.array([2.]))
    assert np.isnan(theta5)
    assert np.isnan(phi5)


def test_answer6():
    [theta6, phi6] = calc_theta_phi_exact(np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([0.]),
                                          np.array([2.]))
    assert np.isnan(theta6)
    np.testing.assert_allclose(phi6, np.array([2]), atol=.0001)


def test_answer7():
    [theta7, phi7] = calc_theta_phi_exact(np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([2.]), np.array([2.]),
                                          np.array([0.]))
    assert np.isnan(theta7)
    assert np.isnan(phi7)


def test_answer8():
    [theta8, phi8] = calc_theta_phi_exact(np.array([1., -1.]),
                                          np.array([-1., 1.]),
                                          np.array([1., -1.]),
                                          np.array([-1., 1.]),
                                          np.array([1., -1.]),
                                          np.array([-1., 1.]),
                                          np.array([1., -1.]))
    assert np.isnan(theta8[0])
    assert np.isnan(theta8[1])
    assert np.isnan(phi8[0])
    np.testing.assert_allclose(phi8[1], np.array([2.2079]), atol=.0001)
