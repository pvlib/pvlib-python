from pvlib import update_io_known_n
import numpy as np

v = update_io_known_n.v_from_i(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                               np.array([2.]))


def test_answer():
    np.testing.assert_array_almost_equal(v, np.array([-4]), 4)

v1 = update_io_known_n.v_from_i(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                np.array([2.]))


def test_answer1():
    np.testing.assert_array_almost_equal(v1, np.array([-4]), 4)

v2 = update_io_known_n.v_from_i(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                np.array([2.]))


def test_answer2():
    np.testing.assert_array_almost_equal(v2, np.array([0]), 4)

v3 = update_io_known_n.v_from_i(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]),
                                np.array([2.]))


def test_answer3():
    assert np.isnan(v3)

v4 = update_io_known_n.v_from_i(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]),
                                np.array([2.]))


def test_answer4():
    np.testing.assert_array_almost_equal(v4, np.array([1.0926]), 4)

v5 = update_io_known_n.v_from_i(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]),
                                np.array([2.]))


def test_answer5():
    np.testing.assert_array_almost_equal(v5, np.array([-4]), 4)

v6 = update_io_known_n.v_from_i(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                np.array([0.]))


def test_answer6():
    np.testing.assert_array_almost_equal(v6, np.array([-5.7052]), 4)

v8 = update_io_known_n.v_from_i(np.array([-1., 1.]), np.array([1., -1.]), np.array([-1., 1.]), np.array([1., -1.]),
                                np.array([-1., 1.]), np.array([1., -1.]))


def test_answer8():
    np.testing.assert_array_almost_equal(v8, np.array([-1, -1]), 4)

outio = update_io_known_n.update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                            np.array([2.]), np.array([2.]))


def test_answer9():
    np.testing.assert_array_almost_equal(outio, np.array([0.5911]), 4)

outio1 = update_io_known_n.update_io_known_n(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                             np.array([2.]), np.array([2.]))


def test_answer10():
    np.testing.assert_array_almost_equal(outio1, np.array([1.0161e-4]), 4)

outio2 = update_io_known_n.update_io_known_n(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]),
                                             np.array([2.]), np.array([2.]))


def test_answer11():
    np.testing.assert_array_almost_equal(outio2, np.array([0.5911]), 4)

outio3 = update_io_known_n.update_io_known_n(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]),
                                             np.array([2.]), np.array([2.]))


def test_answer12():
    assert np.isnan(outio3)

outio4 = update_io_known_n.update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]),
                                             np.array([2.]), np.array([2.]))


def test_answer13():
    np.testing.assert_array_almost_equal(outio4, np.array([0]), 4)

outio5 = update_io_known_n.update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                             np.array([0.]), np.array([2.]))


def test_answer14():
    np.testing.assert_array_almost_equal(outio5, np.array([1.0161e-4]), 4)

outio6 = update_io_known_n.update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                                             np.array([2.]), np.array([0.]))


def test_answer15():
    np.testing.assert_array_almost_equal(outio6, np.array([17.9436]), 4)

outio7 = update_io_known_n.update_io_known_n(np.array([-1.]), np.array([-1.]), np.array([-1.]), np.array([-1.]),
                                             np.array([-1.]), np.array([-1.]))


def test_answer16():
    np.testing.assert_array_almost_equal(outio7, np.array([.1738]), 4)
