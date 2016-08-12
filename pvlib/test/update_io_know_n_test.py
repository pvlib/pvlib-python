from pvlib.update_io_known_n import update_io_known_n
import numpy as np


outio = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                          np.array([2.]))


def test_answer9():
    np.testing.assert_array_almost_equal(outio, np.array([0.5911]), 4)

outio1 = update_io_known_n(np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                           np.array([2.]))


def test_answer10():
    np.testing.assert_array_almost_equal(outio1, np.array([1.0161e-4]), 4)

outio2 = update_io_known_n(np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]), np.array([2.]),
                           np.array([2.]))


def test_answer11():
    np.testing.assert_array_almost_equal(outio2, np.array([0.5911]), 4)

outio3 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]), np.array([2.]),
                           np.array([2.]))


def test_answer12():
    assert np.isnan(outio3)

outio4 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]), np.array([2.]),
                           np.array([2.]))


def test_answer13():
    np.testing.assert_array_almost_equal(outio4, np.array([0]), 4)

outio5 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([0.]),
                           np.array([2.]))


def test_answer14():
    np.testing.assert_array_almost_equal(outio5, np.array([1.0161e-4]), 4)

outio6 = update_io_known_n(np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]), np.array([2.]),
                           np.array([0.]))


def test_answer15():
    np.testing.assert_array_almost_equal(outio6, np.array([17.9436]), 4)

outio7 = update_io_known_n(np.array([-1.]), np.array([-1.]), np.array([-1.]), np.array([-1.]), np.array([-1.]),
                           np.array([-1.]))


def test_answer16():
    np.testing.assert_array_almost_equal(outio7, np.array([.1738]), 4)
